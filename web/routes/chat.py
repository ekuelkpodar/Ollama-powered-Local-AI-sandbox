"""Chat API routes — send messages, stream responses via SSE."""

import asyncio
import json
import uuid
import time
from flask import Blueprint, request, jsonify, Response, current_app

from web.app import SessionState
from agent.session_store import SessionStore
from agent.models import OllamaClient

chat_bp = Blueprint("chat", __name__)


@chat_bp.route("/chat/send", methods=["POST"])
def send_message():
    """Send a user message and start the agent monologue."""
    data = request.get_json()
    message = data.get("message", "").strip()
    session_id = data.get("session_id")

    if not message:
        return jsonify({"error": "No message provided"}), 400

    config = current_app.config["agent_config"]
    sessions = current_app.config["sessions"]

    # Get or create session
    if session_id and session_id in sessions:
        session = sessions[session_id]
    else:
        ok, error = _preflight_ollama(config)
        if not ok:
            return jsonify({"error": error}), 503

        if session_id and config.session.persist:
            store = _get_session_store(current_app)
            if store and not store.get_session(session_id):
                return jsonify({"error": "Session not found"}), 404
            session = SessionState(config, session_id=session_id)
            sessions[session_id] = session
        else:
            session_id = uuid.uuid4().hex[:12]
            session = SessionState(config, session_id=session_id)
            sessions[session_id] = session

    if session.is_running:
        return jsonify({"error": "Agent is already processing"}), 409

    # Start the monologue
    session.send_message(message)

    return jsonify({"session_id": session_id, "status": "processing"})


@chat_bp.route("/chat/stream/<session_id>")
def stream_response(session_id):
    """SSE endpoint — streams agent chunks in real-time."""
    sessions = current_app.config["sessions"]
    session = sessions.get(session_id)

    if session is None:
        return jsonify({"error": "Session not found"}), 404

    def generate():
        while True:
            try:
                event = session.stream_queue.get(timeout=30)
                yield f"data: {json.dumps(event)}\n\n"
                if event.get("type") in ("done", "error"):
                    break
            except Exception:
                # Timeout — send keepalive
                yield f"data: {json.dumps({'type': 'keepalive'})}\n\n"
                if not session.is_running:
                    break

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@chat_bp.route("/chat/history/<session_id>")
def get_history(session_id):
    """Get conversation history for a session."""
    sessions = current_app.config["sessions"]
    session = sessions.get(session_id)

    if session is None:
        store = _get_session_store(current_app)
        if store:
            data = store.get_session(session_id)
            if data:
                return jsonify({
                    "history": data["messages"],
                    "tool_calls": data["tool_calls"],
                    "title": data.get("title"),
                })
        return jsonify({"error": "Session not found"}), 404

    agent = session.context.get_agent(0)
    if agent is None:
        return jsonify({"history": []})

    return jsonify({
        "history": agent.history,
        "tool_calls": session.context.data.get("tool_calls", []),
        "title": session.context.data.get("session_title"),
    })


@chat_bp.route("/chat/sessions", methods=["GET"])
def list_sessions():
    """List all active sessions."""
    sessions = current_app.config["sessions"]
    result = []
    for sid, session in sessions.items():
        agent = session.context.get_agent(0)
        msg_count = len(agent.history) if agent else 0
        title = session.context.data.get("session_title")
        created_at = session.context.data.get("session_created_at")
        updated_at = session.context.data.get("session_updated_at")
        result.append({
            "session_id": sid,
            "title": title,
            "created_at": created_at,
            "updated_at": updated_at,
            "message_count": msg_count,
            "is_running": session.is_running,
            "persisted": False,
        })

    store = _get_session_store(current_app)
    if store:
        persisted = store.list_sessions()
        active_ids = {s["session_id"] for s in result}
        for s in persisted:
            if s["session_id"] in active_ids:
                continue
            result.append({
                "session_id": s["session_id"],
                "title": s.get("title"),
                "created_at": s.get("created_at"),
                "message_count": s["message_count"],
                "is_running": False,
                "persisted": True,
                "updated_at": s["updated_at"],
                "model": s["model"],
            })

    return jsonify({"sessions": result})


@chat_bp.route("/chat/session/<session_id>", methods=["DELETE"])
def delete_session(session_id):
    """Delete a session."""
    sessions = current_app.config["sessions"]
    if session_id in sessions:
        del sessions[session_id]
        if _delete_persisted_session(current_app, session_id):
            return jsonify({"status": "deleted"})
        return jsonify({"status": "deleted"})

    if _delete_persisted_session(current_app, session_id):
        return jsonify({"status": "deleted"})

    return jsonify({"error": "Session not found"}), 404


@chat_bp.route("/chat/session/<session_id>", methods=["PATCH"])
def rename_session(session_id):
    """Rename a session."""
    data = request.get_json() or {}
    title = (data.get("title") or "").strip()
    if not title:
        return jsonify({"error": "Title is required"}), 400

    sessions = current_app.config["sessions"]
    session = sessions.get(session_id)

    if session:
        session.context.data["session_title"] = title
        session.context.data["session_updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())

    store = _get_session_store(current_app)
    persisted = False
    if store:
        persisted = store.get_session(session_id) is not None
        if persisted:
            store.set_title(session_id, title)

    if session or persisted:
        return jsonify({"status": "renamed", "title": title})

    return jsonify({"error": "Session not found"}), 404


def _run_async(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _preflight_ollama(config) -> tuple[bool, str]:
    base_url_map: dict[str, list[str]] = {}

    def _add_model(base_url: str, model_name: str) -> None:
        models = base_url_map.setdefault(base_url, [])
        if model_name not in models:
            models.append(model_name)

    _add_model(config.chat_model.base_url, config.chat_model.model_name)
    _add_model(config.utility_model.base_url, config.utility_model.model_name)

    for base_url, required_models in base_url_map.items():
        client = OllamaClient(
            base_url=base_url,
            connect_timeout=config.ollama.connect_timeout,
            read_timeout=config.ollama.read_timeout,
            max_retries=config.ollama.max_retries,
        )

        if config.ollama.health_check_on_start:
            healthy = _run_async(client.health_check())
            if not healthy:
                return False, f"Cannot connect to Ollama at {base_url}. Is ollama running?"

        try:
            models = _run_async(client.list_models())
        except Exception as e:
            return False, str(e)

        model_names = [m.get("name", "") for m in models]
        missing = OllamaClient.filter_missing_models(required_models, model_names)
        if missing:
            missing_list = ", ".join(missing)
            return False, f"Missing models at {base_url}: {missing_list}. Pull with: ollama pull <model>"

    return True, ""


def _get_session_store(app) -> SessionStore | None:
    config = app.config["agent_config"]
    if not config.session.persist:
        return None
    return SessionStore(
        db_path=config.session.storage_path,
        max_sessions=config.session.max_sessions,
    )


def _delete_persisted_session(app, session_id: str) -> bool:
    store = _get_session_store(app)
    if not store:
        return False
    return store.delete_session(session_id)
