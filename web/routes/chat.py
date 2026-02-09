"""Chat API routes — send messages, stream responses via SSE."""

import json
import uuid
import time
from flask import Blueprint, request, jsonify, Response, current_app

from web.app import SessionState

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
        session_id = uuid.uuid4().hex[:12]
        session = SessionState(config)
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
        return jsonify({"error": "Session not found"}), 404

    agent = session.context.get_agent(0)
    if agent is None:
        return jsonify({"history": []})

    return jsonify({"history": agent.history})


@chat_bp.route("/chat/sessions", methods=["GET"])
def list_sessions():
    """List all active sessions."""
    sessions = current_app.config["sessions"]
    result = []
    for sid, session in sessions.items():
        agent = session.context.get_agent(0)
        msg_count = len(agent.history) if agent else 0
        result.append({
            "session_id": sid,
            "message_count": msg_count,
            "is_running": session.is_running,
        })
    return jsonify({"sessions": result})


@chat_bp.route("/chat/session/<session_id>", methods=["DELETE"])
def delete_session(session_id):
    """Delete a session."""
    sessions = current_app.config["sessions"]
    if session_id in sessions:
        del sessions[session_id]
        return jsonify({"status": "deleted"})
    return jsonify({"error": "Session not found"}), 404
