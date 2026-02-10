"""Models API route — list available Ollama models."""

import asyncio
import subprocess
import threading
import uuid
import time
from flask import Blueprint, jsonify, current_app, request

from agent.models import OllamaClient

models_bp = Blueprint("models", __name__)
_pull_jobs: dict[str, dict] = {}
_pull_lock = threading.Lock()


@models_bp.route("/models", methods=["GET"])
def list_models():
    """List all locally available Ollama models."""
    config = current_app.config["agent_config"]
    client = OllamaClient(
        config.chat_model.base_url,
        connect_timeout=config.ollama.connect_timeout,
        read_timeout=config.ollama.read_timeout,
        max_retries=config.ollama.max_retries,
    )

    loop = asyncio.new_event_loop()
    try:
        healthy = loop.run_until_complete(client.health_check())
        if not healthy:
            return jsonify({
                "error": "Cannot connect to Ollama",
                "models": [],
            }), 503

        models = loop.run_until_complete(client.list_models())
        return jsonify({
            "models": [
                {
                    "name": m.get("name", ""),
                    "size": m.get("size", 0),
                    "modified_at": m.get("modified_at", ""),
                }
                for m in models
            ]
        })
    except Exception as e:
        return jsonify({"error": str(e), "models": []}), 500
    finally:
        loop.close()


@models_bp.route("/models/pull", methods=["POST"])
def pull_model():
    """Trigger an Ollama model pull."""
    data = request.get_json() or {}
    model = (data.get("model") or "").strip()
    if not model:
        return jsonify({"error": "model is required"}), 400

    job_id = _start_pull_job(model)
    return jsonify({"job_id": job_id, "status": "started"})


@models_bp.route("/models/pull/<job_id>", methods=["GET"])
def pull_status(job_id: str):
    """Get status for a model pull job."""
    job = _pull_jobs.get(job_id)
    if not job:
        return jsonify({"error": "job not found"}), 404
    return jsonify(job)


def _start_pull_job(model: str) -> str:
    job_id = uuid.uuid4().hex[:12]
    job = {
        "job_id": job_id,
        "model": model,
        "status": "running",
        "output": [],
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "ended_at": None,
        "error": None,
    }
    _pull_jobs[job_id] = job

    def _run():
        try:
            process = subprocess.Popen(
                ["ollama", "pull", model],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            if process.stdout:
                for line in process.stdout:
                    _append_job_output(job_id, line.strip())
            process.wait()
            job["status"] = "completed" if process.returncode == 0 else "failed"
            if process.returncode != 0:
                job["error"] = f"ollama pull failed with code {process.returncode}"
        except FileNotFoundError:
            job["status"] = "failed"
            job["error"] = "ollama CLI not found"
        except Exception as e:
            job["status"] = "failed"
            job["error"] = str(e)
        finally:
            job["ended_at"] = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    return job_id


def _append_job_output(job_id: str, line: str, limit: int = 200) -> None:
    with _pull_lock:
        job = _pull_jobs.get(job_id)
        if not job:
            return
        output = job.get("output", [])
        output.append(line)
        if len(output) > limit:
            output[:] = output[-limit:]
        job["output"] = output
