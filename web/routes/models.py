"""Models API route — list available Ollama models."""

import asyncio
from flask import Blueprint, jsonify, current_app

from agent.models import OllamaClient

models_bp = Blueprint("models", __name__)


@models_bp.route("/models", methods=["GET"])
def list_models():
    """List all locally available Ollama models."""
    config = current_app.config["agent_config"]
    client = OllamaClient(config.chat_model.base_url)

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
