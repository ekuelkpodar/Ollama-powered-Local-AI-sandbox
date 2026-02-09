"""Settings API routes — get and update configuration."""

import json
import os
from flask import Blueprint, request, jsonify, current_app

settings_bp = Blueprint("settings", __name__)


@settings_bp.route("/settings", methods=["GET"])
def get_settings():
    """Return current configuration."""
    config = current_app.config["agent_config"]
    return jsonify({
        "chat_model": {
            "model_name": config.chat_model.model_name,
            "base_url": config.chat_model.base_url,
            "temperature": config.chat_model.temperature,
            "ctx_length": config.chat_model.ctx_length,
        },
        "utility_model": {
            "model_name": config.utility_model.model_name,
            "base_url": config.utility_model.base_url,
            "temperature": config.utility_model.temperature,
            "ctx_length": config.utility_model.ctx_length,
        },
        "embedding_model": config.embedding_model,
        "prompt_profile": config.prompt_profile,
        "max_monologue_iterations": config.max_monologue_iterations,
        "memory_recall_enabled": config.memory_recall_enabled,
        "memory_recall_threshold": config.memory_recall_threshold,
        "memory_recall_count": config.memory_recall_count,
    })


@settings_bp.route("/settings", methods=["POST"])
def save_settings():
    """Save configuration to config.json."""
    data = request.get_json()

    config_path = "config.json"
    try:
        with open(config_path, "w") as f:
            json.dump(data, f, indent=2)
        return jsonify({"status": "saved", "message": "Restart to apply changes"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
