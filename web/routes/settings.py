"""Settings API routes — get and update configuration."""

import json
import os
import importlib
import inspect
from flask import Blueprint, request, jsonify, current_app

from extensions.base_extension import Extension

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
    config_path = _config_path()
    try:
        existing = {}
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                existing = json.load(f)
        if isinstance(existing, dict) and isinstance(data, dict):
            existing.update(data)
            payload = existing
        else:
            payload = data
        with open(config_path, "w") as f:
            json.dump(payload, f, indent=2)
        return jsonify({"status": "saved", "message": "Restart to apply changes"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@settings_bp.route("/settings/raw", methods=["GET"])
def get_raw_settings():
    """Return the raw config.json."""
    config_path = _config_path()
    try:
        with open(config_path, "r") as f:
            data = json.load(f)
        return jsonify({"config": data})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@settings_bp.route("/settings/raw", methods=["POST"])
def save_raw_settings():
    """Save raw config.json content."""
    data = request.get_json()
    if data is None:
        return jsonify({"error": "Invalid JSON payload"}), 400

    config_path = _config_path()
    try:
        with open(config_path, "w") as f:
            json.dump(data, f, indent=2)
        return jsonify({"status": "saved", "message": "Restart to apply changes"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@settings_bp.route("/extensions", methods=["GET"])
def list_extensions():
    """List available extensions and their enabled state."""
    config = current_app.config["agent_config"]
    enabled_map = getattr(config.extensions, "enabled_map", {})
    extensions_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "extensions",
        "builtin",
    )

    items = []
    if os.path.isdir(extensions_dir):
        for filename in sorted(os.listdir(extensions_dir)):
            if not filename.endswith(".py") or filename.startswith("_"):
                continue
            if filename == "__init__.py":
                continue
            module_name = f"extensions.builtin.{filename[:-3]}"
            try:
                module = importlib.import_module(module_name)
                for _, obj in inspect.getmembers(module, inspect.isclass):
                    if issubclass(obj, Extension) and obj is not Extension and obj.name:
                        override = enabled_map.get(obj.name)
                        effective = obj.enabled if override is None else override
                        items.append({
                            "name": obj.name,
                            "enabled": effective,
                        })
            except Exception:
                continue

    return jsonify({"extensions": items})


@settings_bp.route("/extensions", methods=["POST"])
def save_extensions():
    """Update extension enablement in config.json."""
    data = request.get_json() or {}
    updates = data.get("extensions", {})
    if not isinstance(updates, dict):
        return jsonify({"error": "extensions must be an object"}), 400

    config_path = _config_path()
    try:
        with open(config_path, "r") as f:
            config_data = json.load(f)
    except Exception:
        config_data = {}

    current = config_data.get("extensions", {})
    if not isinstance(current, dict):
        current = {}

    for key, value in updates.items():
        if isinstance(value, bool):
            current[key] = value

    config_data["extensions"] = current
    try:
        with open(config_path, "w") as f:
            json.dump(config_data, f, indent=2)
        return jsonify({"status": "saved", "message": "Restart to apply changes"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def _config_path() -> str:
    return current_app.config.get("config_path", "config.json")
