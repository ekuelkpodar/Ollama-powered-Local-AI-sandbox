"""Memory API routes — search and inspect memories."""

import asyncio
from flask import Blueprint, request, jsonify, current_app

from memory.memory_manager import MemoryManager

memory_bp = Blueprint("memory", __name__)


def _get_mm(config) -> MemoryManager:
    return MemoryManager(config)


@memory_bp.route("/memory/search", methods=["GET"])
def search_memories():
    """Search memories by query."""
    query = request.args.get("q", "")
    area = request.args.get("area", None)
    k = int(request.args.get("k", 10))

    if not query:
        return jsonify({"error": "No query provided"}), 400

    config = current_app.config["agent_config"]
    mm = _get_mm(config)

    loop = asyncio.new_event_loop()
    try:
        results = loop.run_until_complete(
            mm.search(query=query, area=area, k=k, threshold=0.3)
        )
        return jsonify({"results": results})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        loop.close()


@memory_bp.route("/memory/stats", methods=["GET"])
def memory_stats():
    """Get memory statistics."""
    config = current_app.config["agent_config"]
    mm = _get_mm(config)

    loop = asyncio.new_event_loop()
    try:
        stats = loop.run_until_complete(mm.get_stats())
        return jsonify(stats)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        loop.close()
