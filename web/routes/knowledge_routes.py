"""Knowledge API routes — import documents and check status."""

import asyncio
from flask import Blueprint, jsonify, current_app

from memory.memory_manager import MemoryManager
from memory.knowledge_import import KnowledgeImporter

knowledge_bp = Blueprint("knowledge", __name__)


@knowledge_bp.route("/knowledge/import", methods=["POST"])
def import_knowledge():
    """Trigger knowledge import from the configured knowledge directory."""
    config = current_app.config["agent_config"]
    mm = MemoryManager(config)
    importer = KnowledgeImporter(mm)

    loop = asyncio.new_event_loop()
    try:
        stats = loop.run_until_complete(importer.import_directory())
        return jsonify(stats)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        loop.close()


@knowledge_bp.route("/knowledge/status", methods=["GET"])
def knowledge_status():
    """Get knowledge base statistics."""
    config = current_app.config["agent_config"]
    mm = MemoryManager(config)

    loop = asyncio.new_event_loop()
    try:
        stats = loop.run_until_complete(mm.get_stats())
        knowledge_count = stats.get("areas", {}).get("knowledge", 0)
        return jsonify({
            "knowledge_chunks": knowledge_count,
            "total_memories": stats["total"],
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        loop.close()
