"""Knowledge API routes — import documents and check status."""

import asyncio
from flask import Blueprint, jsonify, current_app, request

from memory.memory_manager import MemoryManager
from memory.knowledge_import import KnowledgeImporter

knowledge_bp = Blueprint("knowledge", __name__)


@knowledge_bp.route("/knowledge/import", methods=["POST"])
def import_knowledge():
    """Trigger knowledge import from the configured knowledge directory."""
    config = current_app.config["agent_config"]
    mm = MemoryManager(config)
    importer = KnowledgeImporter(mm)
    payload = request.get_json(silent=True) or {}
    directory = payload.get("directory")
    namespace = payload.get("namespace")

    loop = asyncio.new_event_loop()
    try:
        stats = loop.run_until_complete(importer.import_directory(directory=directory, namespace=namespace))
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
    namespace = request.args.get("namespace")

    loop = asyncio.new_event_loop()
    try:
        stats = loop.run_until_complete(mm.get_stats(namespace=namespace))
        if "areas" in stats:
            knowledge_count = stats.get("areas", {}).get("knowledge", 0)
        else:
            knowledge_count = 0
            for ns_data in stats.get("namespaces", {}).values():
                knowledge_count += ns_data.get("areas", {}).get("knowledge", 0)
        return jsonify({
            "knowledge_chunks": knowledge_count,
            "total_memories": stats["total"],
            "namespace": stats.get("namespace"),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        loop.close()
