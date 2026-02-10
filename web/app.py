"""Flask application factory for Local Ollama Agents web UI."""

import asyncio
import json
import queue
import threading
from flask import Flask, send_from_directory
from flask_cors import CORS

from agent.config import AgentConfig
from agent.agent_context import AgentContext
from memory.memory_manager import MemoryManager
from extensions.extension_manager import ExtensionManager
from web.auth import init_auth


def create_app(config: AgentConfig) -> Flask:
    """Create and configure the Flask application."""
    static_dir = __file__.replace("app.py", "static")
    app = Flask(__name__, static_folder=static_dir, static_url_path="/static")
    CORS(app)

    # Shared state
    app.config["agent_config"] = config
    app.config["sessions"] = {}  # session_id -> SessionState
    app.config["config_path"] = "config.json"

    init_auth(app, config.auth)

    # Register blueprints
    from web.routes.chat import chat_bp
    from web.routes.settings import settings_bp
    from web.routes.models import models_bp
    from web.routes.memory_routes import memory_bp
    from web.routes.knowledge_routes import knowledge_bp
    from web.routes.metrics import metrics_bp

    app.register_blueprint(chat_bp, url_prefix="/api")
    app.register_blueprint(settings_bp, url_prefix="/api")
    app.register_blueprint(models_bp, url_prefix="/api")
    app.register_blueprint(memory_bp, url_prefix="/api")
    app.register_blueprint(knowledge_bp, url_prefix="/api")
    app.register_blueprint(metrics_bp, url_prefix="/api")

    @app.route("/")
    def index():
        return send_from_directory(static_dir, "index.html")

    return app


class SessionState:
    """Holds the state for one chat session."""

    def __init__(self, config: AgentConfig, session_id: str | None = None):
        self.config = config
        self.context = AgentContext(config, session_id=session_id)
        self.stream_queue: queue.Queue = queue.Queue()
        self.is_running = False
        self.final_response: str | None = None

        # Initialize memory manager
        mm = MemoryManager(config, telemetry=self.context.telemetry)
        self.context.data["memory_manager"] = mm

        # Initialize extensions
        ext_mgr = ExtensionManager(config)
        ext_mgr.discover_extensions()
        self.context.extension_manager = ext_mgr

        # Set up streaming callback
        self.context.on_stream = self._stream_callback

        # Create agent 0
        self.context.create_agent(agent_id=0)

    def _stream_callback(self, agent, chunk: str):
        """Push stream chunks to the queue for SSE consumption."""
        self.stream_queue.put({
            "type": "chunk",
            "agent_id": agent.agent_id,
            "content": chunk,
        })

    def send_message(self, message: str):
        """Start the agent monologue in a background thread."""
        self.is_running = True
        self.final_response = None

        def run_in_thread():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                agent = self.context.get_agent(0)
                result = loop.run_until_complete(agent.monologue(message))
                self.final_response = result
                self.stream_queue.put({
                    "type": "done",
                    "content": result,
                })
            except Exception as e:
                self.stream_queue.put({
                    "type": "error",
                    "content": str(e),
                })
            finally:
                self.is_running = False
                loop.close()

        thread = threading.Thread(target=run_in_thread, daemon=True)
        thread.start()
