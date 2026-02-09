#!/usr/bin/env python3
"""Web UI entry point for Local Ollama Agents."""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent.config import load_config
from web.app import create_app


def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.json"
    config = load_config(config_path)

    print(f"\n  Local Ollama Agents — Web UI")
    print(f"  Chat model: {config.chat_model.model_name}")
    print(f"  Ollama: {config.chat_model.base_url}")
    print(f"  Open http://localhost:5000 in your browser\n")

    app = create_app(config)
    app.run(host="0.0.0.0", port=5000, debug=True, threaded=True)


if __name__ == "__main__":
    main()
