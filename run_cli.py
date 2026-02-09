#!/usr/bin/env python3
"""CLI entry point for Local Ollama Agents."""

import asyncio
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent.config import load_config
from cli.cli_app import CLIApp


def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.json"
    config = load_config(config_path)
    app = CLIApp(config)
    asyncio.run(app.run())


if __name__ == "__main__":
    main()
