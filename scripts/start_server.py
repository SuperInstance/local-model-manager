#!/usr/bin/env python3
"""
Script to start the Local Model Manager API server
"""

import asyncio
import logging
import signal
import sys
from pathlib import Path

# Add parent to path for local development
sys.path.insert(0, str(Path(__file__).parent.parent))

from local_model_manager.api.server import run_server

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def signal_handler(signum, frame):
    """Handle shutdown signals"""
    logger.info("Received shutdown signal, stopping server...")
    sys.exit(0)

def main():
    """Start the server"""
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Configuration
    host = "0.0.0.0"
    port = 8000

    logger.info(f"Starting Local Model Manager API server")
    logger.info(f"Server will be available at http://{host}:{port}")
    logger.info(f"API documentation available at http://{host}:{port}/docs")

    try:
        run_server(host=host, port=port)
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()