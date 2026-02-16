#!/usr/bin/env python3
"""
Frigate Event Buffer - Entry Point

Listens to Frigate MQTT topics, tracks events through their lifecycle,
sends Ring-style notifications to Home Assistant, and manages rolling retention.

Run with: python -m frigate_buffer.main
"""

import logging
import signal
import sys

from frigate_buffer.config import load_config
from frigate_buffer.logging_utils import setup_logging
from frigate_buffer.orchestrator import StateAwareOrchestrator

# Early logging for config loading (reconfigured after config is loaded)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger('frigate-buffer')

orchestrator = None


def signal_handler(sig, frame):
    """Handle shutdown signals."""
    global orchestrator
    logger.info(f"Received signal {sig}, shutting down...")
    if orchestrator:
        orchestrator.stop()
    sys.exit(0)


def main():
    """Main entry point."""
    # Load configuration
    config = load_config()

    # Setup logging with configured level
    setup_logging(config.get('LOG_LEVEL', 'INFO'))

    # Setup signal handlers
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    # Create and start orchestrator
    global orchestrator
    orchestrator = StateAwareOrchestrator(config)
    orchestrator.start()


if __name__ == '__main__':
    main()
