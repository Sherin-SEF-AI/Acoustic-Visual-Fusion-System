#!/usr/bin/env python3
"""
Acoustic-Visual Fusion System - Main Entry Point

Real-time 3D sound source localization with visual tracking and correlation.
"""

import argparse
import signal
import sys
import time
from pathlib import Path
from loguru import logger


def setup_logging(log_level: str = "INFO"):
    """Configure logging."""
    logger.remove()
    logger.add(
        sys.stderr,
        level=log_level,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>"
    )
    
    # Create logs directory
    Path("logs").mkdir(exist_ok=True)
    logger.add(
        "logs/avfusion_{time}.log",
        rotation="1 day",
        retention="7 days",
        level="DEBUG"
    )


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Acoustic-Visual Fusion System for Source Localization"
    )
    parser.add_argument(
        "--config", "-c",
        default="config/settings.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log level"
    )
    
    args = parser.parse_args()
    setup_logging(args.log_level)
    
    logger.info("=" * 60)
    logger.info("Acoustic-Visual Fusion System")
    logger.info("=" * 60)
    
    # Load configuration
    from src.core.config import get_config
    config = get_config(args.config)
    logger.info(f"Loaded configuration from {args.config}")
    
    # Run PyQt6 GUI
    logger.info("Starting PyQt6 GUI...")
    from src.gui.main_window import run_gui
    run_gui(config)


if __name__ == "__main__":
    main()
