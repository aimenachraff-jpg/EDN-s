"""
EDN-S Logging Utilities
=======================
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[Path] = None,
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    Set up logging for EDN-S.
    
    Args:
        level: Logging level
        log_file: Optional file to log to
        format_string: Custom format string
        
    Returns:
        Configured logger
    """
    if format_string is None:
        format_string = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    
    handlers = [logging.StreamHandler(sys.stdout)]
    
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=level,
        format=format_string,
        handlers=handlers,
        force=True
    )
    
    logger = logging.getLogger("edns")
    logger.setLevel(level)
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the edns namespace."""
    return logging.getLogger(f"edns.{name}")