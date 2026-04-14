# core/logger.py

import logging
import sys
from typing import Optional

try:
    from core.config import settings
    DEFAULT_LOG_LEVEL = settings.LOG_LEVEL
except ImportError:
    DEFAULT_LOG_LEVEL = "INFO"

# logging format: timestamp | log level | logger name | message
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

def setup_logger(name: str, level: Optional[str] = None) -> logging.Logger:
    """
    Returns a standardized logger instance for use in each module.  
    Usage: logger = setup_logger(__name__)
    """
    logger = logging.getLogger(name)
    
    # If the logger already has a handler set up (to prevent duplicate calls), return as is.
    if logger.handlers:
        return logger

    # Set the log level
    log_level = level or DEFAULT_LOG_LEVEL
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    
    # Prevent duplicate logging (do not propagate to parent loggers)
    logger.propagate = False

    # Create console handler and set level
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logger.level)
    
    # Apply formatter
    formatter = logging.Formatter(fmt=LOG_FORMAT, datefmt=DATE_FORMAT)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(console_handler)
    
    return logger