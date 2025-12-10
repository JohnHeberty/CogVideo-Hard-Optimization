"""
Structured logging configuration for CogVideoX.

This module provides centralized logging setup with consistent formatting,
log levels, and optional file output for debugging.
"""

import logging
import sys
from pathlib import Path
from typing import Optional
import os
from datetime import datetime


# ANSI color codes for terminal output
class LogColors:
    RESET = "\033[0m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"
    BOLD = "\033[1m"


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for different log levels."""
    
    COLORS = {
        'DEBUG': LogColors.CYAN,
        'INFO': LogColors.GREEN,
        'WARNING': LogColors.YELLOW,
        'ERROR': LogColors.RED,
        'CRITICAL': LogColors.RED + LogColors.BOLD,
    }
    
    def format(self, record):
        if sys.stderr.isatty():  # Only color if terminal supports it
            color = self.COLORS.get(record.levelname, LogColors.RESET)
            record.levelname = f"{color}{record.levelname}{LogColors.RESET}"
            record.name = f"{LogColors.BLUE}{record.name}{LogColors.RESET}"
        return super().format(record)


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    format_string: Optional[str] = None,
    include_timestamp: bool = True,
) -> logging.Logger:
    """
    Setup logging configuration for CogVideoX.
    
    Args:
        log_level: Logging level ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
        log_file: Optional path to log file. If None, logs only to console.
        format_string: Custom format string. If None, uses default.
        include_timestamp: Whether to include timestamp in logs
        
    Returns:
        Configured root logger
        
    Example:
        >>> from logging_config import setup_logging
        >>> logger = setup_logging(log_level="DEBUG")
        >>> logger.info("Pipeline loaded successfully")
        
    Environment Variables:
        LOG_LEVEL: Override log level (DEBUG, INFO, WARNING, ERROR)
        LOG_FILE: Path to log file
    """
    # Get log level from environment or parameter
    log_level = os.getenv("LOG_LEVEL", log_level).upper()
    log_file = os.getenv("LOG_FILE", log_file)
    
    # Validate log level
    numeric_level = getattr(logging, log_level, None)
    if not isinstance(numeric_level, int):
        print(f"Invalid log level: {log_level}, using INFO", file=sys.stderr)
        numeric_level = logging.INFO
    
    # Default format string
    if format_string is None:
        if include_timestamp:
            format_string = "[%(asctime)s] %(levelname)-8s [%(name)s] %(message)s"
        else:
            format_string = "%(levelname)-8s [%(name)s] %(message)s"
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    
    # Remove existing handlers
    root_logger.handlers.clear()
    
    # Console handler with colors
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_formatter = ColoredFormatter(
        format_string,
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, mode='a')
        file_handler.setLevel(numeric_level)
        file_formatter = logging.Formatter(
            format_string,
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
        
        root_logger.info(f"Logging to file: {log_file}")
    
    return root_logger


def get_logger(name: str) -> logging.Logger:
    """
    Get logger instance for a module.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Logger instance
        
    Example:
        >>> from logging_config import get_logger
        >>> logger = get_logger(__name__)
        >>> logger.info("Starting generation")
    """
    return logging.getLogger(name)


class LogContext:
    """Context manager for temporary log level changes."""
    
    def __init__(self, logger: logging.Logger, level: str):
        self.logger = logger
        self.new_level = getattr(logging, level.upper())
        self.old_level = logger.level
    
    def __enter__(self):
        self.logger.setLevel(self.new_level)
        return self.logger
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.logger.setLevel(self.old_level)


def log_vram_status_structured(logger: logging.Logger) -> None:
    """
    Log VRAM status in structured format.
    
    Args:
        logger: Logger instance
    """
    try:
        from vram_utils import get_gpu_memory_info
        
        total_gb, used_gb, available_gb = get_gpu_memory_info()
        
        if total_gb > 0:
            usage_percent = (used_gb / total_gb) * 100
            logger.info(
                f"VRAM Status: {used_gb:.1f}GB / {total_gb:.1f}GB used "
                f"({usage_percent:.1f}%), {available_gb:.1f}GB available"
            )
        else:
            logger.warning("CUDA not available - running in CPU mode")
    except ImportError:
        logger.debug("vram_utils not available, skipping VRAM logging")


def log_generation_params(logger: logging.Logger, **kwargs) -> None:
    """
    Log generation parameters in structured format.
    
    Args:
        logger: Logger instance
        **kwargs: Generation parameters (prompt, model_path, num_frames, etc.)
        
    Example:
        >>> log_generation_params(
        ...     logger,
        ...     prompt="A golden retriever running",
        ...     model_path="THUDM/CogVideoX-5b",
        ...     num_frames=49,
        ...     fps=8
        ... )
    """
    logger.info("=" * 60)
    logger.info("Generation Parameters:")
    for key, value in kwargs.items():
        # Truncate long prompts
        if key in ["prompt", "negative_prompt"] and isinstance(value, str) and len(value) > 100:
            value = value[:100] + "..."
        logger.info(f"  {key:20s}: {value}")
    logger.info("=" * 60)


def log_timing(logger: logging.Logger, operation: str, duration_seconds: float) -> None:
    """
    Log operation timing information.
    
    Args:
        logger: Logger instance
        operation: Operation name (e.g., "Model Load", "Generation", "VAE Decode")
        duration_seconds: Duration in seconds
        
    Example:
        >>> import time
        >>> start = time.time()
        >>> # ... do something ...
        >>> log_timing(logger, "Model Load", time.time() - start)
    """
    minutes = int(duration_seconds // 60)
    seconds = duration_seconds % 60
    
    if minutes > 0:
        time_str = f"{minutes}m {seconds:.1f}s"
    else:
        time_str = f"{seconds:.1f}s"
    
    logger.info(f"⏱️  {operation}: {time_str}")


def create_session_log_file(base_dir: str = "./logs") -> str:
    """
    Create a unique log file for this session.
    
    Args:
        base_dir: Base directory for logs
        
    Returns:
        Path to log file
        
    Example:
        >>> log_file = create_session_log_file()
        >>> logger = setup_logging(log_file=log_file)
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path(base_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    return str(log_dir / f"cogvideo_{timestamp}.log")


if __name__ == "__main__":
    # Test logging configuration
    print("Testing logging configuration...")
    
    # Setup with default settings
    logger = setup_logging(log_level="DEBUG")
    
    print("\n=== Testing log levels ===")
    logger.debug("This is a DEBUG message")
    logger.info("This is an INFO message")
    logger.warning("This is a WARNING message")
    logger.error("This is an ERROR message")
    logger.critical("This is a CRITICAL message")
    
    print("\n=== Testing structured logging ===")
    log_generation_params(
        logger,
        prompt="A golden retriever sprinting",
        model_path="THUDM/CogVideoX-5b",
        num_frames=49,
        fps=8,
        seed=42,
        guidance_scale=6.5
    )
    
    log_timing(logger, "Model Load", 15.3)
    log_timing(logger, "Generation", 182.7)
    
    print("\n=== Testing VRAM logging ===")
    log_vram_status_structured(logger)
    
    print("\n=== Testing context manager ===")
    logger.info("Current level: INFO")
    with LogContext(logger, "DEBUG"):
        logger.debug("Temporarily in DEBUG mode")
    logger.debug("Back to INFO (this won't show)")
    
    print("\n=== Testing file logging ===")
    log_file = create_session_log_file("./test_logs")
    print(f"Log file created: {log_file}")
    
    logger_with_file = setup_logging(log_level="INFO", log_file=log_file)
    logger_with_file.info("This message goes to both console and file")
    
    print(f"\nCheck log file at: {log_file}")
