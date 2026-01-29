import logging

def setup_logging(level: str = "INFO") -> logging.Logger:
    """Configure and return the application logger."""
    
    # Create logger
    logger = logging.getLogger("FINRAG")
    logger.setLevel(getattr(logging, level.upper()))
    
    # Avoid duplicate handlers if called multiple times
    if logger.handlers:
        return logger
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, level.upper()))
    
    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s"
    )
    
    # Add formatter to handler
    console_handler.setFormatter(formatter)
    
    # Add handler to logger
    logger.addHandler(console_handler)
    
    return logger


# Create a default logger instance
logger = setup_logging()
