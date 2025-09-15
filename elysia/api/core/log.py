import logging
from rich.logging import RichHandler
import litellm
import dspy

# Initialize the logger as a module-level variable
_logger = None


def get_logger():
    """Get the singleton logger instance."""
    global _logger
    if _logger is None:
        # Configure basic logging if not already configured
        if not logging.root.handlers:
            FORMAT = "%(message)s"
            logging.basicConfig(
                level="DEBUG",
                format=FORMAT,
                datefmt="[%X]",
                handlers=[RichHandler(rich_tracebacks=True, markup=True)],
            )

        # Set log levels for specific loggers
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING)
        logging.getLogger("urllib3").setLevel(logging.WARNING)
        logging.getLogger("asyncio").setLevel(logging.INFO)
        logging.getLogger("fastapi").setLevel(logging.INFO)
        logging.getLogger("uvicorn").setLevel(logging.INFO)
        logging.getLogger("starlette").setLevel(logging.INFO)
        logging.getLogger("grpc").setLevel(logging.INFO)
        logging.getLogger("grpc._channel").setLevel(logging.WARNING)
        logging.getLogger("dspy").setLevel(logging.WARNING)
        logging.getLogger("dspy").propagate = False
        logging.getLogger("LiteLLM").setLevel(logging.WARNING)
        logging.getLogger("LiteLLM").propagate = False
        logging.getLogger("LiteLLM Router").setLevel(logging.WARNING)
        logging.getLogger("LiteLLM Router").propagate = False
        logging.getLogger("LiteLLM Proxy").setLevel(logging.WARNING)
        logging.getLogger("LiteLLM Proxy").propagate = False
        logging.getLogger("matplotlib").setLevel(logging.WARNING)
        logging.getLogger("matplotlib").propagate = False

        _logger = logging.getLogger("rich")
        _logger.setLevel(logging.DEBUG)

    return _logger


def set_log_level(level: str):
    """Set the level for the main application logger."""
    logger = get_logger()
    logger.setLevel(level)


# Initialize the logger
logger = get_logger()

dspy.disable_litellm_logging()
dspy.disable_logging()
litellm.suppress_debug_info = True
