import logging
import sys

NNCF_LOGGER_NAME = "nncs"

logger = logging.getLogger(NNCF_LOGGER_NAME)
logger.propagate = False

stdout_handler = logging.StreamHandler(sys.stdout)
fmt = logging.Formatter("%(levelname)s:%(name)s:%(message)s")
stdout_handler.setFormatter(fmt)
stdout_handler.setLevel(logging.INFO)
logger.addHandler(stdout_handler)
logger.setLevel(logging.INFO)


def set_log_level(level):
    logger.setLevel(level)
    for handler in logger.handlers:
        handler.setLevel(level)


def disable_logging():
    logger.handlers = []
