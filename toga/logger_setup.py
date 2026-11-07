"""Logger factory used across toga (INFO by default; call ``logger.setLevel`` for more)."""
import logging


def setup_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    logging.getLogger("tensorflow").setLevel(logging.ERROR)
    logging.getLogger("jax").setLevel(logging.ERROR)

    handler = logging.StreamHandler()
    formatter = logging.Formatter('[%(asctime)s][%(levelname)s] %(name)s: %(message)s', datefmt='%m-%d %H:%M:%S')
    handler.setFormatter(formatter)

    # Our own handler; don't also propagate to the root logger (duplicate lines).
    if not logger.handlers:
        logger.addHandler(handler)
        logger.propagate = False

    return logger
