"""TOGA: JAX finite elements and topology optimization with nonlinear creep."""
import faulthandler
import importlib
import os

from .logger_setup import setup_logger

# Reduce noisy XLA / TensorFlow backend logs unless the user explicitly opts in.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

# The WSL NVIDIA driver segfaults when XLA updates a CUDA graph (second evaluation of a jitted
# function), so turn CUDA graphs off there. XLA reads this when JAX first uses the GPU, so toga
# must be imported before any JAX array is created.
if "microsoft" in os.uname().release.lower() and "xla_gpu_enable_command_buffer" not in os.environ.get("XLA_FLAGS", ""):
    os.environ["XLA_FLAGS"] = os.environ.get("XLA_FLAGS", "") + " --xla_gpu_enable_command_buffer="

faulthandler.enable()  # Python traceback on segfaults inside XLA / sparse solvers

logger = setup_logger(__name__)

_LAZY_SUBMODULES = {"fem", "opt", "mat", "physics", "utils", "utils_for_creep", "logger_setup"}

__all__ = ["setup_logger", "logger", "fem", "opt", "mat", "physics", "utils", "utils_for_creep", "print_banner"]


def print_banner():
    """Print the TOGA banner (needs the optional ``pyfiglet``)."""
    try:
        from pyfiglet import Figlet
    except ImportError:
        logger.debug("pyfiglet not installed; skipping TOGA banner.")
        return

    figlet = Figlet(font="starwars")
    print(figlet.renderText("TOGA"))


if os.environ.get("TOGA_PRINT_BANNER") == "1":
    print_banner()


def __getattr__(name):
    if name in _LAZY_SUBMODULES:
        module = importlib.import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
