import importlib

_LAZY_MODULES = {"basis", "fe", "generate_mesh", "problem", "solver"}

__all__ = ["basis", "fe", "generate_mesh", "problem", "solver"]


def __getattr__(name):
    if name in _LAZY_MODULES:
        module = importlib.import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
