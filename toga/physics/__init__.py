"""Physics problems built on toga.fem.Problem; classes are imported on first access."""
import importlib

_LAZY_CLASSES = {
    "Elasticity": "linear_elasticity",
    "Thermal": "thermal",
    "NonlinearThermal": "thermal",
    "CreepExplicit": "creep_explicit",
    "CreepImplicit": "creep_implicit",
    "ThermalCreepExplicit": "thermal_creep_explicit",
    "ThermalCreepImplicit": "thermal_creep_implicit",
}

__all__ = list(_LAZY_CLASSES)


def __getattr__(name):
    if name in _LAZY_CLASSES:
        module = importlib.import_module(f"{__name__}.{_LAZY_CLASSES[name]}")
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
