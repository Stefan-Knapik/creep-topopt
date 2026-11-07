"""Material property models (constant or RBF fits to the CSV data in material_data/)."""
from __future__ import annotations

import csv
from pathlib import Path

import jax
import jax.numpy as jnp

from ..logger_setup import setup_logger
from .constant_model import ConstantModel
from .RBF_interpolator import RBFInterpolator

DATA_DIR = Path(__file__).resolve().parent / "material_data"
logger = setup_logger(__name__)

# Properties that can be derived from a sibling CSV column when their own column
# is absent. A creep pre-exponential factor spans many orders of magnitude
# between alloys, so a material set may tabulate either 'A' or 'logA' (base-10
# log of A) and be fitted in either space -- fitting 'logA' interpolates the
# exponent linearly instead of the raw value. Each entry maps the requested
# property to (source column, transform applied to the tabulated values).
DERIVED_PROPERTIES = {
    "logA": ("A", lambda values: jnp.log10(values)),
    "A": ("logA", lambda values: 10.0 ** values),
}


def design_to_physical(theta):
    """Split a design vector [density, w_1, ..., w_m] into density and normalized composition."""
    theta = jnp.atleast_1d(theta)
    geometry = theta[0]
    composition = theta[1:]
    composition_normalized = composition / jnp.sum(composition)
    return geometry, composition_normalized

def vmap_design_to_physical(theta):
    """Batched :func:`design_to_physical` over rows of theta."""
    return jax.vmap(design_to_physical)(theta)


class MaterialProperties:
    """Named property models, exposed as attributes (e.g. ``props.E(composition)``)."""

    def __init__(self, property_models):
        self._property_models = dict(property_models)
        for property_name, model in self._property_models.items():
            setattr(self, property_name, model)

    def __repr__(self):
        names = ", ".join(sorted(self._property_models))
        return f"MaterialProperties({names})"

    def available_models(self):
        return tuple(sorted(self._property_models))

    def has_model(self, property_name):
        return property_name in self._property_models


def build_material_properties(material_names, property_configs, data_dir=None):
    """Build one model per property from ``{name: {'model': 'constant'|'rbf', ...}}`` (units in material_data/README.md)."""
    property_models = {}
    for property_name, config in property_configs.items():
        property_models[property_name] = _build_property_model(
            property_name=property_name,
            material_names=material_names,
            config=config,
            data_dir=data_dir,
        )

    return MaterialProperties(property_models)


def build_material_models(material_names, property_configs, data_dir=None):
    """Alias of :func:`build_material_properties`."""
    return build_material_properties(
        material_names=material_names,
        property_configs=property_configs,
        data_dir=data_dir,
    )


def _build_property_model(property_name, material_names, config, data_dir=None):
    if callable(config):
        return config

    if not isinstance(config, dict):
        raise TypeError(
            f"Configuration for property '{property_name}' must be a dict or callable."
        )

    model_name = config.get("model")
    if model_name == "constant":
        if "value" not in config:
            raise ValueError(
                f"Constant model for property '{property_name}' requires a 'value'."
            )
        return ConstantModel(config["value"])

    if model_name == "rbf":
        return _build_rbf_model(
            property_name=property_name,
            material_names=material_names,
            config=config,
            data_dir=data_dir,
        )

    raise ValueError(
        f"Unsupported model '{model_name}' for property '{property_name}'. "
        "Choose from 'constant' or 'rbf'."
    )


def _build_rbf_model(property_name, material_names, config, data_dir=None):
    kernel = config.get("kernel", "linear")
    epsilon = config.get("parameter", 1.0)
    reg = config.get("reg", 0.0)
    temperature_dependent = bool(config.get("temperature_dependent", True))
    temperatures = config.get("temperatures")

    x_train, y_train = _load_rbf_training_data(
        material_names=material_names,
        property_name=property_name,
        temperatures=temperatures,
        temperature_dependent=temperature_dependent,
        data_dir=data_dir,
    )

    input_scale = _resolve_input_scale(
        config.get("scale_input"), x_train
    )

    return RBFInterpolator(
        x_train=x_train,
        y_train=y_train,
        kernel=kernel,
        epsilon=epsilon,
        reg=reg,
        input_scale=input_scale,
    )


def _load_rbf_training_data(
    material_names,
    property_name,
    temperatures,
    temperature_dependent,
    data_dir=None,
):
    material_tables = _load_material_tables(material_names, property_name, data_dir)
    temperatures = _resolve_temperatures(
        material_tables, temperatures, temperature_dependent, property_name
    )

    n_materials = len(material_names)
    x_rows = []
    y_rows = []

    for material_index, material_table in enumerate(material_tables):
        composition = [0.0] * n_materials
        composition[material_index] = 1.0

        for temperature in temperatures:
            if temperature is None:
                property_value = _constant_property_value(
                    material_table, property_name, material_names[material_index]
                )
            else:
                property_value = _interpolate_property(material_table, temperature)
            if temperature_dependent:
                x_rows.append(composition + [temperature])
            else:
                x_rows.append(list(composition))
            y_rows.append(property_value)

    return jnp.asarray(x_rows, dtype=float), jnp.asarray(y_rows, dtype=float)


def _load_material_tables(material_names, property_name, data_dir=None):
    data_dir = Path(data_dir) if data_dir is not None else DATA_DIR
    return [
        _load_material_table(data_dir / f"{material_name}.csv", property_name)
        for material_name in material_names
    ]


def _resolve_temperatures(material_tables, temperatures, temperature_dependent, property_name):
    if temperatures is None:
        available_temperatures = set()
        for material_temperatures, _ in material_tables:
            available_temperatures.update(float(value) for value in material_temperatures)
        temperatures = sorted(available_temperatures)

        if not temperature_dependent:
            temperatures = [None]
    else:
        temperatures = [float(value) for value in temperatures]

    if not temperatures:
        raise ValueError(
            f"Property '{property_name}' does not have any temperatures to fit."
        )

    if (not temperature_dependent) and len(temperatures) != 1:
        raise ValueError(
            f"Property '{property_name}' is not temperature dependent, so "
            "'temperatures' must contain exactly one temperature."
        )

    return temperatures


def _load_material_table(csv_path, property_name):
    if not csv_path.exists():
        raise FileNotFoundError(f"Material data file not found: {csv_path}")

    with csv_path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file, skipinitialspace=True)
        if reader.fieldnames is None:
            raise ValueError(f"Material data file '{csv_path}' is empty.")

        if "T" not in reader.fieldnames:
            raise ValueError(f"Material data file '{csv_path}' is missing column 'T'.")

        column_name = property_name
        transform = None
        if property_name not in reader.fieldnames:
            source_column, transform = DERIVED_PROPERTIES.get(
                property_name, (None, None)
            )
            if source_column is None or source_column not in reader.fieldnames:
                raise ValueError(
                    f"Material data file '{csv_path}' is missing column '{property_name}'."
                )
            column_name = source_column
            logger.info(
                f"Material data file '{csv_path.name}' has no '{property_name}' "
                f"column; deriving it from '{source_column}'."
            )

        temperatures = []
        values = []
        for row in reader:
            temperature_text = (row.get("T") or "").strip()
            value_text = (row.get(column_name) or "").strip()
            if not temperature_text or not value_text:
                continue

            temperatures.append(float(temperature_text))
            values.append(float(value_text))

    if not temperatures:
        raise ValueError(
            f"Material data file '{csv_path}' does not contain usable '{column_name}' data."
        )

    order = sorted(range(len(temperatures)), key=temperatures.__getitem__)
    sorted_temperatures = [temperatures[index] for index in order]
    sorted_values = jnp.asarray([values[index] for index in order], dtype=float)
    if transform is not None:
        sorted_values = transform(sorted_values)
    return jnp.asarray(sorted_temperatures, dtype=float), sorted_values


def _interpolate_property(material_table, temperature):
    temperatures, values = material_table
    if temperatures.shape[0] == 1:
        return values[0]
    if temperature not in temperatures:
        logger.warning(
            f"Interpolating material property from tabulated data at T={temperature}."
        )
    return jnp.interp(temperature, temperatures, values)


def _constant_property_value(material_table, property_name, material_name):
    _, values = material_table
    unique_values = jnp.unique(values)
    if unique_values.shape[0] != 1:
        raise ValueError(
            f"Property '{property_name}' is not temperature dependent, but "
            f"material '{material_name}' has multiple distinct property values in "
            "its CSV data. Specify 'temperatures' explicitly."
        )
    return unique_values[0]


def _resolve_input_scale(scale_input, x_train):
    input_dim = x_train.shape[1]
    if scale_input in (None, False):
        return None

    input_scale = jnp.asarray(scale_input, dtype=float)
    if input_scale.shape != (input_dim,):
        raise ValueError(
            f"scale_input must have shape ({input_dim},), got {input_scale.shape}."
        )
    return input_scale
