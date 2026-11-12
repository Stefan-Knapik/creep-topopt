import jax.numpy as jnp


class TableModel:
    """Rule of mixtures over per-alloy property tables.

    Each alloy's property is piecewise linear in temperature between its tabulated rows and held
    at the end values outside them. The blend is linear in the composition fractions.
    """

    def __init__(self, material_tables, temperature_dependent=True):
        """
        material_tables: one (temperatures, values) pair per alloy, temperatures ascending
        temperature_dependent: if False, the query is the composition only and each table must
            hold a single value
        """
        self.temperature_dependent = temperature_dependent
        self.tables = [(jnp.asarray(t, dtype=float), jnp.asarray(v, dtype=float)) for t, v in material_tables]
        self.input_dim = len(self.tables) + (1 if temperature_dependent else 0)

    def __call__(self, x_query):
        """
        x_query: (M, d) or (d,), rows [w_1, ..., w_m, T] (or [w_1, ..., w_m] if temperature independent)
        returns: (M,) or scalar
        """
        x_query = jnp.asarray(x_query)
        if x_query.shape[-1] != self.input_dim:
            raise ValueError(f"x_query has last axis of length {x_query.shape[-1]}, expected {self.input_dim}.")
        n_materials = len(self.tables)
        fractions = x_query[..., :n_materials]
        if self.temperature_dependent:
            temperature = x_query[..., n_materials]
            pure = [jnp.interp(temperature, t, v) for t, v in self.tables]
        else:
            pure = [v[0] for _, v in self.tables]
        return sum(fractions[..., i] * pure[i] for i in range(n_materials))
