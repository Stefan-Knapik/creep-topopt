import jax.numpy as np

from toga.fem.problem import Problem
from toga.mat.material_properties import design_to_physical
from toga.opt import penalize


class Thermal(Problem):
    """Steady heat conduction with RAMP-penalized conductivity k [W/(mm K)], T in °C."""

    def custom_init(self, material_models):
        # Override base class method.
        # Set up 'self.fe.flex_inds' so that location-specific TO can be realized.
        self.fe = self.fes[0]
        self.fe.flex_inds = np.arange(len(self.fe.cells))
        self.model_k = material_models.k
        self.penalize = lambda x: penalize(x, {'RAMP':8.0})
        
    def get_element_properties(self, mats):
        k = self.model_k(mats)
        return k
        
    def get_tensor_map(self):
        def fn(u_grad, theta):
            geom, mats = design_to_physical(theta)
            k = self.get_element_properties(mats) 
            penal = self.penalize(geom)
            k_penal = k * penal
            return k_penal * u_grad
        return fn
                
    def set_params(self, params):
        full_params = np.ones((self.fe.num_cells, params.shape[1]))
        full_params = full_params.at[self.fe.flex_inds].set(params)
        thetas = np.repeat(full_params[:, None, :], self.fe.num_quads, axis=1)
        self.full_params = full_params
        self.internal_vars = [thetas]


class NonlinearThermal(Thermal):
    """Steady heat conduction with temperature-dependent conductivity k(composition, T)."""

    def custom_init(self, material_models):
        super().custom_init(material_models)
        self.temperature_dependent_laplace = True
        # If set (e.g. 600.), k is evaluated at this fixed temperature instead
        # of the local solution temperature. k then no longer depends on the
        # unknown T, so the thermal problem becomes linear in T.
        self.k_temperature = None

    def get_element_properties(self, mats, temperature):
        if self.k_temperature is not None:
            temperature = self.k_temperature
        material_state = np.concatenate((mats, np.atleast_1d(temperature)))
        k = self.model_k(material_state)
        return np.squeeze(k)

    def get_tensor_map(self):
        def fn(u_grad, temperature, theta):
            geom, mats = design_to_physical(theta)
            k = self.get_element_properties(mats, temperature) * self.penalize(geom)
            return k * u_grad
        return fn
