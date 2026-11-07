from typing import Dict

import jax
import jax.numpy as np

from toga.fem.problem import Problem
from toga.fem.solver import ad_wrapper
from toga.mat.material_properties import design_to_physical, vmap_design_to_physical
from toga.opt import penalize
from toga.utils_for_creep import pad_to_3x3, safe_divide, safe_sqrt, trim_to_dim


class ThermalCreepExplicit(Problem):
    """Thermo-mechanical Norton creep with Arrhenius factor, forward-Euler in time (MPa, s, °C)."""

    POSSIBLE_MATERIAL_MODELS = ("E", "alpha", "YS", "rho", "cost", "A", "logA", "n", "Q", "criticality")
    GAS_CONSTANT = 8.31446261815324  # J/(mol K)
    CELSIUS_TO_KELVIN = 273.15

    def available_material_models(self, material_models):
        return tuple(
            name for name in self.POSSIBLE_MATERIAL_MODELS
            if material_models.has_model(name)
        )

    def bind_material_models(self, material_models):
        for model_name in self.available_material_models(material_models):
            setattr(self, f"model_{model_name}", getattr(material_models, model_name))

    def custom_init(self, material_models):
        # Override base class method.
        # Set up 'self.fe.flex_inds' so that location-specific TO can be realized.
        self.fe = self.fes[0]
        self.fe.flex_inds = np.arange(len(self.fe.cells))

        self.bind_material_models(material_models)
        if hasattr(self, "model_A") == hasattr(self, "model_logA"):
            raise ValueError(
                "Creep needs exactly one model for the pre-exponential factor: "
                "supply 'A' or 'logA' (log10 of A) in the property configs, not both."
            )

        self.penalize_E = lambda x: penalize(x, {'RAMP':8.0})

        # Make low-density regions creep faster
        self.penalize_A = lambda x: 1 / penalize(x, {'RAMP':8.0})**2.0

        # Recompute the per-element property/stress map in the backward pass instead of
        # storing intermediates at every quadrature point; bounds reverse-mode memory.
        self.remat_element = True
        # Cells per lax.map chunk in the creep update (None: one vmap over all cells).
        self.element_chunk_size = None

    def get_tensor_map(self):
        tensor_map = lambda *x: self.creep_stress_strain_map(*x)[0]
        if getattr(self, "remat_element", False):
            tensor_map = jax.checkpoint(tensor_map)
        return tensor_map

    def strain(self, u_grad):
        return 0.5 * (u_grad + u_grad.T)

    def _call_model(self, model, mats, T):
        # Models with input_dim == len(mats) + 1 take [mats..., T]; others take mats only.
        if getattr(model, 'input_dim', None) == len(mats) + 1:
            return model(np.concatenate((mats, np.atleast_1d(T))))
        return model(mats)

    def creep_prefactor(self, mats, T):
        """Creep pre-exponential factor A.

        A material set may supply 'A' directly, or 'logA' = log10(A). The latter
        lets the property model interpolate the exponent linearly, which keeps
        the fit meaningful when A spans orders of magnitude between materials.
        """
        if hasattr(self, "model_logA"):
            return 10.0 ** self._call_model(self.model_logA, mats, T)
        return self._call_model(self.model_A, mats, T)

    def get_element_properties(self, mats, T):
        E = self._call_model(self.model_E, mats, T)
        A = self.creep_prefactor(mats, T)
        n = self._call_model(self.model_n, mats, T)
        alpha = self._call_model(self.model_alpha, mats, T) if hasattr(self, "model_alpha") else 0.0
        Q = self._call_model(self.model_Q, mats, T) if hasattr(self, "model_Q") else 0.0
        nu = 0.3
        return E, nu, A, n, alpha, Q

    def stress(self, epsilon, theta, T, reference_T):
        geom, mats = design_to_physical(theta)
        E, nu, _, _, alpha, _ = self.get_element_properties(mats, T)
        penal = self.penalize_E(geom)
        E = E * penal

        delta_temperature = T - reference_T
        epsilon_thermal = alpha * delta_temperature * np.eye(self.dim)
        epsilon_mechanical = epsilon - epsilon_thermal

        if self.dim == 2:
            # Plane stress assumption for 2D elasticity
            eps11 = epsilon_mechanical[0, 0]
            eps22 = epsilon_mechanical[1, 1]
            eps12 = epsilon_mechanical[0, 1]
            sig11 = E / (1 + nu) / (1 - nu) * (eps11 + nu * eps22)
            sig22 = E / (1 + nu) / (1 - nu) * (nu * eps11 + eps22)
            sig12 = E / (1 + nu) * eps12
            sigma = np.array([[sig11, sig12], [sig12, sig22]])
        elif self.dim == 3:
            lmbda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
            mu = E / (2.0 * (1.0 + nu))
            sigma = (lmbda * np.trace(epsilon_mechanical) * np.eye(3)
                    + 2.0 * mu * epsilon_mechanical)
        else:
            raise ValueError(f"Unsupported dimension {self.dim}")
        return sigma

    def creep_stress_strain_map(
        self,
        u_grad,
        sigma_old,
        epsilon_creep_old,
        theta,
        T,
        reference_T,
    ):
        geom, mats = design_to_physical(theta)
        _, _, A, n, _, Q = self.get_element_properties(mats, T)
        
        # Make low-density regions creep faster so gray regions are less attractive
        # during topology optimization under sustained loading.
        penal = self.penalize_A(geom)
        A = A * penal
        # Temperatures are carried in Celsius throughout (BCs, thermal solve,
        # thermal expansion), but the Arrhenius term needs absolute temperature.
        T_kelvin = np.maximum(T + self.CELSIUS_TO_KELVIN, 1.0)
        arrhenius = np.exp(-Q / (self.GAS_CONSTANT * T_kelvin))

        epsilon_new = self.strain(u_grad)

        sigma_old = pad_to_3x3(sigma_old)
        s_dev = sigma_old - 1.0 / 3.0 * np.trace(sigma_old) * np.eye(3)
        s_norm = safe_sqrt(3.0 / 2.0 * np.sum(s_dev * s_dev))
        n_d = 3.0 / 2.0 * safe_divide(s_dev, s_norm)
        n_d = trim_to_dim(n_d, self.dim)
        creep_strain_rate = A * arrhenius * s_norm**n * n_d

        epsilon_creep_inc = creep_strain_rate * self.dt

        epsilon_elastic = epsilon_new - epsilon_creep_old - epsilon_creep_inc
        sigma_new = self.stress(
            epsilon_elastic,
            theta,
            T,
            reference_T,
        )

        return sigma_new, epsilon_creep_inc

    def stress_strain_fns(self):
        # Per-element map, vmapped over quadrature points; the caller maps over cells.
        cssm = self.creep_stress_strain_map
        if getattr(self, "remat_element", False):
            cssm = jax.checkpoint(cssm)
        return jax.vmap(cssm)

    def update_stress_strain(self, sol, params):
        u_grads = self.fe.sol_to_grad(sol)

        sigmas_old, epsilons_creep_old, thetas, T_quad, ref_T_quad = params

        per_cell = self.stress_strain_fns()
        cell_inputs = (u_grads, sigmas_old, epsilons_creep_old,
                       thetas, T_quad, ref_T_quad)

        chunk_size = getattr(self, "element_chunk_size", None)
        if chunk_size:
            # Evaluate cells in chunks to bound the forward peak memory too.
            stress, creep_strain_inc = jax.lax.map(
                lambda cell: per_cell(*cell), cell_inputs, batch_size=chunk_size
            )
        else:
            stress, creep_strain_inc = jax.vmap(
                lambda cell: per_cell(*cell)
            )(cell_inputs)

        stress_update = np.squeeze(stress)
        creep_strain_update = epsilons_creep_old + creep_strain_inc
        return stress_update, creep_strain_update, thetas, T_quad, ref_T_quad

    def init_params(self, theta, T, reference_T):
        sigmas_old = np.zeros((len(self.fe.cells), self.fe.num_quads, self.fe.vec, self.dim))
        epsilons_creep_old = np.zeros_like(sigmas_old)

        full_params = np.ones((self.num_cells, theta.shape[1]))
        full_params = full_params.at[self.fe.flex_inds].set(theta)
        thetas = np.repeat(full_params[:, None, :], self.fe.num_quads, axis=1)
        self.full_params = full_params

        T_quad = self.fe.convert_from_dof_to_quad(T)

        if np.ndim(reference_T) == 0:
            ref_T_quad = np.full((self.fe.num_cells, self.fe.num_quads), reference_T)
        else:
            ref_T_quad = self.fe.convert_from_dof_to_quad(reference_T)

        return sigmas_old, epsilons_creep_old, thetas, T_quad, ref_T_quad

    def set_params(self, params):
        body_params, surface_params = params
        self.internal_vars = body_params
        self.internal_vars_surfaces = [[surface_params]]

    def compute_vol_avg_stress(self, body_params):
        sigmas, _, _, _, _ = body_params

        sigma = np.sum(
            sigmas.reshape(-1, self.fe.vec, self.dim)
            * self.fe.JxW.reshape(-1)[:, None, None],
            axis=0,
        )
        vol = np.sum(self.fe.JxW)
        return sigma / vol

    def compute_compliance(self, sol):
        """Compliance: work of the surface tractions on displacement ``sol``."""
        total = 0.0
        for i in range(len(self.boundary_inds_list)):
            boundary_inds = self.boundary_inds_list[i]
            _, nanson_scale = self.fe.get_face_shape_grads(boundary_inds)
            u_face = (
                sol[self.fe.cells][boundary_inds[:, 0]][:, None, :, :]
                * self.fe.face_shape_vals[boundary_inds[:, 1]][:, :, :, None]
            )
            u_face = np.sum(u_face, axis=2)
            subset_quad_points = self.physical_surface_quad_points[i]
            neumann_fn = self.get_surface_maps()[i]
            traction = -jax.vmap(jax.vmap(neumann_fn))(u_face, subset_quad_points)
            total += np.sum(traction * u_face * nanson_scale[:, :, None])
        return total

    def forward_solve(
        self,
        theta: np.ndarray,
        T: np.ndarray,
        reference_T,
        end_time: float,
        num_time_steps: int,
        fwd_solver_options: Dict,
        vjp_solver_options: Dict,
        save_history: bool = True,
    ):
        """Explicit creep time stepping.

        Returns the displacement at every step and the body params (stress, creep strain, ...)
        at every step, or with ``save_history=False`` only at the first and last step. The
        stress history is the part of GPU memory that grows with ``num_time_steps``.
        """
        fwd_pred = ad_wrapper(
            self,
            solver_options=fwd_solver_options,
            adjoint_solver_options=vjp_solver_options,
        )

        self.end_time = end_time
        self.dt = end_time / num_time_steps

        body_params = self.init_params(theta, T, reference_T)

        # Constant across all time steps.
        surface_params = [
            np.ones((surf_quads.shape[0], self.fe.num_face_quads))
            for surf_quads in self.physical_surface_quad_points
        ]

        def step(bp):
            sol_list = fwd_pred([bp, surface_params])
            return sol_list, self.update_stress_strain(sol_list[0], bp)

        if save_history:
            def scan_fn(carry, _):
                sol_list, bp = step(carry)
                return bp, (sol_list, bp)

            _, (full_sol_list, body_params_list) = jax.lax.scan(
                scan_fn, body_params, xs=None, length=num_time_steps + 1
            )
            return full_sol_list, body_params_list

        def scan_fn(carry, _):
            sol_list, bp = step(carry)
            return bp, sol_list

        sol_first, bp_first = step(body_params)
        bp_last, sols = jax.lax.scan(scan_fn, bp_first, xs=None, length=num_time_steps)
        full_sol_list = jax.tree_util.tree_map(lambda a, b: np.concatenate((a[None], b)), sol_first, sols)
        body_params_list = jax.tree_util.tree_map(lambda a, b: np.stack((a, b)), bp_first, bp_last)
        return full_sol_list, body_params_list

    def compute_volume_fraction(self, params):
        geom, _ = vmap_design_to_physical(params)
        return np.sum(geom * np.sum(self.JxW, axis=[1, 2])) / np.sum(self.JxW)
    
    def compute_volume_fraction_of_material(self, params, mat_idx):
        _, mat = vmap_design_to_physical(params)
        return np.sum(mat[:,mat_idx] * np.sum(self.JxW, axis=[1, 2])) / np.sum(self.JxW)

    def compute_total_by_volume(self, params, property="rho", normalize_by_volume=False):
        geom, mat = vmap_design_to_physical(params)
        model = getattr(self, f"model_{property}")
        prop_values = model(mat)
        weighted_prop = prop_values * geom # weight by geometry (density)
        total = np.sum(weighted_prop * np.sum(self.JxW, axis=[1,2])) # weight by quadrature weights
        if normalize_by_volume:
            total /= np.sum(self.JxW)
        return total