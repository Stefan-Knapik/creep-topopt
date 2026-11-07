from typing import Dict

import jax
import jax.numpy as np
import optimistix as optx

from toga.fem.problem import Problem
from toga.fem.solver import ad_wrapper
from toga.mat.material_properties import design_to_physical, vmap_design_to_physical
from toga.opt import penalize
from toga.utils_for_creep import pad_to_3x3, safe_divide, safe_sqrt, trim_to_dim


class ThermalCreepImplicit(Problem):
    """Thermo-mechanical Norton creep with Arrhenius factor, backward-Euler in time (MPa, s, °C)."""

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
        self.fe = self.fes[0]
        self.fe.flex_inds = np.arange(len(self.fe.cells))

        self.bind_material_models(material_models)
        if hasattr(self, "model_A") == hasattr(self, "model_logA"):
            raise ValueError(
                "Creep needs exactly one model for the pre-exponential factor: "
                "supply 'A' or 'logA' (log10 of A) in the property configs, not both."
            )

        self.penalize_E = lambda x: penalize(x, {"RAMP": 8.0})
        self.penalize_A = lambda x: 1.0 / penalize(x, {"RAMP": 8.0}) ** 2.0

        self.creep_solver = optx.Newton(rtol=1e-4, atol=1e-10)
        self.creep_solver_max_steps = 100

    def get_tensor_map(self):
        return lambda *x: self.creep_stress_strain_map(*x)[0]

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

        E = E * self.penalize_E(geom)

        delta_temperature = T - reference_T
        epsilon_thermal = alpha * delta_temperature * np.eye(self.dim)
        epsilon_mechanical = epsilon - epsilon_thermal

        if self.dim == 2:
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
            sigma = (
                lmbda * np.trace(epsilon_mechanical) * np.eye(3)
                + 2.0 * mu * epsilon_mechanical
            )
        else:
            raise ValueError(f"Unsupported dimension: {self.dim}")

        return sigma

    def creep_rate_from_sigma(self, sigma, A, n, Q, T):
        sigma_3d = pad_to_3x3(sigma)
        s_dev = sigma_3d - (np.trace(sigma_3d) / 3.0) * np.eye(3)
        s_norm = safe_sqrt(1.5 * np.sum(s_dev * s_dev))
        n_d = 1.5 * safe_divide(s_dev, s_norm)
        n_d = trim_to_dim(n_d, self.dim)
        # Temperatures are carried in Celsius throughout (BCs, thermal solve,
        # thermal expansion), but the Arrhenius term needs absolute temperature.
        T_kelvin = np.maximum(T + self.CELSIUS_TO_KELVIN, 1.0)
        arrhenius = np.exp(-Q / (self.GAS_CONSTANT * T_kelvin))
        return A * arrhenius * s_norm**n * n_d

    def creep_stress_strain_map(
        self,
        u_grad,
        sigma_old,
        epsilon_creep_old,
        theta,
        T,
        reference_T,
        dt,
    ):
        del sigma_old

        geom, mats = design_to_physical(theta)
        _, _, A, n, _, Q = self.get_element_properties(mats, T)

        A = A * self.penalize_A(geom)
        epsilon_new = self.strain(u_grad)

        def creep_residual(epsilon_creep_inc, args):
            (
                epsilon_trial,
                epsilon_creep_prev,
                theta_local,
                T_local,
                reference_T_local,
                A_local,
                n_local,
                Q_local,
                dt_local,
            ) = args
            epsilon_elastic = epsilon_trial - epsilon_creep_prev - epsilon_creep_inc
            sigma_new = self.stress(
                epsilon_elastic,
                theta_local,
                T_local,
                reference_T_local,
            )
            creep_rate = self.creep_rate_from_sigma(
                sigma_new,
                A_local,
                n_local,
                Q_local,
                T_local,
            )
            return epsilon_creep_inc - dt_local * creep_rate

        epsilon_creep_inc_0 = np.zeros_like(epsilon_creep_old)
        solver_args = (
            epsilon_new,
            epsilon_creep_old,
            theta,
            T,
            reference_T,
            A,
            n,
            Q,
            dt,
        )
        sol = optx.root_find(
            creep_residual,
            self.creep_solver,
            epsilon_creep_inc_0,
            args=solver_args,
            max_steps=self.creep_solver_max_steps,
            throw=False,
        )

        epsilon_creep_inc = sol.value
        epsilon_elastic = epsilon_new - epsilon_creep_old - epsilon_creep_inc
        sigma_new = self.stress(epsilon_elastic, theta, T, reference_T)
        return sigma_new, epsilon_creep_inc

    def stress_strain_fns(self):
        return jax.vmap(jax.vmap(self.creep_stress_strain_map))

    def update_stress_strain(self, sol, params):
        u_grads = self.fe.sol_to_grad(sol)

        sigmas_old, epsilons_creep_old, thetas, T_quad, ref_T_quad, dts = params

        vmap_creep_stress_strain = self.stress_strain_fns()
        stress, creep_strain_inc = vmap_creep_stress_strain(
            u_grads,
            sigmas_old,
            epsilons_creep_old,
            thetas,
            T_quad,
            ref_T_quad,
            dts,
        )
        stress_update = np.squeeze(stress)
        creep_strain_update = epsilons_creep_old + creep_strain_inc
        return stress_update, creep_strain_update, thetas, T_quad, ref_T_quad, dts

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

        dts = np.zeros((len(self.fe.cells), self.fe.num_quads))
        return sigmas_old, epsilons_creep_old, thetas, T_quad, ref_T_quad, dts

    def set_params(self, params):
        body_params, surface_params = params
        self.internal_vars = body_params
        self.internal_vars_surfaces = [[surface_params]]

    def compute_vol_avg_stress(self, body_params):
        sigmas, _, _, _, _, _ = body_params

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
        surface_params=None,
        save_history: bool = True,
    ):
        """Implicit creep time stepping.

        Returns the displacement at every step and the body params (stress, creep strain, ...)
        at every step, or with ``save_history=False`` only at the first and last step. The
        stress history is the part of GPU memory that grows with ``num_time_steps``.
        """
        fwd_pred = ad_wrapper(
            self,
            solver_options=fwd_solver_options,
            adjoint_solver_options=vjp_solver_options,
        )

        body_params = self.init_params(theta, T, reference_T)

        time_vector = np.linspace(0.0, end_time, num_time_steps + 1)
        self.end_time = end_time

        # Constant across all time steps; captured in closure below. Passing
        # surface_params in (one entry per surface) makes the surface load a traced
        # argument of the solve rather than a constant closed over by get_surface_maps,
        # so a load that changes between runs does not force a recompilation.
        if surface_params is None:
            surface_params = [
                np.ones((surf_quads.shape[0], self.fe.num_face_quads))
                for surf_quads in self.physical_surface_quad_points
            ]

        # dts[0] = 0 (initial elastic step), dts[i] = time_vector[i] - time_vector[i-1].
        dts = np.concatenate([np.array([0.0]), np.diff(time_vector)])

        def step(bp, dt):
            sigmas_old, epsilons_creep_old, thetas, T_quad, ref_T_quad, _ = bp
            bp = (
                sigmas_old,
                epsilons_creep_old,
                thetas,
                T_quad,
                ref_T_quad,
                np.full((len(self.fe.cells), self.fe.num_quads), dt),
            )
            sol_list = fwd_pred([bp, surface_params])
            return sol_list, self.update_stress_strain(sol_list[0], bp)

        if save_history:
            def scan_fn(carry, dt):
                sol_list, bp = step(carry, dt)
                return bp, (sol_list, bp)

            _, (full_sol_list, body_params_list) = jax.lax.scan(
                scan_fn, body_params, dts
            )
            return full_sol_list, body_params_list

        def scan_fn(carry, dt):
            sol_list, bp = step(carry, dt)
            return bp, sol_list

        sol_first, bp_first = step(body_params, dts[0])
        bp_last, sols = jax.lax.scan(scan_fn, bp_first, dts[1:])
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