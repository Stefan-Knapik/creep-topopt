from typing import Dict

import jax
import jax.numpy as np
import optimistix as optx

from toga.fem.problem import Problem
from toga.fem.solver import ad_wrapper
from toga.mat.material_properties import design_to_physical, vmap_design_to_physical
from toga.opt import penalize
from toga.utils_for_creep import pad_to_3x3, safe_divide, safe_sqrt, trim_to_dim


class CreepImplicit(Problem):
    """Isothermal Norton creep, backward-Euler in time (stress in MPa, time in s)."""

    POSSIBLE_MATERIAL_MODELS = ("E", "YS", "rho", "cost", "A", "logA", "n", "Q") # "nu"

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
        self.penalize_E = lambda x: penalize(x, {'RAMP':8.0})
        
        # Make low-density regions creep faster
        self.penalize_A = lambda x: 1 / penalize(x, {'RAMP':8.0})**2.0 

        self.creep_solver = optx.Newton(rtol=1e-4, atol=1e-10)
        self.creep_solver_max_steps = 100

    def get_tensor_map(self):
        return lambda *x: self.creep_stress_strain_map(*x)[0]

    def strain(self, u_grad):
        return 0.5 * (u_grad + u_grad.T)

    def stress(self, epsilon, theta):
        geom, mats = design_to_physical(theta)
        E = self.model_E(mats)
        nu = 0.3
        penal = self.penalize_E(geom)
        E = E * penal

        if self.dim == 2:
            # Plane stress assumption for 2D elasticity
            eps11 = epsilon[0, 0]
            eps22 = epsilon[1, 1]
            eps12 = epsilon[0, 1]
            sig11 = E / (1 + nu) / (1 - nu) * (eps11 + nu * eps22)
            sig22 = E / (1 + nu) / (1 - nu) * (nu * eps11 + eps22)
            sig12 = E / (1 + nu) * eps12
            sigma = np.array([[sig11, sig12], [sig12, sig22]])
        elif self.dim == 3:
            # 3D elasticity
            lmbda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
            mu = E / (2.0 * (1.0 + nu))
            sigma = lmbda * np.trace(epsilon) * np.eye(3) + 2 * mu * epsilon
        else:
            raise ValueError(f"Unsupported dimension {self.dim}")
        return sigma

    def creep_rate_from_sigma(self, sigma, A, n):
        sigma_3d = pad_to_3x3(sigma)
        s_dev = sigma_3d - (np.trace(sigma_3d) / 3.0) * np.eye(3)
        s_norm = safe_sqrt(1.5 * np.sum(s_dev * s_dev))
        n_d = 1.5 * safe_divide(s_dev, s_norm)
        n_d = trim_to_dim(n_d, self.dim)
        return A * s_norm**n * n_d

    def creep_prefactor(self, mats):
        """Creep pre-exponential factor A.

        A material set may supply 'A' directly, or 'logA' = log10(A). The latter
        lets the property model interpolate the exponent linearly, which keeps
        the fit meaningful when A spans orders of magnitude between materials.
        """
        if hasattr(self, "model_logA"):
            return 10.0 ** self.model_logA(mats)
        return self.model_A(mats)

    def creep_stress_strain_map(self, u_grad, sigma_old, epsilon_creep_old, theta, dt):
        del sigma_old

        geom, mats = design_to_physical(theta)
        A = self.creep_prefactor(mats)
        n = self.model_n(mats)

        # Make low-density regions creep faster so gray regions are less attractive
        # during topology optimization under sustained loading.
        penal = self.penalize_A(geom)
        A = A * penal

        epsilon_new = self.strain(u_grad)

        def creep_residual(epsilon_creep_inc, args):
            epsilon_trial, epsilon_creep_prev, theta_local, A_local, n_local, dt_local = args
            epsilon_elastic = epsilon_trial - epsilon_creep_prev - epsilon_creep_inc
            sigma_new = self.stress(epsilon_elastic, theta_local)
            creep_rate = self.creep_rate_from_sigma(sigma_new, A_local, n_local)
            return epsilon_creep_inc - dt_local * creep_rate

        epsilon_creep_inc_0 = np.zeros_like(epsilon_creep_old)
        solver_args = (epsilon_new, epsilon_creep_old, theta, A, n, dt)
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
        sigma_new = self.stress(epsilon_elastic, theta)
        return sigma_new, epsilon_creep_inc

    def stress_strain_fns(self):
        return jax.vmap(jax.vmap(self.creep_stress_strain_map))

    def update_stress_strain(self, sol, params):
        u_grads = self.fe.sol_to_grad(sol)
        sigmas_old, epsilons_creep_old, thetas, dts = params

        vmap_creep_stress_strain = self.stress_strain_fns()
        stress, creep_strain_inc = vmap_creep_stress_strain(u_grads, sigmas_old, 
                                                            epsilons_creep_old, 
                                                            thetas, dts)
        stress_update = np.squeeze(stress)
        creep_strain_update = epsilons_creep_old + creep_strain_inc
        return stress_update, creep_strain_update, thetas, dts

    def init_params(self, theta):
        # Initializing stress and cumulative creep strain.
        sigmas_old = np.zeros((len(self.fe.cells), self.fe.num_quads, self.fe.vec, self.dim))
        epsilons_creep_old = np.zeros_like(sigmas_old)

        # Initializing design variables.
        full_params = np.ones((self.num_cells, theta.shape[1]))
        full_params = full_params.at[self.fe.flex_inds].set(theta)
        thetas = np.repeat(full_params[:, None, :], self.fe.num_quads, axis=1)
        dts = np.zeros((len(self.fe.cells), self.fe.num_quads))
        self.full_params = full_params

        return sigmas_old, epsilons_creep_old, thetas, dts

    def set_params(self, params):
        body_params, surface_params = params
        self.internal_vars = body_params
        self.internal_vars_surfaces = [[surface_params]]

    def compute_vol_avg_stress(self, body_params):
        sigmas, _, _, _ = body_params
        sigma = np.sum(
            sigmas.reshape(-1, self.fe.vec, self.dim) * 
            self.fe.JxW.reshape(-1)[:, None, None], axis=0)
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
        end_time: float,
        num_time_steps: int,
        fwd_solver_options: Dict,
        vjp_solver_options: Dict,
    ):
        """March from t=0 to end_time [s] in num_time_steps steps; returns the solutions and internal variables of every step."""
        fwd_pred = ad_wrapper(
            self,
            solver_options=fwd_solver_options,
            adjoint_solver_options=vjp_solver_options,
        )

        body_params = self.init_params(theta)

        time_vector = np.linspace(0.0, end_time, num_time_steps + 1)
        self.end_time = end_time

        # Constant across all time steps; captured in closure below.
        surface_params = [
            np.ones((surf_quads.shape[0], self.fe.num_face_quads))
            for surf_quads in self.physical_surface_quad_points
        ]

        # dts[0] = 0 (initial elastic step), dts[i] = time_vector[i] - time_vector[i-1].
        dts = np.concatenate([np.array([0.0]), np.diff(time_vector)])

        def scan_fn(carry, dt):
            bp = carry
            sigmas_old, epsilons_creep_old, thetas, _ = bp
            bp = (
                sigmas_old,
                epsilons_creep_old,
                thetas,
                np.full((len(self.fe.cells), self.fe.num_quads), dt),
            )
            sol_list = fwd_pred([bp, surface_params])
            bp = self.update_stress_strain(sol_list[0], bp)
            return bp, (sol_list, bp)

        _, (full_sol_list, body_params_list) = jax.lax.scan(
            scan_fn, body_params, dts
        )

        return full_sol_list, body_params_list
    
    def compute_volume_fraction(self, params):
        geom, _ = vmap_design_to_physical(params)
        return np.sum(geom * np.sum(self.JxW, axis=[1, 2])) / np.sum(self.JxW)
