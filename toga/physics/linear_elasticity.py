import jax
import jax.numpy as np

from toga.fem.problem import Problem
from toga.mat.material_properties import design_to_physical, vmap_design_to_physical
from toga.opt import penalize


class Elasticity(Problem):
    """Linear elasticity (plane stress in 2D) with RAMP-penalized, composition-dependent E [MPa]."""

    POSSIBLE_MATERIAL_MODELS = ("E", "YS", "rho", "cost") # "nu", "alpha"

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

        available_models = self.available_material_models(material_models)
        
        if "E" not in available_models:
            raise ValueError("Elasticity requires an 'E' material model. "
                            f"Available models: {material_models.available_models()}")
            
        self.bind_material_models(material_models)
        
        self.penalize = lambda x: penalize(x, {'RAMP':8.0})
        
    def get_element_properties(self, mats):
        E = self.model_E(mats)
        nu = 0.3
        return E, nu
    
    def stress(self, u_grad, theta, penalize=False):
        geom, mats = design_to_physical(theta)
        E, nu = self.get_element_properties(mats)
        penal = self.penalize(geom)
        
        if penalize:
            E = E * penal
        
        epsilon = 0.5 * (u_grad + u_grad.T)
        
        if self.dim == 2:
            # Plane stress assumption for 2D elasticity
            eps11 = epsilon[0, 0]
            eps22 = epsilon[1, 1]
            eps12 = epsilon[0, 1]
            sig11 = E/(1 + nu)/(1 - nu)*(eps11 + nu*eps22) 
            sig22 = E/(1 + nu)/(1 - nu)*(nu*eps11 + eps22)
            sig12 = E/(1 + nu)*eps12
            sigma = np.array([[sig11, sig12], [sig12, sig22]])
        elif self.dim == 3:
            # 3D elasticity
            lmbda = E*nu/((1. + nu)*(1. - 2.*nu))
            mu = E/(2.*(1. + nu))
            sigma = lmbda * np.trace(epsilon) * np.eye(3) + 2 * mu * epsilon
        return sigma
        
    def get_tensor_map(self):
        return lambda u_grad, theta: self.stress(u_grad, theta, penalize=True)
                
    def set_params(self, params):
        full_params = np.ones((self.fe.num_cells, params.shape[1]))
        full_params = full_params.at[self.fe.flex_inds].set(params)
        thetas = np.repeat(full_params[:, None, :], self.fe.num_quads, axis=1)
        self.full_params = full_params
        self.internal_vars = [thetas]

    def compute_compliance(self, sol):
        """Compliance: work of the surface tractions on displacement ``sol``."""
        total = 0.
        for i in range(len(self.boundary_inds_list)):
            boundary_inds = self.boundary_inds_list[i]
            _, nanson_scale = self.fe.get_face_shape_grads(boundary_inds)
            u_face = sol[self.fe.cells][boundary_inds[:, 0]][:, None, :, :] * \
                    self.fe.face_shape_vals[boundary_inds[:, 1]][:, :, :, None]
            u_face = np.sum(u_face, axis=2)
            subset_quad_points = self.physical_surface_quad_points[i]
            neumann_fn = self.get_surface_maps()[i]
            traction = -jax.vmap(jax.vmap(neumann_fn))(u_face, subset_quad_points)
            total += np.sum(traction * u_face * nanson_scale[:, :, None])
        return total
    
    def get_von_mises_stress_fn(self):
        def vm_stress_fn_helper(sigma):
            dim = 3
            s_dev = sigma - 1./dim*np.trace(sigma)*np.eye(dim)
            vm_s = np.sqrt(3./2.*np.sum(s_dev*s_dev))
            return vm_s
        
        if self.dim == 2: # from 2D plane stress to 3D for von Mises calculation
            def vm_stress_fn(u_grad, theta):
                sigma2d = self.stress(u_grad, theta, penalize=False)
                zero = np.zeros_like(sigma2d[0, 0])
                sigma3d = np.stack((
                    np.stack((sigma2d[0, 0], sigma2d[0, 1], zero)),
                    np.stack((sigma2d[1, 0], sigma2d[1, 1], zero)),
                    np.stack((zero, zero, zero)),
                ))
                return vm_stress_fn_helper(sigma3d)
        elif self.dim == 3:
            def vm_stress_fn(u_grad, theta):
                sigma = self.stress(u_grad, theta, penalize=False)
                return vm_stress_fn_helper(sigma)
        return vm_stress_fn

    def compute_von_mises_stress(self, sol, params=None):
        # (num_cells, 1, num_nodes, vec, 1) * (num_cells, num_quads, num_nodes, 1, dim) -> (num_cells, num_quads, num_nodes, vec, dim) 
        u_grads = np.take(sol, self.fe.cells, axis=0)[:, None, :, :, None] * self.shape_grads[:, :, :, None, :] 
        u_grads = np.sum(u_grads, axis=2) # (num_cells, num_quads, vec, dim) 
        vm_stress_fn = self.get_von_mises_stress_fn()
        if params is None:
            internal_vars = self.internal_vars
        else:
            full_params = np.ones((self.fe.num_cells, params.shape[1]))
            full_params = full_params.at[self.fe.flex_inds].set(params)
            thetas = np.repeat(full_params[:, None, :], self.fe.num_quads, axis=1)
            internal_vars = [thetas]
        vm_stress = jax.vmap(jax.vmap(vm_stress_fn))(u_grads, *internal_vars) # (num_cells, num_quads)
        volume_avg_vm_stress = np.sum(vm_stress * self.JxW[:,0,:], axis=1) / np.sum(self.JxW, axis=[1,2]) # (num_cells,)
        return volume_avg_vm_stress

    def compute_volume_fraction(self, params):
        geom, _ = vmap_design_to_physical(params)
        return np.sum(geom * np.sum(self.JxW, axis=[1, 2])) / np.sum(self.JxW)

    def compute_total_by_volume(self, params, property="rho", normalize_by_volume=False):
        geom, mat = vmap_design_to_physical(params)
        model = getattr(self, f"model_{property}")
        prop_values = model(mat)
        weighted_prop = prop_values * geom # weight by geometry (density)
        total = np.sum(weighted_prop * np.sum(self.JxW, axis=[1,2])) # weight by quadrature weights
        if normalize_by_volume:
            total /= np.sum(self.JxW)
        return total