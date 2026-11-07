import jax
import jax.numpy as jnp


@jax.custom_jvp
def safe_norm(x): 
    # (N, M, d) -> (N, M)
    return jnp.linalg.norm(x, axis=-1)

@safe_norm.defjvp
def safe_norm_jvp(primals, tangents):
    x, = primals # (N, M, d)
    x_dot, = tangents # (N, M, d)

    y = safe_norm(x) # (N, M)
    denom = jnp.linalg.norm(x, axis=-1, keepdims=True) # (N, M, 1)

    # Safe directional derivative:
    # If denom == 0, set the gradient to zero instead of NaN
    unit = jnp.where(denom > 0, x / denom, 0.0) # (N, M, d)
    y_dot = jnp.sum(unit * x_dot, axis=-1) # (N, M)
    return y, y_dot


def pairwise_distances(x1, x2):
    """
    Compute pairwise Euclidean distances.

    x1: (N, d)
    x2: (M, d)
    returns: (N, M)
    """
    diff = x1[:, None, :] - x2[None, :, :] # (N, M, d)
    return safe_norm(diff) # (N, M)


def rbf_kernel(x1, x2, kernel="linear", epsilon=1.0):
    """
    Compute pairwise RBF kernel matrix.

    Supported kernels:
      - linear:       phi(r) = r
      - gaussian:     phi(r) = exp(-(epsilon * r)^2)
      - multiquadric: phi(r) = sqrt(1 + (epsilon * r)^2)

    x1: (N, d)
    x2: (M, d)
    returns: (N, M)
    """
    r = pairwise_distances(x1, x2)

    if kernel == "linear":
        return r
    elif kernel == "gaussian":
        return jnp.exp(-(epsilon * r) ** 2)
    elif kernel == "multiquadric":
        return jnp.sqrt(1.0 + (epsilon * r) ** 2)
    else:
        raise ValueError(
            f"Unsupported kernel '{kernel}'. "
            "Choose from: 'linear', 'gaussian', 'multiquadric'."
        )


class RBFInterpolator:
    def __init__(self, x_train, y_train, kernel="linear", epsilon=1.0, reg=0.0, input_scale=None):
        """
        x_train: (N, d)
        y_train: (N,) or (N, m)
        kernel:  'linear', 'gaussian', or 'multiquadric'
        epsilon: shape parameter for gaussian/multiquadric
        reg:     diagonal regularization
        input_scale: optional scaling factors for input features
        """
        self.x_train = jnp.asarray(x_train)
        self.y_train = jnp.asarray(y_train)
        self.kernel = kernel
        self.epsilon = epsilon
        self.reg = reg
        self.input_scale = input_scale
        self.input_dim = self.x_train.shape[1]

        if self.input_scale is not None:
            assert self.input_scale.shape == (self.x_train.shape[1],), "input_scale must have shape (d,)"
            self.x_train = self.x_train / self.input_scale

        K = rbf_kernel(
            self.x_train,
            self.x_train,
            kernel=self.kernel,
            epsilon=self.epsilon,
        )
        n = K.shape[0]
        K_reg = K + self.reg * jnp.eye(n, dtype=K.dtype)

        self.weights = jnp.linalg.solve(K_reg, self.y_train)

    def __call__(self, x_query):
        """
        x_query: (M, d) or (d,)
        returns: (M,) or (M, m)
        """
        x_query = jnp.asarray(x_query)
        squeeze_output = x_query.ndim == 1
        if squeeze_output:
            x_query = x_query[None, :]
            
        if self.input_scale is not None:
            x_query = x_query / self.input_scale

        Kq = rbf_kernel(
            x_query,
            self.x_train,
            kernel=self.kernel,
            epsilon=self.epsilon,
        )
        
        y_pred = Kq @ self.weights

        if squeeze_output:
            return jnp.squeeze(y_pred, axis=0)
        return y_pred
