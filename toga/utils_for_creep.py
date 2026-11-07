"""Small numerically safe helpers for the creep models."""
import jax.numpy as jnp


def safe_sqrt(x: jnp.ndarray,
              ) -> jnp.ndarray:
  """Square root that returns 0 for x <= 0, with a finite (zero) gradient there."""
  positive = x > 0.
  return jnp.where(positive, jnp.sqrt(jnp.where(positive, x, 1.)), 0.)


def safe_divide(x: jnp.ndarray,
                y: jnp.ndarray
                ) -> jnp.ndarray:
  """Safe division function that returns 0 for division by zero.
  
  Args:
    x: The numerator.
    y: The denominator.
  
  Returns: x/y if y is not zero, 0 otherwise.
  """
  return jnp.where(y == 0., 0., x/y)

def pad_to_3x3(sigma):
    if sigma.ndim >= 3:
        sigma = jnp.squeeze(sigma)
    if sigma.shape == (3, 3):
        return sigma
    elif sigma.shape == (2, 2):
        sigma_padded = jnp.zeros((3, 3))
        sigma_padded = sigma_padded.at[:2,:2].set(sigma)
        return sigma_padded
    else:
        raise ValueError("Input matrix must be 2x2 or 3x3.")
      
def trim_to_dim(n_d, dim):
    if dim == 3:
        return n_d
    elif dim == 2:
        return n_d[:2,:2]
    else:
        raise ValueError("Dimension must be 2 or 3.")

