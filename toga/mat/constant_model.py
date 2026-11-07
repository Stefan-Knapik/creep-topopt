import jax.numpy as jnp


class ConstantModel:
    def __init__(self, value):
        self.value = value

    def __call__(self, *args):
        if len(args) == 0:
            return jnp.asarray(self.value)

        first_arg = jnp.asarray(args[0])
        if first_arg.ndim <= 1:
            shape = (1,)
        else:
            shape = (first_arg.shape[0],)
        return jnp.full(shape, self.value)
