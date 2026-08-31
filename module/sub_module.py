import jax
import jax.numpy as jnp


class LinearGuy:

    def __init__(self, W: jnp.array, b: jnp.array) -> None:
        self.W = W
        self.b = b

    def __call__(self, X: jnp.array) -> jnp.array:
        Y = jax.vmap(
            _dot,
            in_axes=(0, None, None),
        )(X, self.W, self.b)
        return Y


@jax.jit
def _dot(x, w, b) -> jnp.array:
    y = w @ x + b
    return y
