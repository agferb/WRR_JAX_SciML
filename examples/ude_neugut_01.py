from collections.abc import Callable

import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray, PyTree

SEED = 32
IN_DIM = 8
OUT_DIM = 1
N_SAMPLES = 100


class UDE_MLP(eqx.Module):

    layers: list
    activations: tuple = eqx.field(static=True)

    def __init__(
        self,
        sizes,
        activations,
        *,
        key: PRNGKeyArray,
    ):

        depth = len(sizes) - 1
        keys = jax.random.split(key, depth)
        self.layers = [
            eqx.nn.Linear(i, o, key=k) for i, o, k in zip(sizes[:-1], sizes[1:], keys)
        ]
        self.activations = tuple(activations)

    def __call__(self, X: Float[Array, "IN_DIM"], args=None) -> Float[Array, "OUT_DIM"]:

        for layer, act in zip(self.layers, self.activations):
            X = act(layer(X))
        return X


gkey = jax.random.PRNGKey(SEED)
key1, key2, gkey = jax.random.split(gkey, 3)
UDE_denit = UDE_MLP(
    sizes=(IN_DIM, 16, 4, OUT_DIM),
    activations=[jax.nn.tanh, jax.nn.leaky_relu, jax.nn.relu],
    key=key1,
)

X = jax.random.uniform(key2, (N_SAMPLES, IN_DIM))
Y = jax.vmap(UDE_denit)(X)

pass
