"""Neural network components built on Equinox.

Everything here is an `eqx.Module`, which means everything here is a PyTree:
it can be passed straight through `jax.vmap`, `jax.lax.scan`, or a Diffrax
solver without any wrapping.
"""

from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray


class Autoencoder(eqx.Module):
    """A plain MLP autoencoder.

    Useful on its own, and as the coordinate transform in a latent-space
    dynamics model: encode to `latent_size`, learn dynamics there, decode back.
    """

    encoder: eqx.nn.MLP
    decoder: eqx.nn.MLP

    def __init__(
        self,
        data_size: int,
        latent_size: int,
        width_size: int,
        depth: int,
        *,
        key: PRNGKeyArray,
        activation: Callable = jax.nn.softplus,
    ):
        ekey, dkey = jax.random.split(key)
        self.encoder = eqx.nn.MLP(
            data_size, latent_size, width_size, depth, activation=activation, key=ekey
        )
        self.decoder = eqx.nn.MLP(
            latent_size, data_size, width_size, depth, activation=activation, key=dkey
        )

    def encode(self, x: Float[Array, " data"]) -> Float[Array, " latent"]:
        return self.encoder(x)

    def decode(self, z: Float[Array, " latent"]) -> Float[Array, " data"]:
        return self.decoder(z)

    def __call__(self, x: Float[Array, " data"]) -> Float[Array, " data"]:
        return self.decoder(self.encoder(x))


def count_params(model: eqx.Module) -> int:
    """Number of trainable (inexact array) scalars in a model."""
    params = eqx.filter(model, eqx.is_inexact_array)
    return sum(x.size for x in jax.tree_util.tree_leaves(params))
