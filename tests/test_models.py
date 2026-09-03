"""Tests for `src.models`."""

import jax
import jax.numpy as jnp

from src import models


def test_autoencoder_roundtrip_shape():
    ae = models.Autoencoder(3, 2, 16, 2, key=jax.random.key(0))
    x = jnp.ones(3)
    assert ae.encode(x).shape == (2,)
    assert ae(x).shape == (3,)
    assert models.count_params(ae) > 0
