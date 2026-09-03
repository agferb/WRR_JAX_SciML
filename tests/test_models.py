"""Smoke tests for the models.

Note the `eqx.partition` in `test_chex_needs_filtering`: chex tree assertions
assume array leaves and raise on an `eqx.Module` directly, because the module
carries its activation function as a leaf. Filter first.
"""

import chex
import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from src import models


def test_autoencoder_roundtrip_shape():
    ae = models.Autoencoder(3, 2, 16, 2, key=jax.random.key(0))
    x = jnp.ones(3)
    assert ae.encode(x).shape == (2,)
    assert ae(x).shape == (3,)
    assert models.count_params(ae) > 0
