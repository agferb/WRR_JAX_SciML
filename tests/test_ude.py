"""Smoke tests for the ude.

Note the `eqx.partition` in `test_chex_needs_filtering`: chex tree assertions
assume array leaves and raise on an `eqx.Module` directly, because the module
carries its activation function as a leaf. Filter first.
"""

import chex
import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from src import ude


def test_ude_is_a_pytree_and_solves():
    model = ude.LotkaVolterraUDE(1.3, 1.8, key=jax.random.key(0))
    ts = jnp.linspace(0.0, 1.0, 10)
    ys = ude.solve(model, jnp.array([1.0, 1.0]), ts)
    chex.assert_shape(ys, (10, 2))
    chex.assert_tree_all_finite(ys)


def _grads(model):
    ts = jnp.linspace(0.0, 1.0, 10)
    target = ude.solve(ude.lotka_volterra(), jnp.array([1.0, 1.0]), ts)

    @eqx.filter_grad
    def loss(m):
        return jnp.mean((ude.solve(m, jnp.array([1.0, 1.0]), ts) - target) ** 2)

    return loss(model)


def test_ude_gradients_flow_to_all_parts():
    grads = _grads(ude.LotkaVolterraUDE(1.3, 1.8, key=jax.random.key(0), gain=0.1))
    assert jnp.abs(grads.alpha) > 0, "mechanistic parameter got no gradient"
    assert jnp.abs(grads.gain) > 0, "gain got no gradient"
    net_grads = jax.tree_util.tree_leaves(eqx.filter(grads.net, eqx.is_inexact_array))
    assert any(jnp.abs(g).sum() > 0 for g in net_grads), "closure got no gradient"


def test_zero_gain_starves_the_network():
    """Why the gain must not start at 0: `d/dnet = gain * ...` vanishes."""
    grads = _grads(ude.LotkaVolterraUDE(1.3, 1.8, key=jax.random.key(0), gain=0.0))
    net_grads = jax.tree_util.tree_leaves(eqx.filter(grads.net, eqx.is_inexact_array))
    assert all(jnp.abs(g).sum() == 0 for g in net_grads)
    assert jnp.abs(grads.gain) > 0  # only the gain itself can move


def test_closure_includes_the_gain():
    model = ude.LotkaVolterraUDE(1.3, 1.8, key=jax.random.key(0), gain=0.1)
    y = jnp.array([1.0, 2.0])
    chex.assert_trees_all_close(model.closure(y), model.gain * model.net(y))
    # bound methods of an eqx.Module are PyTrees, so this vmaps
    chex.assert_shape(jax.vmap(model.closure)(jnp.ones((4, 2))), (4, 2))


def test_gain_scales_the_closure():
    ts = jnp.linspace(0.0, 1.0, 5)
    y0 = jnp.array([1.0, 1.0])
    off = ude.LotkaVolterraUDE(1.3, 1.8, key=jax.random.key(0), gain=0.0)
    on = ude.LotkaVolterraUDE(1.3, 1.8, key=jax.random.key(0), gain=1.0)
    assert not jnp.allclose(ude.solve(off, y0, ts), ude.solve(on, y0, ts))


def test_freeze_mechanistic():
    model = ude.LotkaVolterraUDE(1.3, 1.8, key=jax.random.key(0))
    spec = ude.freeze_mechanistic(model)
    assert spec.alpha is False and spec.delta is False
    assert spec.gain is True
    assert any(jax.tree_util.tree_leaves(spec.net))

    frozen = ude.freeze_mechanistic(model, train_gain=False)
    assert frozen.gain is False
    assert any(jax.tree_util.tree_leaves(frozen.net))


def test_multiple_shooting_windows():
    ts = jnp.arange(10.0)
    ys = jnp.stack([ts, 2 * ts], axis=1)
    tw, yw = ude.multiple_shooting_windows(ts, ys, length=4, stride=2)
    chex.assert_shape(tw, (4, 4))
    chex.assert_shape(yw, (4, 4, 2))
    # each window starts `stride` samples after the previous one
    assert tw[0, 0] == 0.0 and tw[1, 0] == 2.0
    # windows are contiguous slices of the original trajectory
    chex.assert_trees_all_close(yw[0], ys[0:4])


def test_chex_needs_filtering_on_equinox_modules():
    """Documents the one real chex/Equinox sharp edge."""
    m = eqx.nn.MLP(2, 2, 8, 1, key=jax.random.key(0))

    with pytest.raises(TypeError):
        chex.assert_tree_all_finite(m)  # trips on the activation-function leaf

    params, _ = eqx.partition(m, eqx.is_inexact_array)
    chex.assert_tree_all_finite(params)  # fine
