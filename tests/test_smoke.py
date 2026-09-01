"""Smoke tests for the stack.

Note the `eqx.partition` in `test_chex_needs_filtering`: chex tree assertions
assume array leaves and raise on an `eqx.Module` directly, because the module
carries its activation function as a leaf. Filter first.
"""

import chex
import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from src import models, sindy, ude


def test_autoencoder_roundtrip_shape():
    ae = models.Autoencoder(3, 2, 16, 2, key=jax.random.key(0))
    x = jnp.ones(3)
    assert ae.encode(x).shape == (2,)
    assert ae(x).shape == (3,)
    assert models.count_params(ae) > 0


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


def test_sindy_recovers_a_known_system():
    # dx/dt = -2 x + 3 x z  /  dz/dt = 1.5 z
    key = jax.random.key(0)
    X = jax.random.uniform(key, (200, 2), minval=0.5, maxval=2.0)
    dX = jnp.stack([-2.0 * X[:, 0] + 3.0 * X[:, 0] * X[:, 1], 1.5 * X[:, 1]], axis=1)

    model = sindy.SINDy(n_states=2, library={"degree": 2}, var_names=["x", "z"])
    xi = model.solve(X, dX, threshold=0.1)
    names = model.feature_names

    assert xi[names.index("x"), 0] == pytest.approx(-2.0, abs=1e-3)
    assert xi[names.index("x*z"), 0] == pytest.approx(3.0, abs=1e-3)
    assert xi[names.index("z"), 1] == pytest.approx(1.5, abs=1e-3)
    assert xi[names.index("1"), 1] == pytest.approx(0.0, abs=1e-6)


def test_sindy_solve_is_jittable():
    X = jax.random.uniform(jax.random.key(1), (50, 2), minval=0.5, maxval=2.0)
    dX = jnp.stack([-2.0 * X[:, 0], 1.5 * X[:, 1]], axis=1)
    model = sindy.SINDy(n_states=2, library={"degree": 2})
    theta = model._build_theta(X)
    xi = jax.jit(sindy._stlsq, static_argnames=("n_iters",))(
        theta, dX, model.library_mask, 0.1, 20
    )
    chex.assert_tree_all_finite(xi)


def test_sindy_single_spec_and_list_agree():
    X = jax.random.uniform(jax.random.key(3), (200, 2), minval=0.5, maxval=2.0)
    dX = jnp.stack([-2.0 * X[:, 0] + 3.0 * X[:, 0] * X[:, 1], 1.5 * X[:, 1]], axis=1)
    shared = sindy.SINDy(n_states=2, library={"degree": 2})
    listed = sindy.SINDy(n_states=2, library=[{"degree": 2}, {"degree": 2}])
    assert shared.feature_names == listed.feature_names
    chex.assert_trees_all_close(shared.solve(X, dX), listed.solve(X, dX))


def test_sindy_per_equation_libraries():
    """Each equation may only use terms inside its own library."""
    X = jax.random.uniform(jax.random.key(4), (300, 2), minval=0.5, maxval=2.0)
    dX = jnp.stack([-2.0 * X[:, 0] + 3.0 * X[:, 0] * X[:, 1], 1.5 * X[:, 1]], axis=1)

    # eq0: quadratic with bias.  eq1: linear, no bias.
    model = sindy.SINDy(
        n_states=2,
        library=[{"degree": 2}, {"degree": 1, "bias": False}],
        var_names=["x", "z"],
    )
    xi = model.solve(X, dX, threshold=0.1)
    names = model.feature_names

    assert names == ["1", "x", "z", "x*x", "x*z", "z*z"]
    assert not bool(model.library_mask[1, names.index("1")])  # eq1 has no bias
    assert not bool(model.library_mask[1, names.index("x*z")])  # eq1 is degree 1

    # eq0 still recovers its quadratic term; eq1 recovers only the linear one
    assert xi[names.index("x*z"), 0] == pytest.approx(3.0, abs=1e-3)
    assert xi[names.index("z"), 1] == pytest.approx(1.5, abs=1e-3)
    # every term outside an equation's library is exactly zero
    assert bool(jnp.all(jnp.abs(xi.T[~model.library_mask]) == 0.0))


def test_sindy_var_degree_caps_single_variable():
    """Level (ii): a variable's own power is capped, still under the total degree."""
    model = sindy.SINDy(
        n_states=2,
        library={"degree": 3, "var_degree": (2, 1)},
        var_names=["x", "z"],
    )
    allowed = [
        name
        for i, name in enumerate(model.feature_names)
        if bool(model.library_mask[0, i])
    ]
    assert "x*x*z" in allowed  # x twice, z once, total 3
    assert "x*x" in allowed
    assert "z*z" not in allowed  # z twice exceeds var_degree[1]
    assert "x*x*x" not in allowed  # x three times exceeds var_degree[0]


def test_sindy_exclude_drops_named_terms():
    """Level (iii): exponent vectors drop individual terms."""
    model = sindy.SINDy(
        n_states=2,
        library={"degree": 2, "exclude": [(1, 1), (0, 2)]},
        var_names=["x", "z"],
    )
    names = model.feature_names
    assert not bool(model.library_mask[0, names.index("x*z")])  # (1,1)
    assert not bool(model.library_mask[0, names.index("z*z")])  # (0,2)
    assert bool(model.library_mask[0, names.index("x*x")])  # untouched


def test_sindy_exclusions_vary_between_libraries():
    model = sindy.SINDy(
        n_states=2,
        library=[
            {"degree": 2, "exclude": [(1, 1)]},
            {"degree": 2, "exclude": [(2, 0)]},
        ],
        var_names=["x", "z"],
    )
    names = model.feature_names
    assert not bool(model.library_mask[0, names.index("x*z")])
    assert bool(model.library_mask[1, names.index("x*z")])
    assert bool(model.library_mask[0, names.index("x*x")])
    assert not bool(model.library_mask[1, names.index("x*x")])


def test_sindy_all_pruned_returns_finite_zeros():
    X = jax.random.uniform(jax.random.key(2), (50, 2), minval=0.5, maxval=2.0)
    dX = jnp.stack([-2.0 * X[:, 0], 1.5 * X[:, 1]], axis=1)
    model = sindy.SINDy(n_states=2, library={"degree": 2})
    xi = model.solve(X, dX, threshold=1e6)  # threshold prunes every term
    chex.assert_tree_all_finite(xi)
    assert bool(jnp.all(xi == 0))


def test_chex_needs_filtering_on_equinox_modules():
    """Documents the one real chex/Equinox sharp edge."""
    m = eqx.nn.MLP(2, 2, 8, 1, key=jax.random.key(0))

    with pytest.raises(TypeError):
        chex.assert_tree_all_finite(m)  # trips on the activation-function leaf

    params, _ = eqx.partition(m, eqx.is_inexact_array)
    chex.assert_tree_all_finite(params)  # fine
