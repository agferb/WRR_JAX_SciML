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
    theta = model._build_theta(X, jnp.zeros(X.shape[0]))
    xi = jax.jit(sindy._stlsq, static_argnames=("max_iters",))(
        theta[None], dX[None], model.library_mask[None], 0.1, 20
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


def test_sindy_rejects_unknown_library_keys():
    """The guard that would have caught the `bias`/`include_bias` drift."""
    with pytest.raises(ValueError, match="unknown library key"):
        sindy.SINDy(n_states=2, library={"degree": 2, "interaction_degree": 1})


def test_sindy_rejects_a_library_list_of_the_wrong_length():
    with pytest.raises(AssertionError):
        sindy.SINDy(n_states=3, library=[{"degree": 2}, {"degree": 1}])


def test_stlsq_stops_once_the_active_set_settles():
    """`max_iters` is a cap, so more iterations cannot change a settled fit."""
    X = jax.random.uniform(jax.random.key(5), (200, 2), minval=0.5, maxval=2.0)
    dX = jnp.stack([-2.0 * X[:, 0] + 3.0 * X[:, 0] * X[:, 1], 1.5 * X[:, 1]], axis=1)
    model = sindy.SINDy(n_states=2, library={"degree": 3})
    chex.assert_trees_all_equal(
        model.solve(X, dX, max_iters=20), model.solve(X, dX, max_iters=200)
    )


def test_sindy_interactions_degree_matches_the_spec():
    """The library from `prompt.md`, plus the bare derivative `bias` no longer gates."""
    model = sindy.SINDy(
        n_states=2,
        library={"degree": 2, "interactions_degree": 1, "bias": False},
        implicit=True,
    )
    allowed = [
        name
        for name, ok in zip(model.feature_names, model.library_mask[0])
        if bool(ok)
    ]
    assert allowed == [
        "x0",
        "x1",
        "x0*x0",
        "x0*x1",
        "x1*x1",
        "dx_k",
        "x0*dx_k",
        "x1*dx_k",
    ]


def test_sindy_bias_does_not_gate_the_bare_derivative():
    """`bias` governs the constant column only; the derivative is level (iv)."""
    spec = {"degree": 2, "interactions_degree": 1}
    with_bias = sindy.SINDy(n_states=2, library=spec)
    without = sindy.SINDy(n_states=2, library={**spec, "bias": False})
    assert bool(with_bias.library_mask[0][with_bias.feature_names.index("1")])
    assert not bool(without.library_mask[0][without.feature_names.index("1")])
    for model in (with_bias, without):
        i = model.feature_names.index("dx_k")
        assert bool(model.library_mask[0][i]), "the bare derivative must survive"


def test_sindy_interactions_degree_zero_gives_only_the_bare_derivative():
    model = sindy.SINDy(n_states=2, library={"degree": 2, "interactions_degree": 0})
    derivative_terms = [n for n in model.feature_names if "dx_k" in n]
    assert derivative_terms == ["dx_k"]


def test_sindy_no_mixed_or_higher_order_derivative_terms():
    model = sindy.SINDy(
        n_states=3,
        library={"degree": 2, "interactions_degree": 2},
        var_names=["x", "y", "z"],
    )
    assert all(c.count(3) <= 1 for c in model._combos)  # index 3 is the sentinel
    for i, own in enumerate(["dx/dt", "dy/dt", "dz/dt"]):
        foreign = [d for d in ["dx/dt", "dy/dt", "dz/dt"] if d != own]
        names = " ".join(model.feature_names_for(i))
        assert all(d not in names for d in foreign)


def test_sindy_var_interactions_degree_caps_the_interaction_family_only():
    model = sindy.SINDy(
        n_states=2,
        library={
            "degree": 2,
            "interactions_degree": 2,
            "var_interactions_degree": (2, 0),
        },
    )
    allowed = [
        name
        for name, ok in zip(model.feature_names, model.library_mask[0])
        if bool(ok)
    ]
    assert "x1*x1" in allowed  # untouched: that is `var_degree`'s family
    assert "x0*x0*dx_k" in allowed
    assert "x1*dx_k" not in allowed


def test_sindy_short_exclude_vectors_still_work():
    """A length-`n_states` exclusion keeps its meaning in an implicit library."""
    short = sindy.SINDy(
        n_states=2, library={"degree": 2, "interactions_degree": 1, "exclude": [(1, 1)]}
    )
    padded = sindy.SINDy(
        n_states=2,
        library={"degree": 2, "interactions_degree": 1, "exclude": [(1, 1, 0)]},
    )
    chex.assert_trees_all_equal(short.library_mask, padded.library_mask)
    assert not bool(short.library_mask[0][short.feature_names.index("x0*x1")])


def test_sindy_library_terms_lists_what_each_equation_holds():
    model = sindy.SINDy(
        n_states=2,
        library=[{"degree": 2}, {"degree": 1}],
        var_names=["x", "z"],
    )
    text = model.library_terms()
    assert text.splitlines()[0].startswith("dx: 1, x, z, x*x")
    assert text.splitlines()[1] == "dz: 1, x, z"


# --- SINDy-PI (parallel implicit) ------------------------------------------

JX, VMAX, KM = 0.6, 1.5, 0.3


def _michaelis_menten(key, n=300):
    x = jax.random.uniform(key, (n, 1), minval=0.2, maxval=3.0)
    return x, JX - VMAX * x / (KM + x)


def test_sindy_pi_recovers_michaelis_menten():
    """dx/dt = jx - Vmax x/(Km + x): no explicit polynomial library can hold it."""
    X, dX = _michaelis_menten(jax.random.key(0))
    held_out = _michaelis_menten(jax.random.key(1), n=150)

    model = sindy.SINDy(
        n_states=1,
        library={"degree": 3, "interactions_degree": 1},
        var_names=["x"],
        implicit=True,
    )
    model.solve(X, dX, threshold=0.05)
    model.select(*held_out)
    _, deriv, _ = model.scores(*held_out)

    assert float(deriv[0, int(model.selected_[0])]) < 1e-3
    # normalised against the truth 0.3 dx/dt + x dx/dt - 0.18 + 0.9 x = 0
    text = model.equations()
    assert "-0.1800" in text and "+0.9000" in text
    assert "+0.3000" in text and "+1.0000" in text


def test_sindy_pi_excludes_each_candidate_from_its_own_regression():
    X, dX = _michaelis_menten(jax.random.key(2))
    model = sindy.SINDy(
        n_states=1, library={"degree": 2, "interactions_degree": 1}, implicit=True
    )
    model.solve(X, dX, threshold=0.05)
    chex.assert_trees_all_equal(
        jnp.diagonal(model.models_[0]), jnp.zeros(len(model.feature_names))
    )


def test_sindy_pi_uses_each_equation_own_derivative():
    """Only a per-equation theta can fit a rational eq0 beside a linear eq1."""
    key = jax.random.key(3)
    X = jax.random.uniform(key, (400, 2), minval=0.5, maxval=2.0)
    dx0 = (1.0 - X[:, 1]) / (1.0 + X[:, 0])
    dx1 = 0.5 * X[:, 0] - 0.7 * X[:, 1]
    dX = jnp.stack([dx0, dx1], axis=1)

    model = sindy.SINDy(
        n_states=2, library={"degree": 1, "interactions_degree": 1}, implicit=True
    )
    model.solve(X, dX, threshold=0.05)
    model.select(X, dX)
    _, deriv, _ = model.scores(X, dX)
    for i in range(2):
        assert float(deriv[i, int(model.selected_[i])]) < 1e-3


def test_sindy_pi_scores_mark_out_of_library_candidates_invalid():
    X, dX = _michaelis_menten(jax.random.key(4))
    model = sindy.SINDy(
        n_states=1,
        library={"degree": 3, "interactions_degree": 1, "bias": False},
        implicit=True,
    )
    model.solve(X, dX, threshold=0.05)
    fit, deriv, _ = model.scores(X, dX)
    bias_column = model.feature_names.index("1")
    assert not bool(model.library_mask[0, bias_column])
    assert jnp.isinf(fit[0, bias_column]) and jnp.isinf(deriv[0, bias_column])


def test_sindy_pi_solve_is_jittable():
    X, dX = _michaelis_menten(jax.random.key(5), n=80)
    model = sindy.SINDy(
        n_states=1, library={"degree": 2, "interactions_degree": 1}, implicit=True
    )
    thetas = model._build_thetas(X, dX)
    models = jax.jit(sindy._stlsq, static_argnames=("max_iters",))(
        thetas, thetas, model._candidate_masks(), 0.05, 20
    )
    chex.assert_tree_all_finite(models)


def test_sindy_pi_all_pruned_returns_finite_zeros():
    X, dX = _michaelis_menten(jax.random.key(6), n=80)
    model = sindy.SINDy(
        n_states=1, library={"degree": 2, "interactions_degree": 1}, implicit=True
    )
    models = model.solve(X, dX, threshold=1e6)
    chex.assert_tree_all_finite(models)
    assert bool(jnp.all(models == 0))
