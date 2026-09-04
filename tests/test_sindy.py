"""Tests for `src.sindy`.

Kept deliberately small: each test either pins a design decision a future edit
could plausibly reverse, or exercises an end-to-end recovery. Library-level
gating is table-driven in `test_library_levels_gate_the_right_terms` rather than
spread over one test per level.
"""

import chex
import jax
import jax.numpy as jnp
import pytest

from src import sindy, sindy_utils

JX, VMAX, KM = 0.6, 1.5, 0.3


def known_system(n_samples: int, seed: int = 0):
    # dx/dt = -2 x + 3 x z  /  dz/dt = 1.5 z + u
    key = jax.random.key(seed)
    X = jax.random.uniform(key, (n_samples, 3), minval=0.5, maxval=2.0)
    dX = jnp.stack(
        [-2.0 * X[:, 0] + 3.0 * X[:, 0] * X[:, 1], 1.5 * X[:, 1] + X[:, 2]], axis=1
    )
    return X, dX


def michaelis_menten(n_samples: int, seed: int = 0):
    # dx/dt = jx - Vmax x / (Km + x), a rational law no explicit library can hold
    x = jax.random.uniform(jax.random.key(seed), (n_samples, 1), minval=0.2, maxval=3.0)
    return x, JX - VMAX * x / (KM + x)


def rational_with_control(n_samples: int, seed: int = 0):
    # dx/dt = (u - x) / (1 + x)  ->  dx/dt + x*dx/dt - u + x = 0
    X = jax.random.uniform(jax.random.key(seed), (n_samples, 2), minval=0.5, maxval=2.0)
    return X, ((X[:, 1] - X[:, 0]) / (1.0 + X[:, 0]))[:, None]


def correlated_system(n_samples: int, seed: int = 0):
    """Near-collinear features, so STLSQ needs a second thresholding pass."""
    X = jax.random.uniform(jax.random.key(seed), (n_samples, 2), minval=0.5, maxval=2.0)
    X = jnp.stack([X[:, 0], 0.9 * X[:, 0] + 0.1 * X[:, 1]], axis=1)
    dX = jnp.stack([-2.0 * X[:, 0] + 0.6 * X[:, 1] ** 2, 1.5 * X[:, 1]], axis=1)
    return X, dX


def allowed_terms(model: sindy.SINDy, equation: int = 0) -> list[str]:
    return [
        name
        for name, ok in zip(model.feature_names, model.library_mask[equation])
        if bool(ok)
    ]


# --- library construction ---------------------------------------------------

LIBRARY_CASES = [
    (
        "var_degree caps one variable under the total degree",
        dict(
            n_states=2,
            n_controls=1,
            var_names=["x", "z", "u"],
            library={"degree": 3, "var_degree": (2, 1, 1)},
        ),
        {"x*x*z", "x*x", "x*u"},
        {"z*z", "x*x*x", "z*u*u"},
    ),
    (
        "exclude drops individual exponent vectors",
        dict(
            n_states=2,
            n_controls=1,
            var_names=["x", "z", "u"],
            library={"degree": 2, "exclude": [(1, 1, 0), (0, 2, 0), (0, 0, 1)]},
        ),
        {"x*x"},
        {"x*z", "z*z", "u"},
    ),
    (
        "short exclude vectors are padded, not ignored",
        dict(
            n_states=2,
            library={"degree": 2, "interactions_degree": 1, "exclude": [(1, 1)]},
        ),
        {"x0*x0"},
        {"x0*x1"},
    ),
    (
        "bias gates the constant column but never the bare derivative",
        dict(
            n_states=2,
            library={"degree": 2, "interactions_degree": 1, "bias": False},
        ),
        {"x0", "dx_k"},
        {"1"},
    ),
    (
        "interactions_degree=0 admits the bare derivative alone",
        dict(n_states=2, library={"degree": 2, "interactions_degree": 0}),
        {"dx_k"},
        {"x0*dx_k"},
    ),
    (
        "a True slot in exclude drops every power of that variable",
        dict(n_states=2, library={"degree": 3, "exclude": [(2, True, 0)]}),
        {"x0*x0", "x0*x1*x1"},
        {"x0*x0*x1"},
    ),
    (
        "var_degree caps how far a True slot unfolds",
        dict(
            n_states=2,
            library={"degree": 3, "var_degree": (3, 1), "exclude": [(1, True, 0)]},
        ),
        {"x0", "x0*x0", "x0*x0*x0"},
        {"x0*x1"},
    ),
    (
        "var_interactions_degree caps the interaction family only",
        dict(
            n_states=2,
            library={
                "degree": 2,
                "interactions_degree": 2,
                "var_interactions_degree": (2, 0),
            },
        ),
        {"x1*x1", "x0*x0*dx_k"},
        {"x1*dx_k"},
    ),
]


@pytest.mark.parametrize(
    "kwargs,present,absent",
    [case[1:] for case in LIBRARY_CASES],
    ids=[case[0] for case in LIBRARY_CASES],
)
def test_library_levels_gate_the_right_terms(kwargs, present, absent):
    allowed = set(allowed_terms(sindy.SINDy(**kwargs)))
    assert present <= allowed, f"missing {present - allowed}"
    assert not (absent & allowed), f"should have been dropped: {absent & allowed}"


def test_library_matches_the_documented_example():
    """The canonical spec, asserted in order -- `feature_names` is an axis label."""
    model = sindy.SINDy(
        n_states=2,
        library={"degree": 2, "interactions_degree": 1, "bias": False},
        implicit=True,
    )
    assert allowed_terms(model) == [
        "x0",
        "x1",
        "x0*x0",
        "x0*x1",
        "x1*x1",
        "dx_k",
        "x0*dx_k",
        "x1*dx_k",
    ]


def test_libraries_may_differ_per_equation():
    """A list spec gives each equation its own mask over shared columns."""
    X, dX = known_system(200)
    shared = sindy.SINDy(n_states=2, library={"degree": 2}, n_controls=1)
    listed = sindy.SINDy(
        n_states=2, library=[{"degree": 2}, {"degree": 2}], n_controls=1
    )
    assert shared.feature_names == listed.feature_names
    chex.assert_trees_all_close(shared.solve(X, dX), listed.solve(X, dX))

    split = sindy.SINDy(
        n_states=2,
        library=[{"degree": 2}, {"degree": 1, "bias": False}],
        var_names=["x", "z"],
    )
    names = split.feature_names
    assert not bool(split.library_mask[1, names.index("1")])
    assert not bool(split.library_mask[1, names.index("x*z")])
    assert bool(split.library_mask[0, names.index("x*z")])
    # library_terms() reports the mask, not the union library
    assert "x*z" not in split.library_terms().splitlines()[1]

    xi = split.solve(X[:, :2], dX, threshold=0.1)
    assert bool(jnp.all(jnp.abs(xi.T[~split.library_mask]) == 0.0))


def test_controls_build_the_library_without_getting_an_equation():
    model = sindy.SINDy(
        n_states=1,
        library={"degree": 2, "interactions_degree": 1},
        n_controls=1,
        var_names=["x", "u"],
        implicit=True,
    )
    # one equation per state, but library columns over every variable
    assert model.library_mask.shape[0] == 1
    assert {"u", "x*u", "u*dx_k"} <= set(allowed_terms(model))
    # the derivative sentinel is `n_variables`, and appears at most once
    assert [bool(f) for f in model._is_derivative] == [
        model.n_variables in c for c in model._combos
    ]
    assert max(c.count(model.n_variables) for c in model._combos) == 1
    # default names must not collide across the two families
    assert sindy.SINDy(2, {"degree": 1}, n_controls=1).var_names == ["x0", "x1", "u0"]


def test_unknown_library_keys_and_bad_shapes_are_rejected():
    with pytest.raises(ValueError, match="unknown library key"):
        sindy.SINDy(n_states=2, library={"degree": 2, "interaction_degree": 1})
    with pytest.raises(AssertionError):
        sindy.SINDy(n_states=3, library=[{"degree": 2}, {"degree": 1}])


# --- the STLSQ solver -------------------------------------------------------


def test_stlsq_converges_and_respects_the_threshold():
    """`max_iters` is a cap the loop settles well inside.

    The `max_iters=1` leg matters: comparing only two uncapped runs would agree
    trivially even if the loop were broken to stop after one pass.
    """
    X, dX = correlated_system(300)
    model = sindy.SINDy(n_states=2, library={"degree": 3})
    settled = model.solve(X, dX, threshold=0.1, max_iters=20)
    chex.assert_trees_all_equal(
        settled, model.solve(X, dX, threshold=0.1, max_iters=200)
    )
    # this system needs a second pass, so a loop that stopped early would differ
    assert not jnp.allclose(settled, model.solve(X, dX, threshold=0.1, max_iters=1))

    # thresholding prunes on coefficient magnitude: the true `u` coefficient is 1.0
    Xc, dXc = known_system(200)
    controlled = sindy.SINDy(
        n_states=2, library={"degree": 3}, n_controls=1, var_names=["x", "z", "u"]
    )
    names = controlled.feature_names
    assert float(controlled.solve(Xc, dXc, threshold=1.5)[names.index("u"), 1]) == 0.0
    assert controlled.solve(Xc, dXc, threshold=0.5)[
        names.index("u"), 1
    ] == pytest.approx(1.0, abs=1e-3)


def test_solve_is_jittable_and_survives_total_pruning():
    """Both modes trace, and a fully masked solve returns zeros rather than NaN."""
    X, dX = known_system(50)
    explicit = sindy.SINDy(n_states=2, library={"degree": 2}, n_controls=1)
    theta = explicit._build_theta(X, jnp.zeros(X.shape[0]))
    xi = jax.jit(sindy_utils._stlsq, static_argnames=("max_iters",))(
        theta[None], dX[None], explicit.library_mask[None], 0.1, 20
    )
    chex.assert_tree_all_finite(xi)
    pruned = explicit.solve(X, dX, threshold=1e6)
    chex.assert_tree_all_finite(pruned)
    assert bool(jnp.all(pruned == 0))

    Xm, dXm = michaelis_menten(80, seed=5)
    implicit = sindy.SINDy(
        n_states=1, library={"degree": 2, "interactions_degree": 1}, implicit=True
    )
    thetas = implicit._build_thetas(Xm, dXm)
    models = jax.jit(sindy_utils._stlsq, static_argnames=("max_iters",))(
        thetas, thetas, implicit._candidate_masks(), 0.05, 20
    )
    chex.assert_tree_all_finite(models)
    pruned = implicit.solve(Xm, dXm, threshold=1e6)
    chex.assert_tree_all_finite(pruned)
    assert bool(jnp.all(pruned == 0))


# --- recovery ---------------------------------------------------------------


def test_explicit_recovers_a_known_system_with_control():
    X, dX = known_system(300)
    model = sindy.SINDy(
        n_states=2, library={"degree": 2}, n_controls=1, var_names=["x", "z", "u"]
    )
    xi = model.solve(X, dX, threshold=0.1)
    names = model.feature_names

    assert xi[names.index("x"), 0] == pytest.approx(-2.0, abs=1e-3)
    assert xi[names.index("x*z"), 0] == pytest.approx(3.0, abs=1e-3)
    assert xi[names.index("u"), 0] == pytest.approx(0.0, abs=1e-6)
    assert xi[names.index("z"), 1] == pytest.approx(1.5, abs=1e-3)
    assert xi[names.index("u"), 1] == pytest.approx(1.0, abs=1e-3)
    assert xi[names.index("1"), 1] == pytest.approx(0.0, abs=1e-6)


def test_sindy_pi_recovers_michaelis_menten():
    X, dX = michaelis_menten(300)
    held_out = michaelis_menten(150, seed=1)
    model = sindy.SINDy(
        n_states=1,
        library={"degree": 3, "interactions_degree": 1},
        var_names=["x"],
        implicit=True,
    )
    model.solve(X, dX, threshold=0.05)
    model.select(*held_out)
    _, deriv_fit, _ = model.scores(*held_out)

    assert float(deriv_fit[0, int(model.selected_[0])]) < 1e-3
    # truth: 0.3 dx/dt + x dx/dt - 0.18 + 0.9 x = 0, denominator normalised to +1
    text = model.equations()
    assert "-0.1800" in text and "+0.9000" in text
    assert "+0.3000" in text and "+1.0000" in text


def test_sindy_pi_recovers_a_rational_system_with_control():
    """The regression guard for the derivative sentinel under control."""
    X, dX = rational_with_control(400)
    model = sindy.SINDy(
        n_states=1,
        library={"degree": 2, "interactions_degree": 1},
        n_controls=1,
        var_names=["x", "u"],
        implicit=True,
    )
    model.solve(X, dX, threshold=0.05)
    model.select(X, dX)
    _, deriv_fit, _ = model.scores(X, dX)
    assert float(deriv_fit[0, int(model.selected_[0])]) < 1e-3
    text = model.equations()
    assert "+1.0000*x" in text and "-1.0000*u" in text


def test_sindy_pi_uses_each_equation_own_derivative():
    """Only a per-equation theta fits a rational eq0 beside a linear eq1."""
    X = jax.random.uniform(jax.random.key(3), (400, 2), minval=0.5, maxval=2.0)
    dX = jnp.stack(
        [(1.0 - X[:, 1]) / (1.0 + X[:, 0]), 0.5 * X[:, 0] - 0.7 * X[:, 1]], axis=1
    )
    model = sindy.SINDy(
        n_states=2, library={"degree": 1, "interactions_degree": 1}, implicit=True
    )
    model.solve(X, dX, threshold=0.05)
    model.select(X, dX)
    _, deriv_fit, _ = model.scores(X, dX)
    assert all(float(deriv_fit[i, int(model.selected_[i])]) < 1e-3 for i in range(2))


# --- the SINDy-PI sweep -----------------------------------------------------


def test_sindy_pi_excludes_each_candidate_from_its_own_regression():
    """Without the `~jnp.eye` self-exclusion every candidate trivially fits itself."""
    X, dX = michaelis_menten(300, seed=2)
    model = sindy.SINDy(
        n_states=1, library={"degree": 2, "interactions_degree": 1}, implicit=True
    )
    model.solve(X, dX, threshold=0.05)
    chex.assert_trees_all_equal(
        jnp.diagonal(model.models_[0]), jnp.zeros(len(model.feature_names))
    )


def test_sindy_pi_scoring_and_selection():
    X, dX = michaelis_menten(300, seed=4)
    model = sindy.SINDy(
        n_states=1,
        library={"degree": 3, "interactions_degree": 1, "bias": False},
        var_names=["x"],
        implicit=True,
    )
    model.solve(X, dX, threshold=0.05)
    fit, deriv_fit, n_terms = model.scores(X, dX)

    # candidates outside the library are unusable, not merely bad
    bias_column = model.feature_names.index("1")
    assert not bool(model.library_mask[0, bias_column])
    assert jnp.isinf(fit[0, bias_column]) and jnp.isinf(deriv_fit[0, bias_column])

    model.select(X, dX)
    chosen = int(model.selected_[0])
    assert float(deriv_fit[0, chosen]) == pytest.approx(float(jnp.min(deriv_fit[0])))
    assert int(n_terms[0, chosen]) > 0


def test_candidates_rank_records_and_equations_render_them():
    X, dX = michaelis_menten(300, seed=7)
    model = sindy.SINDy(
        n_states=1,
        library={"degree": 3, "interactions_degree": 1},
        var_names=["x"],
        implicit=True,
    )
    model.solve(X, dX, threshold=0.05)

    every = model.candidates(X, dX)
    assert len(every) == model.n_states
    assert set(every[0]) == {"term", "error", "explicit_form"}
    assert len(every[0]["term"]) == len(model.feature_names)  # top=None keeps all
    assert every[0]["error"] == sorted(every[0]["error"])

    payload = model.candidates(X, dX, top=2)
    assert payload[0]["term"] == every[0]["term"][:2]
    lines = model.equations(payload).splitlines()
    assert len(lines) == 2
    assert payload[0]["term"][0] in lines[0]
    assert payload[0]["explicit_form"][0] in lines[0]

    # the rational form is normalised so the denominator's dominant term is +1,
    # which is what makes rescaled duplicates comparable
    assert "+1.0000" in payload[0]["explicit_form"][0].split(" / ")[1]

    # without a payload it still renders the selected models
    model.select(X, dX)
    assert model.equations().startswith("dx/dt = ")


def test_exclude_wildcard_unfolds_over_every_power():
    """`True` means "present at any power", so power 0 is left alone."""
    spec = sindy_utils._normalise_spec({"degree": 3, "exclude": [(2, True, 0)]}, 2)
    assert sorted(spec["exclude"]) == [(2, 1, 0), (2, 2, 0), (2, 3, 0)]

    # a tighter per-variable cap shortens the unfolding
    capped = sindy_utils._normalise_spec(
        {"degree": 3, "var_degree": (3, 2), "exclude": [(2, True, 0)]}, 2
    )
    assert sorted(capped["exclude"]) == [(2, 1, 0), (2, 2, 0)]

    # the derivative slot only ever holds 0 or 1
    deriv = sindy_utils._normalise_spec(
        {"degree": 2, "interactions_degree": 1, "exclude": [(1, 0, True)]}, 2
    )
    assert sorted(deriv["exclude"]) == [(1, 0, 1)]

    # a wildcard reaches both families, and spares the variable-absent term
    model = sindy.SINDy(
        n_states=2, library={"degree": 2, "interactions_degree": 2,
                             "exclude": [(True, 0, 0)]}
    )
    allowed = set(allowed_terms(model))
    assert not ({"x0", "x0*x0"} & allowed)  # every pure power of x0 is gone
    assert "x0*dx_k" in allowed  # but the interaction family is untouched

    # `True == 1`, so a literal power of 1 must not be read as the wildcard
    literal = sindy_utils._normalise_spec({"degree": 3, "exclude": [(2, 1, 0)]}, 2)
    assert sorted(literal["exclude"]) == [(2, 1, 0)]
