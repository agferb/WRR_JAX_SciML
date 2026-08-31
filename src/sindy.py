"""SINDy-style sparse regression, with Lineax handling the linear solves.

The thresholding loop runs in Python (shapes change as terms are pruned, so it
is not jittable as written) but each inner least-squares solve goes through
Lineax, which is the Equinox-native linear solver.
"""

import itertools

import jax.numpy as jnp
import lineax as lx
from jaxtyping import Array, Float


def polynomial_library(
    X: Float[Array, "samples vars"],
    degree: int = 2,
    include_bias: bool = True,
    var_names: list[str] | None = None,
) -> tuple[Float[Array, "samples features"], list[str]]:
    """Build the candidate feature matrix Theta(X) from monomials up to `degree`."""
    n_samples, n_vars = X.shape
    if var_names is None:
        var_names = [f"x{i}" for i in range(n_vars)]

    cols, names = [], []
    if include_bias:
        cols.append(jnp.ones((n_samples,)))
        names.append("1")
    for d in range(1, degree + 1):
        for combo in itertools.combinations_with_replacement(range(n_vars), d):
            cols.append(jnp.prod(jnp.stack([X[:, i] for i in combo]), axis=0))
            names.append("*".join(var_names[i] for i in combo))
    return jnp.stack(cols, axis=1), names


def _lstsq(A: Float[Array, "m n"], b: Float[Array, " m"]) -> Float[Array, " n"]:
    """Least-squares solve of A @ x ~ b via Lineax."""
    operator = lx.MatrixLinearOperator(A)
    solution = lx.linear_solve(
        operator, b, solver=lx.AutoLinearSolver(well_posed=False)
    )
    return solution.value


def stlsq(
    theta: Float[Array, "samples features"],
    dxdt: Float[Array, "samples targets"],
    threshold: float = 0.1,
    n_iters: int = 10,
) -> Float[Array, "features targets"]:
    """Sequentially thresholded least squares -- the core SINDy solve.

    Fits each target column independently, repeatedly zeroing coefficients
    below `threshold` and refitting on the surviving terms.
    """
    n_features = theta.shape[1]
    n_targets = dxdt.shape[1]
    xi = jnp.zeros((n_features, n_targets))

    for j in range(n_targets):
        coef = _lstsq(theta, dxdt[:, j])
        active = jnp.ones(n_features, dtype=bool)
        for _ in range(n_iters):
            new_active = active & (jnp.abs(coef) >= threshold)
            if bool(jnp.array_equal(new_active, active)):
                break  # converged: nothing else was pruned
            active = new_active
            if not bool(active.any()):
                coef = jnp.zeros(n_features)
                break
            refit = _lstsq(theta[:, active], dxdt[:, j])
            coef = jnp.zeros(n_features).at[jnp.where(active)[0]].set(refit)
        xi = xi.at[:, j].set(coef)
    return xi


def format_equations(
    xi: Float[Array, "features targets"],
    feature_names: list[str],
    target_names: list[str] | None = None,
    tol: float = 1e-8,
) -> str:
    """Render a coefficient matrix as human-readable differential equations."""
    n_targets = xi.shape[1]
    if target_names is None:
        target_names = [f"x{i}" for i in range(n_targets)]

    lines = []
    for j in range(n_targets):
        terms = []
        for i, name in enumerate(feature_names):
            c = float(xi[i, j])
            if abs(c) < tol:
                continue
            terms.append(f"{c:+.4f} {name}" if name != "1" else f"{c:+.4f}")
        rhs = " ".join(terms) if terms else "0"
        lines.append(f"d{target_names[j]}/dt = {rhs}")
    return "\n".join(lines)
