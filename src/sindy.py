"""SINDy-style sparse regression, with Lineax handling the linear solves.

The STLSQ loop prunes terms by masking columns rather than slicing them, which
keeps shapes static and lets the whole solve run under `jax.jit`.
"""

import itertools
import jax
import jax.numpy as jnp
import lineax as lx
from jaxtyping import Array, Float


def _lstsq(A: Float[Array, "m n"], b: Float[Array, " m"]) -> Float[Array, " n"]:
    """Least-squares solve of A @ x ~ b via Lineax."""
    operator = lx.MatrixLinearOperator(A)
    solution = lx.linear_solve(
        operator, b, solver=lx.AutoLinearSolver(well_posed=False)
    )
    return solution.value


@jax.jit(static_argnames=("n_iters",))
def _stlsq(
    theta: Float[Array, "samples features"],
    dxdt: Float[Array, "samples targets"],
    threshold: float = 0.1,
    n_iters: int = 20,
) -> Float[Array, "features targets"]:
    """Sequentially thresholded least squares, vmapped over targets.

    Inactive terms are zeroed by masking `theta`'s columns, not by slicing them:
    `well_posed=False` is required so the resulting rank-deficient solve returns
    zero for the masked columns.
    """

    def one_target(y):
        coef = _lstsq(theta, y)
        mask = jnp.ones(theta.shape[1], dtype=bool)

        def body(_, carry):
            coef, mask = carry
            mask = mask & (jnp.abs(coef) >= threshold)
            coef = jnp.where(mask, _lstsq(theta * mask[None, :], y), 0.0)
            return coef, mask

        coef, _ = jax.lax.fori_loop(0, n_iters, body, (coef, mask))
        return coef

    return jax.vmap(one_target, in_axes=1, out_axes=1)(dxdt)


class SINDy:
    """Sparse identification of nonlinear dynamics over a polynomial library.

    The library is fixed at construction from `n_states` alone, independently of
    any dataset, so the same instance can be reused across datasets of matching
    width.
    """

    def __init__(
        self,
        n_states: int,
        degree: int = 2,
        include_bias: bool = True,
        var_names: list[str] | None = None,
    ):
        self.n_states = n_states
        self.degree = degree
        self.include_bias = include_bias
        self.var_names = var_names or [f"x{i}" for i in range(n_states)]
        self.coefficients_: Float[Array, "features targets"] | None = None

        # `_combos` is the library: one tuple of variable indices per monomial.
        # Names are derived from the same pass, so they cannot drift from the
        # columns `_build_theta` produces.
        self._combos = [()] if include_bias else []
        for d in range(1, degree + 1):
            self._combos += itertools.combinations_with_replacement(
                range(n_states), d
            )
        self.feature_names = [
            "*".join(self.var_names[i] for i in c) if c else "1"
            for c in self._combos
        ]

    def _build_theta(
        self, X: Float[Array, "samples vars"]
    ) -> Float[Array, "samples features"]:
        """Evaluate the library on `X`."""
        cols = [
            jnp.prod(jnp.stack([X[:, i] for i in c]), axis=0)
            if c else jnp.ones(X.shape[0])
            for c in self._combos
        ]
        return jnp.stack(cols, axis=1)

    def solve(
        self,
        X: Float[Array, "samples vars"],
        dXdt: Float[Array, "samples targets"],
        threshold: float = 0.1,
        n_iters: int = 20,
    ) -> Float[Array, "features targets"]:
        """Fit sparse coefficients mapping the library of `X` onto `dxdt`."""
        self.coefficients_ = _stlsq(self._build_theta(X), dXdt, threshold, n_iters)
        return self.coefficients_

    def equations(
        self, target_names: list[str] | None = None, tol: float = 1e-8
    ) -> str:
        """Render the fitted coefficients as human-readable equations."""
        xi = self.coefficients_
        n_targets = xi.shape[1]
        if target_names is None:
            target_names = [f"x{i}" for i in range(n_targets)]

        lines = []
        for j in range(n_targets):
            terms = []
            for i, name in enumerate(self.feature_names):
                c = float(xi[i, j])
                if abs(c) < tol:
                    continue
                terms.append(f"{c:+.4f} {name}" if name != "1" else f"{c:+.4f}")
            lines.append(f"d{target_names[j]}/dt = {' '.join(terms) or '0'}")
        return "\n".join(lines)
