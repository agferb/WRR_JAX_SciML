"""SINDy-style sparse regression, with Lineax handling the linear solves.

The STLSQ loop prunes terms by masking columns rather than slicing them, which
keeps shapes static and lets the whole solve run under `jax.jit`.
"""

import itertools
from collections.abc import Sequence

import jax
import jax.numpy as jnp
import lineax as lx
from jaxtyping import Array, Bool, Float


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
    library_mask: Bool[Array, "targets features"],
    threshold: float = 0.1,
    n_iters: int = 20,
) -> Float[Array, "features targets"]:
    """Sequentially thresholded least squares, vmapped over targets.

    Inactive terms are zeroed by masking `theta`'s columns, not by slicing them:
    `well_posed=False` is required so the resulting rank-deficient solve returns
    zero for the masked columns. `library_mask` seeds each target's active set,
    so terms outside an equation's library never enter it -- the mask only
    shrinks, so they stay zero.
    """

    def one_target(y, allowed):
        coef = jnp.where(allowed, _lstsq(theta * allowed[None, :], y), 0.0)
        mask = allowed

        def body(_, carry):
            coef, mask = carry
            mask = mask & (jnp.abs(coef) >= threshold)
            coef = jnp.where(mask, _lstsq(theta * mask[None, :], y), 0.0)
            return coef, mask

        coef, _ = jax.lax.fori_loop(0, n_iters, body, (coef, mask))
        return coef

    return jax.vmap(one_target, in_axes=(1, 0), out_axes=1)(dxdt, library_mask)


def _normalise_spec(spec: dict, n_states: int) -> dict:
    """Fill in a library spec's optional keys and index `exclude` for lookup."""
    exclude = spec.get("exclude") or ()
    return {
        "degree": spec["degree"],
        "var_degree": spec.get("var_degree"),
        "exclude": {tuple(e) for e in exclude},
        "bias": spec.get("bias", True),
    }


class SINDy:
    """Sparse identification of nonlinear dynamics over a polynomial library.

    The library is fixed at construction from `n_states` alone, independently of
    any dataset.

    `library` is one spec dict shared by every equation, or a list of `n_states`
    dicts giving each equation its own library over shared columns. A spec has
    three nested levels of control, each falling back to the level above when
    omitted:

        {
            "degree": 3,                        # (i)   required
            "var_degree": (2, 1, 3),            # (ii)  optional
            "exclude": [(3, 0, 0), (0, 1, 2)],  # (iii) optional
            "bias": True,                       #       optional, default True
        }

    (i)   `degree` bounds a monomial's total degree.
    (ii)  `var_degree` bounds each variable's own power, still capped by
          `degree`. A tuple of length `n_states`, positional: entry `i` is the
          highest power `x_i` may take. Omit to let every variable reach
          `degree`.
    (iii) `exclude` drops individual terms. Each entry is an exponent vector of
          length `n_states`, positional: `(3, 0, 0)` is `x0**3` and `(0, 1, 2)`
          is `x1 * x2**2`. Omit to drop nothing.

    Both `var_degree` and every `exclude` entry must have length `n_states`, and
    `library`, when a list, must have length `n_states` -- check `feature_names`
    and `library_mask` if a fit looks off. DO NOT use `exclude` to exclude the
    bias term, alternatively set bias=False.
    """

    def __init__(
        self,
        n_states: int,
        library: dict | Sequence[dict],
        var_names: list[str] | None = None,
    ):
        self.n_states = n_states
        self.var_names = var_names or [f"x{i}" for i in range(n_states)]
        self.coefficients_: Float[Array, "features targets"] | None = None

        specs = library if isinstance(library, Sequence) else [library] * n_states
        self.library = [_normalise_spec(spec, n_states) for spec in specs]
        max_degree = max(spec["degree"] for spec in self.library)

        # `_combos` is the union library: one tuple of variable indices per monomial.
        self._combos = [()]
        for d in range(1, max_degree + 1):
            self._combos += itertools.combinations_with_replacement(range(n_states), d)

        # Names also represent the union library
        self.feature_names = [
            "*".join(self.var_names[i] for i in c) if c else "1" for c in self._combos
        ]
        self.library_mask = jnp.array(
            [
                [self._allows(spec, c) for c in self._combos]
                for spec in self.library
            ]
        )

    def _allows(self, spec: dict, combo: tuple[int, ...]) -> bool:
        """Whether one equation's library admits `combo`, across all three levels."""
        # Exponent vector: entry i is the power of variable i in this monomial.
        exponents = tuple(combo.count(i) for i in range(self.n_states))
        total_degree = sum(exponents)

        if total_degree > spec["degree"]:
            return False
        if exponents in spec["exclude"]:
            return False
        if total_degree == 0:
            return spec["bias"]

        var_degree = spec["var_degree"]
        if var_degree is None:
            return True
        return all(e <= cap for e, cap in zip(exponents, var_degree))

    def _build_theta(
        self, X: Float[Array, "samples vars"]
    ) -> Float[Array, "samples features"]:
        """Evaluate the library on `X`."""
        cols = [
            (
                jnp.prod(jnp.stack([X[:, i] for i in c]), axis=0)
                if c
                else jnp.ones(X.shape[0])
            )
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

        assert dXdt.shape[1] == self.n_states, "dXdt values must have n_states columns."

        self.coefficients_ = _stlsq(
            self._build_theta(X), dXdt, self.library_mask, threshold, n_iters
        )
        return self.coefficients_

    def equations(
        self, target_names: list[str] | None = None, tol: float = 1e-8
    ) -> str:
        """Render the fitted coefficients as human-readable equations."""
        xi = self.coefficients_
        n_targets = xi.shape[1]
        if target_names is None:
            target_names = self.var_names

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
