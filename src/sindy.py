"""
SINDy-style sparse regression, with Lineax handling the linear solves.

The class is jax jitable but not reverse-differentiable.

Implicit formulation is implemented. The STLSQ loop prunes terms by
masking columns, keeping shapes static and enabling custom libraries.

PDEs or higher derivative equations are not supported.
"""

import itertools
from collections.abc import Sequence

import jax
import jax.numpy as jnp
import lineax as lx
from jaxtyping import Array, Bool, Float, Int

_SPEC_KEYS = {
    "degree",
    "var_degree",
    "exclude",
    "bias",
    "interactions_degree",
    "var_interactions_degree",
}


def _lstsq(A: Float[Array, "m n"], b: Float[Array, " m"]) -> Float[Array, " n"]:
    """Least-squares solve of A @ x ~ b via Lineax."""
    operator = lx.MatrixLinearOperator(A)
    solution = lx.linear_solve(
        operator, b, solver=lx.AutoLinearSolver(well_posed=False)
    )
    return solution.value


def _stlsq_one(
    theta: Float[Array, "samples features"],
    y: Float[Array, " samples"],
    allowed: Bool[Array, " features"],
    threshold: float,
    max_iters: int,
) -> Float[Array, " features"]:
    """
    Sequentially thresholded least-squares fit for 1 library term

    The loop exits when least-square results settle or when
    `max_iters` is reached.
    """
    coef = jnp.where(allowed, _lstsq(theta * allowed[None, :], y), 0.0)

    def cond(carry):
        _, mask, previous, i = carry
        return (i < max_iters) & jnp.any(mask != previous)

    def body(carry):
        coef, mask, _, i = carry
        new = mask & (jnp.abs(coef) >= threshold)
        coef = jnp.where(new, _lstsq(theta * new[None, :], y), 0.0)
        return coef, new, mask, i + 1

    coef, *_ = jax.lax.while_loop(cond, body, (coef, allowed, ~allowed, 0))
    return coef


@jax.jit(static_argnames=("max_iters",))
def _stlsq(
    thetas: Float[Array, "groups samples features"],
    Ys: Float[Array, "groups samples targets"],
    masks: Bool[Array, "groups targets features"],
    threshold: float = 0.1,
    max_iters: int = 20,
) -> Float[Array, "groups features targets"]:
    """
    STLSQ vmapped over targets (inner) and equation groups (outer).

    Explicit mode passes one group of `n_states` targets; SINDy-PI sweep passes
    `n_states` groups, each regressing every library column on the others.
    """

    def over_targets(theta, Y, mask, threshold, max_iters):
        return jax.vmap(_stlsq_one, in_axes=(None, 1, 0, None, None), out_axes=1)(
            theta, Y, mask, threshold, max_iters
        )

    return jax.vmap(over_targets, in_axes=(0, 0, 0, None, None))(
        thetas, Ys, masks, threshold, max_iters
    )


def _normalise_spec(spec: dict, n_vars: int) -> dict:
    """Fill a library spec's optional keys and pad `exclude` to `n_vars + 1`."""
    unknown = set(spec) - _SPEC_KEYS
    if unknown:
        raise ValueError(
            f"unknown library key(s) {sorted(unknown)}; "
            f"expected any of {sorted(_SPEC_KEYS)}"
        )
    exclude = spec.get("exclude") or ()
    padded = {tuple(e) + (0,) * (n_vars + 1 - len(e)) for e in exclude}
    return {
        "degree": spec["degree"],
        "var_degree": spec.get("var_degree"),
        "interactions_degree": spec.get("interactions_degree"),
        "var_interactions_degree": spec.get("var_interactions_degree"),
        "exclude": padded,
        "bias": spec.get("bias", True),
    }


class SINDy:
    """
    Sparse identification of nonlinear dynamics over a polynomial library.

    `n_states` is the number of dynamic state variables, `n_controls` the number
    of exogenous ones: controls build the library but get no equation of their
    own. `var_names` lists the states first, then the controls, and every
    positional tuple below is indexed the same way over
    `n_variables = n_states + n_controls`.

    `library` is one spec dict shared by every equation, or a list of `n_states`
    dicts giving each equation its own library over shared columns. A spec has
    nested levels of control, each falling back to the level above when omitted:

        {
            "degree": 3,                            # (i)   required
            "var_degree": (2, 1, 3),                # (ii)  optional
            "exclude": [(3, 0, 0), (0, 1, 2)],      # (iii) optional
            "bias": True,                           #       optional, default True
            "interactions_degree": 1,               # (iv)  optional, implicit only
            "var_interactions_degree": (1, 1, 0),   # (v) optional, implicit only
        }

    (i)   `degree` bounds a monomial's total degree.
    (ii)  `var_degree` bounds each variable's own power, still capped by
          `degree`. A tuple of length `n_variables`, positional: entry `i` is the
          highest power variable `i` may take. Omit to let every variable reach
          `degree`.
    (iii) `exclude` drops individual terms. Each entry is an exponent vector,
          positional: `(3, 0, 0)` is `x0**3` and `(0, 1, 2)` is `x1 * x2**2`. A
          trailing entry may be added for the derivative factor, so `(1, 0, 1)`
          is `x0 * dx_k`; short vectors are zero-padded. Omit to drop nothing.
    (iv)  `interactions_degree` bounds the total degree of the monomial
          multiplying the equation's own derivative, which is what makes the
          library implicit. The derivative factor itself is not counted by
          `degree`. `0` admits the bare `dx_k` column alone; omit for no
          derivative terms at all (the explicit library).
    (v)   `var_interactions_degree` bounds each variable's power inside an
          interaction monomial. It does not inherit from `var_degree` -- the two
          families have independent caps.

    `bias` gates only the constant column `1`. The bare `dx_k` column is
    governed by `interactions_degree` instead, so a library may carry the
    standalone derivative (`interactions_degree=0`).

    Set `implicit=True` to run the SINDy-PI sweep (needs `interactions_degree`).

    Both `var_degree` and `var_interactions_degree` must have length
    `n_variables`, not `n_states`: a short tuple is silently truncated by `zip`,
    leaving the trailing controls uncapped with no error. `exclude` vectors are
    padded, so those are safe at either length. Check `library_terms()` before
    solving if a fit looks off. DO NOT use `exclude` to drop the bias term; set
    `bias=False` instead.
    """

    def __init__(
        self,
        n_states: int,
        library: dict | Sequence[dict],
        n_controls: int = 0,
        var_names: list[str] | None = None,
        implicit: bool = False,
    ):
        assert (
            isinstance(library, dict) or len(library) == n_states
        ), "`library` must be one spec dict or a sequence of `n_states` dicts."

        self.n_states = n_states
        self.n_controls = n_controls
        self.n_variables = n_states + n_controls

        self.var_names = var_names or (
            [f"x{i}" for i in range(n_states)] + [f"u{j}" for j in range(n_controls)]
        )
        assert (
            len(self.var_names) == self.n_variables
        ), "`var_names` must have length n_states + n_controls"

        self.implicit = implicit
        self.coefficients_: Float[Array, "features targets"] | None = None
        self.models_: Float[Array, "equations features candidates"] | None = None
        self.selected_: Int[Array, " equations"] | None = None

        specs = [library] * n_states if isinstance(library, dict) else library
        self.library = [_normalise_spec(spec, self.n_variables) for spec in specs]

        max_degree = max(spec["degree"] for spec in self.library)
        caps = [spec["interactions_degree"] for spec in self.library]
        max_interactions = max((c for c in caps if c is not None), default=None)

        # `combos` is the union library: pure monomials first, then the ones
        # carrying the sentinel derivative index.
        combos = [()]
        for d in range(1, max_degree + 1):
            combos += itertools.combinations_with_replacement(
                range(self.n_variables), d
            )
        if max_interactions is not None:
            for d in range(max_interactions + 1):
                pure = itertools.combinations_with_replacement(
                    range(self.n_variables), d
                )
                combos += [c + (self.n_variables,) for c in pure]

        # A derivative column no equation admits would still cost a whole
        # candidate regression in implicit mode, so drop it rather than mask it.
        self._combos = [
            c
            for c in combos
            if self.n_variables not in c
            or any(self._allows(s, c) for s in self.library)
        ]

        # One label per union-library column, admitted or not: these line up with
        # the rows of `coefficients_` and `models_`, so an excluded term keeps its
        # column (holding zero) and its name. `library_terms()` is the other view.
        self.feature_names = [self._name(c, "dx_k") for c in self._combos]
        self._mono_names = [
            self._name(tuple(i for i in c if i != self.n_variables), "")
            for c in self._combos
        ]
        self._is_derivative = jnp.array(
            [self.n_variables in c for c in self._combos]
        )
        self.library_mask = jnp.array(
            [[self._allows(spec, c) for c in self._combos] for spec in self.library]
        )

    def _name(self, combo: tuple[int, ...], deriv: str) -> str:
        """Render one monomial, writing the sentinel index as `deriv`."""
        parts = [deriv if i == self.n_variables else self.var_names[i] for i in combo]
        return "*".join(parts) if parts else "1"

    def _allows(self, spec: dict, combo: tuple[int, ...]) -> bool:
        """Whether one equation's library admits `combo`, across all levels."""
        # Exponent vector: entry i is the power of variable i, the last entry the
        # power of the equation's own derivative.
        exponents = tuple(combo.count(i) for i in range(self.n_variables + 1))
        deriv, monomial = exponents[-1], exponents[:-1]
        total = sum(monomial)

        if deriv > 1:  # no d2x/dt2, and no mixed derivative products
            return False
        if exponents in spec["exclude"]:
            return False

        if deriv:
            cap = spec["interactions_degree"]
            if cap is None or total > cap:
                return False
            var_cap = spec["var_interactions_degree"]
        else:
            if total > spec["degree"]:
                return False
            if total == 0:
                return spec["bias"]
            var_cap = spec["var_degree"]

        if var_cap is None:
            return True
        return all(e <= cap for e, cap in zip(monomial, var_cap))

    def library_terms(self) -> str:
        """The terms each equation's library holds -- check this before solving."""
        lines = []
        for i in range(self.n_states):
            deriv = f"d{self.var_names[i]}"
            names = [self._name(c, deriv) for c in self._combos]
            allowed = [n for n, ok in zip(names, self.library_mask[i]) if ok]
            lines.append(f"{deriv}: {', '.join(allowed)}")
        return "\n".join(lines)

    def _build_theta(
        self,
        X: Float[Array, "samples vars"],
        dxdt: Float[Array, " samples"],
    ) -> Float[Array, "samples features"]:
        """Evaluate the library on `X`, using `dxdt` for the derivative column."""
        # The derivative column is always concatenated, so a sentinel index can
        # never fall off the end of `vals` and silently clamp to X's last column.
        vals = jnp.concatenate([X, dxdt[:, None]], axis=1)
        cols = [
            jnp.prod(vals[:, list(c)], axis=1) if c else jnp.ones(X.shape[0])
            for c in self._combos
        ]
        return jnp.stack(cols, axis=1)

    def _build_thetas(
        self,
        X: Float[Array, "samples vars"],
        dXdt: Float[Array, "samples targets"],
    ) -> Float[Array, "equations samples features"]:
        """One theta per equation, each carrying that equation's own derivative."""
        return jax.vmap(self._build_theta, in_axes=(None, 1))(X, dXdt)

    def _candidate_masks(self) -> Bool[Array, "equations candidates features"]:
        """Equation i's library minus candidate j: the SINDy-PI self-exclusion."""
        n_features = len(self._combos)
        eye = jnp.eye(n_features, dtype=bool)
        return self.library_mask[:, None, :] & ~eye[None]

    def solve(
        self,
        X: Float[Array, "samples vars"],
        dXdt: Float[Array, "samples targets"],
        threshold: float = 0.1,
        max_iters: int = 20,
    ) -> Float[Array, "..."]:
        """
        Fit sparse coefficients, running the SINDy-PI sweep when `implicit`.

        Explicit writes `coefficients_`, `(features, targets)`. Implicit writes
        `models_`, `(equations, features, candidates)`, where `models_[i, :, j]`
        is equation `i`'s model with library column `j` on the left-hand side.
        """
        assert X.shape[1] == self.n_variables
        assert dXdt.shape[1] == self.n_states

        if not self.implicit:
            theta = self._build_theta(X, jnp.zeros(X.shape[0]))
            xi = _stlsq(
                theta[None],
                dXdt[None],
                self.library_mask[None],
                threshold,
                max_iters,
            )
            self.coefficients_ = xi[0]
            return self.coefficients_

        thetas = self._build_thetas(X, dXdt)
        self.models_ = _stlsq(
            thetas, thetas, self._candidate_masks(), threshold, max_iters
        )
        return self.models_

    def scores(
        self,
        X: Float[Array, "samples vars"],
        dXdt: Float[Array, "samples targets"],
        p: int = 2,  # Norm order
        eps: float = 1e-8,
    ) -> tuple[
        Float[Array, "equations candidates"],
        Float[Array, "equations candidates"],
        Int[Array, "equations candidates"],
    ]:
        """
        Both SINDy-PI metrics per (equation, candidate), plus term counts.

        Pass held-out data -- the sweep always fits its own training set.
        Returnsthe implicit fit residual, the derivative-prediction error
        of the rearranged rational form, and the number of active terms.
        Candidates outside an equation's own library score `inf`.
        """
        # i = equations
        # s = samples
        # f = features (terms)
        # c = candidates (functions)
        thetas = self._build_thetas(X, dXdt)
        residual = thetas - jnp.einsum("isf,ifc->isc", thetas, self.models_)
        state_fit = jnp.linalg.norm(residual, ord=p, axis=1) / jnp.linalg.norm(
            thetas, ord=p, axis=1
        )

        # Setting the derivative slot to 1s leaves each column's monomial factor,
        # which is the same for every equation.
        mono = self._build_theta(X, jnp.ones(X.shape[0]))
        n_features = len(self._combos)
        coeffs = jnp.eye(n_features)[None] - self.models_
        is_deriv = self._is_derivative[None, :, None]
        denom = jnp.einsum("sf,ifc->isc", mono, coeffs * is_deriv)
        numer = jnp.einsum("sf,ifc->isc", mono, coeffs * ~is_deriv)
        predicted = -numer / jnp.where(jnp.abs(denom) < eps, jnp.nan, denom)

        target = dXdt.T[:, :, None]
        deriv_fit = jnp.linalg.norm(
            predicted - target, ord=p, axis=1
        ) / jnp.linalg.norm(target, ord=p, axis=1)
        deriv_fit = jnp.where(jnp.isfinite(deriv_fit), deriv_fit, jnp.inf)

        n_terms = jnp.sum(jnp.abs(self.models_) > 0, axis=1)
        invalid = ~self.library_mask

        return (
            jnp.where(invalid, jnp.inf, state_fit),
            jnp.where(invalid, jnp.inf, deriv_fit),
            n_terms,
        )

    def select(
        self,
        X: Float[Array, "samples vars"],
        dXdt: Float[Array, "samples targets"],
    ) -> Int[Array, " equations"]:
        """
        Pick one left-hand-side candidate per equation into `selected_`.

        Lowest derivative error wins, with the sparsest model breaking ties
        within `tol` of it -- the top candidates are usually the same relation
        rescaled, so read the rational form rather than the winning term.
        """
        _, deriv_fit, _ = self.scores(X, dXdt)
        self.selected_ = jnp.min(deriv_fit, axis=1, keepdims=True)
        return self.selected_

    def _relation(self, equation: int, candidate: int) -> Float[Array, " features"]:
        """Coefficients of `theta_j - Theta xi_j = 0` for one candidate."""
        n_features = len(self._combos)
        column = jnp.eye(n_features)[:, candidate]
        return column - self.models_[equation][:, candidate]

    def _rational(self, equation: int, candidate: int, tol: float = 1e-8) -> str:
        """
        Render one candidate relation as `dx/dt = -(N) / (D)`.

        Normalised so the denominator's dominant coefficient is +1, which is what
        makes the form comparable across candidates: the best few are usually the
        same relation at different scales. A relation carrying no derivative term
        has no denominator and is left unnormalised.
        """
        coeffs = self._relation(equation, candidate)
        denominator = jnp.where(self._is_derivative, coeffs, 0.0)
        scale = denominator[jnp.argmax(jnp.abs(denominator))]
        if abs(float(scale)) > tol:
            coeffs = coeffs / scale

        def side(carries_derivative: bool) -> str:
            terms = []
            for k, name in enumerate(self._mono_names):
                c = float(coeffs[k])
                if bool(self._is_derivative[k]) != carries_derivative:
                    continue
                if abs(c) < tol:
                    continue
                terms.append(f"{c:+.4f}*{name}" if name != "1" else f"{c:+.4f}")
            return " ".join(terms) or "0"

        lhs = f"d{self.var_names[equation]}/dt"
        return f"{lhs} = -({side(False)}) / ({side(True)})"

    def candidates(
        self,
        X: Float[Array, "samples vars"],
        dXdt: Float[Array, "samples targets"],
        top: int | None = None,
    ) -> list[dict]:
        """
        Ranked candidate relations per equation, best first, for vetting a library.

        One dict per equation holding parallel lists `term` (the left-hand-side
        column), `error` (held-out derivative error) and `explicit_form` (the
        rearranged rational form). `top` keeps only the best few; None keeps all.
        Pass the result to `equations` to render it.
        """
        _, deriv_fit, _ = self.scores(X, dXdt)
        ranked = []
        for i in range(self.n_states):
            deriv = f"d{self.var_names[i]}"
            names = [self._name(c, deriv) for c in self._combos]
            order = [int(j) for j in jnp.argsort(deriv_fit[i])[:top]]
            ranked.append(
                {
                    "term": [names[j] for j in order],
                    "error": [float(deriv_fit[i, j]) for j in order],
                    "explicit_form": [self._rational(i, j) for j in order],
                }
            )
        return ranked

    def equations(
        self,
        candidates: list[dict] | None = None,
        target_names: list[str] | None = None,
        tol: float = 1e-8,
    ) -> str:
        """
        Render the fitted result as human-readable equations.

        Given a `candidates` payload, lists those instead of the selected models.
        """
        if candidates is not None:
            return "\n".join(
                f"[{error:.2e}] LHS={term:<14}  {form}"
                for equation in candidates
                for term, error, form in zip(
                    equation["term"], equation["error"], equation["explicit_form"]
                )
            )

        if self.implicit:
            return "\n".join(
                self._rational(i, int(self.selected_[i]), tol)
                for i in range(self.n_states)
            )

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
                terms.append(f"{c:+.4f}*{name}" if name != "1" else f"{c:+.4f}")
            lines.append(f"d{target_names[j]}/dt = {' '.join(terms) or '0'}")
        return "\n".join(lines)
