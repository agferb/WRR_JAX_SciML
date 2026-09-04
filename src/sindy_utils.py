"""
Module-level helpers behind `sindy.SINDy`: the solver and the spec normaliser.

Two independent groups. `_lstsq`/`_stlsq_one`/`_stlsq` are the jittable
sequentially-thresholded least-squares path, which prunes by masking columns so
shapes stay static. `_slot_cap`/`_expand_exclusion`/`_normalise_spec` turn a
user-written library spec into the canonical dict the class reads, and run once
at construction.
"""

import itertools

import jax
import jax.numpy as jnp
import lineax as lx
from jaxtyping import Array, Bool, Float

_SPEC_KEYS = {
    "degree",
    "var_degree",
    "exclude",
    "bias",
    "interactions_degree",
    "var_interactions_degree",
}


# --- the STLSQ solver -------------------------------------------------------


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


# --- library spec normalisation ---------------------------------------------


def _slot_cap(spec: dict, slot: int, n_vars: int) -> int:

    """Highest power `slot` can reach in any monomial this spec admits."""

    if slot == n_vars:  # the derivative slot: present or absent, never squared
        return 1
    caps = []
    for total, per_var in (
        (spec["degree"], spec["var_degree"]),
        (spec["interactions_degree"], spec["var_interactions_degree"]),
    ):
        if total is None:
            continue
        # a short `per_var` leaves trailing variables uncapped -- see the class
        # docstring; fall back to the family's total degree rather than raising
        if per_var is None or slot >= len(per_var):
            caps.append(total)
        else:
            caps.append(min(total, per_var[slot]))
    return max(caps, default=0)


def _expand_exclusion(
    entry: tuple[int | bool, ...], spec: dict, n_vars: int
) -> list[tuple[int, ...]]:

    """Unfold `True` slots into every power that variable could take."""

    # `is True`, not `== True`: Python has `True == 1`, so an equality test would
    # read a literal power of 1 as the wildcard.
    ranges = [
        range(1, _slot_cap(spec, slot, n_vars) + 1) if power is True else (power,)
        for slot, power in enumerate(entry)
    ]
    return list(itertools.product(*ranges))


def _normalise_spec(spec: dict, n_vars: int) -> dict:

    """Fill a library spec's optional keys and pad `exclude` to `n_vars + 1`."""

    unknown = set(spec) - _SPEC_KEYS
    if unknown:
        raise ValueError(
            f"unknown library key(s) {sorted(unknown)}; "
            f"expected any of {sorted(_SPEC_KEYS)}"
        )
    normalised = {
        "degree": spec["degree"],
        "var_degree": spec.get("var_degree"),
        "interactions_degree": spec.get("interactions_degree"),
        "var_interactions_degree": spec.get("var_interactions_degree"),
        "bias": spec.get("bias", True),
    }
    exclude = spec.get("exclude") or ()
    padded = (tuple(e) + (0,) * (n_vars + 1 - len(e)) for e in exclude)
    # unfold before the set is built: `(2, True, 0)` and `(2, 1, 0)` hash alike
    normalised["exclude"] = {
        unfolded
        for entry in padded
        for unfolded in _expand_exclusion(entry, normalised, n_vars)
    }
    return normalised
