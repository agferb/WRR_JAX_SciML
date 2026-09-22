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


@jax.jit(static_argnames=("max_iters", "normalise"))
def _stlsq(
    thetas: Float[Array, "groups samples features"],
    Ys: Float[Array, "groups samples targets"],
    masks: Bool[Array, "groups targets features"],
    threshold: float = 0.1,
    max_iters: int = 20,
    normalise: bool = True,
) -> Float[Array, "groups features targets"]:
    """
    STLSQ vmapped over targets (inner) and equation groups (outer).

    Explicit mode passes one group of `n_states` targets; SINDy-PI sweep passes
    `n_states` groups, each regressing every library column on the others.
    With `normalise`, columns and targets are scaled to unit 2-norm before the
    fit and un-scaled after, so `threshold` reads as a dimensionless fraction.
    """

    def over_targets(theta, Y, mask, threshold, max_iters):
        return jax.vmap(_stlsq_one, in_axes=(None, 1, 0, None, None), out_axes=1)(
            theta, Y, mask, threshold, max_iters
        )

    if normalise:
        col = jnp.linalg.norm(thetas, axis=1)  # (groups, features)
        tgt = jnp.linalg.norm(Ys, axis=1)  # (groups, targets)
        col = jnp.where(col > 0, col, 1.0)
        tgt = jnp.where(tgt > 0, tgt, 1.0)
        thetas = thetas / col[:, None, :]
        Ys = Ys / tgt[:, None, :]

    xi = jax.vmap(over_targets, in_axes=(0, 0, 0, None, None))(
        thetas, Ys, masks, threshold, max_iters
    )

    if normalise:
        xi = xi * tgt[:, None, :] / col[:, :, None]

    return xi


# --- conditioning diagnostics ------------------------------------------------


def conditioning(
    theta: Float[Array, "samples features"],
    dxdt: Float[Array, " samples"] | None = None,
) -> dict:
    """
    Column scale, spectrum, rank and precision diagnostics for one matrix.

    `theta` should already be sliced to one equation's admitted columns.
    `dxdt`, if given, adds a `settled` entry: the fraction of samples with
    `|dx/dt|` below 1% of its peak, and the first index after which it stays
    below that for good.
    """
    n_samples, n_admitted = theta.shape
    eps = float(jnp.finfo(theta.dtype).eps)
    col_scale = jnp.linalg.norm(theta, axis=0)
    normalised = theta / jnp.where(col_scale > 0, col_scale, 1.0)

    sigma_raw = jnp.linalg.svd(theta, compute_uv=False)
    sigma = jnp.linalg.svd(normalised, compute_uv=False)
    tol = sigma[0] * max(n_samples, n_admitted) * eps
    rank = int(jnp.sum(sigma > tol))

    kappa_raw = float(sigma_raw[0] / sigma_raw[-1])
    kappa_normalised = float(sigma[0] / sigma[-1])
    kappa_eps = kappa_normalised * eps

    report = {
        "n_samples": int(n_samples),
        "n_admitted": int(n_admitted),
        "col_scale_min": float(jnp.min(col_scale)),
        "col_scale_max": float(jnp.max(col_scale)),
        "col_scale_span": float(jnp.max(col_scale) / jnp.min(col_scale)),
        "sigma": [float(s) for s in sigma / sigma[0]],
        "kappa_raw": kappa_raw,
        "kappa_normalised": kappa_normalised,
        "rank": rank,
        "eps": eps,
        "kappa_eps": kappa_eps,
        "digits_lost": float(jnp.log10(kappa_normalised)),
    }

    if dxdt is not None:
        active = jnp.abs(dxdt) > 0.01 * jnp.max(jnp.abs(dxdt))
        idx = jnp.where(active, jnp.arange(dxdt.shape[0]), -1)
        last_active = int(jnp.max(idx))
        report["settled"] = {
            "fraction": float(jnp.mean(~active)),
            "from": last_active + 1 if last_active >= 0 else 0,
        }

    return report


def format_conditioning(report: dict) -> str:
    """Render one `conditioning()` report as a text table with spectrum and verdicts."""
    sigma, rank, n = report["sigma"], report["rank"], len(report["sigma"])

    def bar(s: float) -> str:
        digits = -jnp.log10(s) if s > 0 else 30.0
        return "#" * min(int(digits), 40)

    around = range(max(0, rank - 2), min(n, rank + 2))
    shown = sorted(set(range(min(3, n))) | set(around) | {n - 1})

    lines = [
        f"n_samples = {report['n_samples']}   "
        f"n_admitted = {report['n_admitted']}   rank = {rank}",
        f"col_scale:   min = {report['col_scale_min']:.3e}   "
        f"max = {report['col_scale_max']:.3e}   "
        f"span = {report['col_scale_span']:.3e}",
        "",
        f"kappa_raw = {report['kappa_raw']:.3e}   "
        f"kappa_normalised = {report['kappa_normalised']:.3e}",
        f"eps = {report['eps']:.3e}   kappa x eps = {report['kappa_eps']:.3e}   "
        f"digits_lost = {report['digits_lost']:.1f}",
        "",
        f"numerical rank cut-off:  {sigma[int(rank)-1]:.3e} -> {sigma[int(rank)]:.3e}",
    ]

    if "settled" in report:
        s = report["settled"]
        lines.append(
            f"settled samples (below 1% of peak derivative) = {s['fraction']:.1%}"
        )
        lines.append(f"settling index = {s['from']}")

    return "\n".join(lines)


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
    entry: tuple[int | bool | float, ...], spec: dict, n_vars: int
) -> list[tuple[int, ...]]:
    """
    Unfold `True` slots into every power that variable could take.
    Unfold `inf` slots into every power that variable could take and 0.
    """

    ranges = []
    for slot, power in enumerate(entry):
        if power is True:
            ranges.append(range(1, _slot_cap(spec, slot, n_vars) + 1))
        elif power == jnp.inf:
            ranges.append(range(_slot_cap(spec, slot, n_vars) + 1))
        else:
            ranges.append((power,))
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
