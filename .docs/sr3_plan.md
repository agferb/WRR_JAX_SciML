# SR3 optimizer for SINDy and SINDy-PI (unconstrained)

Variant without constraints. The constrained variant, which builds on the
same solver and adds within-target equality constraints for both explicit and
implicit mode, is in `sr3_constrained_plan.md`.

## Context

`SINDy.solve` has one optimizer: STLSQ (`_stlsq` / `_stlsq_one` in
`src/sindy_utils.py`), a pure jitted function double-vmapped over equation
groups (outer) and targets (inner), with masks for per-equation libraries and
the SINDy-PI self-exclusion, and an optional column/target normalisation.

SR3 (Zheng et al. 2018; Champion et al. 2020, arXiv:1906.10612) relaxes the
hard-thresholding problem:

```
min_{w,u}  ½‖y − Θw‖² + λR(u) + (1/2ν)‖w − u‖²
```

`w` is the unregularised coefficient vector, `u` the sparse one.

### Why not import PySINDy

- PySINDy's `SR3` is a sklearn estimator with a Python loop, `history_` lists
  and warnings: not jittable, not vmappable, and data must leave the device.
- No per-target masks (the `var_degree` / `exclude` structure, the SINDy-PI
  self-exclusion).
- Different semantics: `normalize_columns` does not normalise targets,
  `reg_weight_lam` instead of a threshold, `(targets, features)` orientation.
- PySINDy's `SINDyPI` optimizer is not SR3 at all: a Python loop of CVXPY
  LASSO solves, one per candidate.
- The algorithm itself is ~30 lines; only the math is worth taking.

## Decisions

- **L0 only.** No L1/L2, no trimming, no inequality constraints.
- **Threshold parametrisation, as in kpchamp/SINDySR3.** `λ = threshold²/(2ν)`,
  so the L0 prox thresholds at exactly `√(2λν) = threshold`. `λ` never appears
  in the code; the `u`-update is `u = w · (|w| >= threshold)`, the same
  comparison STLSQ uses. With `normalise=True` the threshold keeps its
  dimensionless meaning.
- **Solver is a pure function**, like `_stlsq`. jit and vmap need pure
  functions; a stateful class (PySINDy style) cannot sit under the
  `(groups, targets)` double vmap.
- **User-facing selection via a `method` keyword** on `SINDy.solve`, not
  optimizer config classes. With two optimizers this is the smallest change.
  If a third arrives, switch to frozen dataclasses (`STLSQ(max_iters)`,
  `SR3(nu, tol, max_iters)`) passed as `optimizer=`.
- **Final refit on the support of `u`** with `_lstsq`. SR3's `u` carries the
  relaxation bias; PySINDy leaves this off by default for SR3, here it is
  always on.

## Implementation

### `src/sindy_optimizers.py` (new)

`src/sindy_utils.py` currently mixes the solver, conditioning diagnostics and
spec normalisation. Split the solver out:

- Move `_lstsq`, `_stlsq_one`, `_stlsq` here unchanged in behaviour.
- `_fit_groups(one_fn, thetas, Ys, masks, normalise, *args)`: the existing
  normalise → double vmap → un-scale logic taken out of `_stlsq`, generic over
  the per-target solver. `_stlsq` becomes a thin jitted entry point over it.
- `_sr3_one` and the jitted entry point `_sr3`.

`_sr3_one` sketch:

```python
def _sr3_one(theta, y, allowed, threshold, nu, tol, max_iters):
    theta = theta * allowed[None, :]
    n = theta.shape[1]

    # w-update = lstsq on [Θ; I/√ν] w ≈ [y; u/√ν]. QR once, never ΘᵀΘ,
    # so the condition number is not squared.
    Q, R = jnp.linalg.qr(jnp.vstack([theta, jnp.eye(n) / jnp.sqrt(nu)]))
    Qy, Qu = Q[:-n].T @ y, Q[-n:].T / jnp.sqrt(nu)
    w_update = lambda u: solve_triangular(R, Qy + Qu @ u)

    # lax.while_loop:
    #   cond: (i < max_iters) & (err >= tol)
    #   body: w = w_update(u)
    #         u_new = where(allowed & (|w| >= threshold), w, 0)
    #         err = ‖u_new − u‖ / ν
    # init: u = where(allowed, _lstsq(theta, y), 0), err = inf, i = 0
    # final: coef = where(support(u), _lstsq(theta * support(u), y), 0)
    return coef, i
```

Notes:

- `H = ΘᵀΘ + I/ν` never changes, so the factorisation happens once per target;
  each iteration is one triangular solve plus an element-wise prox. SR3 needs
  far more iterations than STLSQ (hundreds to thousands vs ~10).
- Masks need no special handling: a zeroed column has `H_jj = 1/ν`, nothing
  drives it, and `u_j` stays 0.
- `threshold`, `nu`, `tol` are traced (a threshold sweep does not recompile);
  `max_iters` is static, as in `_stlsq`.
- Returning the iteration count replaces PySINDy's `ConvergenceWarning`.
- Compatible with `qr_reduction_plan.md` if that is implemented later: `Θ`
  can be replaced by its `R` factor before stacking.

### `src/sindy_utils.py`

Keeps conditioning diagnostics and spec normalisation; drops the moved solver.

### `src/sindy.py`

```python
model.solve(Y, dXdt, threshold=0.1, max_iters=None, normalise=True,
            method="stlsq", nu=1.0, tol=1e-5)
```

- `max_iters=None` resolves per method: 20 for STLSQ, ~1000 for SR3.
- `nu` and `tol` are ignored by STLSQ.
- New attribute `n_iters_`, `(groups, targets)`, so fits that hit `max_iters`
  can be spotted.
- Implicit mode needs nothing extra: the self-exclusion masks already exist and
  the output keeps the `models_` shape, so `scores`, `select`, `candidates` and
  `equations` work unchanged.

### `tests/test_sindy.py`

- Lotka–Volterra (explicit): SR3 recovers the same support as STLSQ and
  coefficients close to the true ones.
- Michaelis–Menten (implicit): SR3 sweep selects the same rational form.
- A threshold sweep under jit does not recompile.
- `n_iters_` below `max_iters` on the clean examples.

### Docs (outside `.docs/`)

- `README.md`: layout row for `src/sindy_optimizers.py`, `sindy_utils.py` row
  trimmed, short SR3 section (method, threshold meaning, `nu`, refit).
- `CLAUDE.md`: `sr3` branch notes (solver moved, `method` keyword, QR instead
  of normal equations, `max_iters` default per method, refit always on).

## Implementation order

1. **Refactor only.** Create `sindy_optimizers.py`, move STLSQ, add
   `_fit_groups`. Existing tests must pass unchanged. Commit point.
2. **`_sr3_one` + `_sr3`** with their tests.
3. **`SINDy.solve(method=...)`, `n_iters_`.**
4. **Docs.** Commit point.

## Risks, not guarded in code

- **STLSQ numbers may shift slightly.** Moving `_stlsq` behind `_fit_groups`
  can change XLA's lowering again, as the extra vmap did (~1e-7 in float32).
  Check the existing tests and the four-decimal Lotka–Volterra example output
  after step 1.
- **`ν` choice.** With normalised columns, `ν = 1` is a reasonable default.
  Small `ν` ties `w` to `u`, so each iteration moves little and convergence
  slows; large `ν` makes `w` nearly the plain least-squares fit regardless of
  `u`, so the result approaches a single threshold of that fit.
- **Convergence.** SR3 can run to `max_iters` on ill-conditioned libraries;
  `n_iters_` exposes it, nothing warns.

## Out of scope

L1/L2 regularisers, trimming, inequality constraints, equality constraints
(see `sr3_constrained_plan.md`).
