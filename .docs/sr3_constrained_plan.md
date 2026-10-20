# SR3 optimizer with equality constraints (explicit and implicit mode)

Variant with constraints. It contains everything in `sr3_plan.md` (the
unconstrained variant; background on why PySINDy is not imported is there)
plus within-target linear equality constraints in both modes. Steps 1–3 of the
implementation order below are the unconstrained plan, so it can be stopped
there.

## Context

`SINDy.solve` has one optimizer: STLSQ (`_stlsq` / `_stlsq_one` in
`src/sindy_utils.py`), a pure jitted function double-vmapped over equation
groups (outer) and targets (inner), with masks for per-equation libraries and
the SINDy-PI self-exclusion, and an optional column/target normalisation.

SR3 (Zheng et al. 2018; Champion et al. 2020, arXiv:1906.10612) relaxes the
hard-thresholding problem, here with equality constraints on `w`:

```
min_{w,u}  ½‖y − Θw‖² + λ‖u‖₀ + (1/2ν)‖w − u‖²     s.t.  C w = d
```

PySINDy's `ConstrainedSR3` (and the original kpchamp/SINDySR3 code) use the
same closed-form KKT update for equality constraints, but compute `inv(H)` and
a dense `kron(inv(H), I_targets)` every iteration, assume one `Θ` shared by all
targets, and couple targets through a flattened coefficient vector. None of
that carries over: here constraints are within-target, `H` differs per target
(per-equation masks), and everything is factorised once.

## Decisions

- **L0 only**, threshold parametrisation as in kpchamp/SINDySR3:
  `λ = threshold²/(2ν)`, so the L0 prox thresholds at exactly `threshold`
  (`u = w · (|w| >= threshold)`).
- **Equality constraints only, within-target only.** Explicit mode: per
  equation. Implicit mode: per (equation, candidate) pair. No cross-target
  constraints, so the target vmap survives. No inequality constraints.
- **Solver is a pure function** under the existing double vmap; the user picks
  it with a `method` keyword on `SINDy.solve` (switch to optimizer config
  dataclasses if a third optimizer arrives).
- **Constrained terms stay in the support** of the final refit, so the
  returned coefficients are both sparse and feasible.
- **Recommended, awaiting confirmation: dead-row rule.** A constraint row
  forces its columns into the support only if its right-hand side is nonzero
  or at least one of its columns survived thresholding. Otherwise the row is
  already satisfied by zeros and is switched off for the final refit (see
  "Implicit mode" below for why this matters).

## Implicit mode: the scale problem

Candidate `j` of equation `i` fits `θ_j = Θξ`. With the relation vector that
`SINDy._relation` returns, `r = e_j − ξ`, the model is `Σ r_k θ_k = 0`. A
physical relation is only defined up to scale, and each candidate fixes that
scale differently, by forcing `r_j = 1`.

Michaelis–Menten (`dx` = dx/dt): the relation is
`Km·dx + x·dx + Vmax·x = 0`. A known value "Km = 0.3":

| Candidate (LHS column) | Scale | What "Km = 0.3" means |
|---|---|---|
| `x·dx` | `r_{x·dx} = 1` | `r_dx = 0.3` ✓ |
| `dx` | `r_dx = 1` | `r_dx = 0.3` contradicts the fixed scale; the true value here is `r_{x·dx} = 1/0.3` |

So a constraint with a nonzero right-hand side only means something relative
to one particular candidate. Constraints with a zero right-hand side
(`G r = 0`, e.g. `Vmax = Km` as `r_x − r_dx = 0`) are scale-invariant and can
be applied to every candidate unchanged.

**Every implicit constraint is stored as `G r = 0`.** A nonzero right-hand
side needs a reference term: "Km = 0.3 when the `x·dx` coefficient is 1"
becomes `r_dx − 0.3·r_{x·dx} = 0`, i.e. `G ← G − h·e_refᵀ`. It then holds in
every candidate's scale. When a row is restricted to a single candidate `j`
and gives no reference, the reference defaults to `j` ("in this model's own
scale"). Rows apply to all of the equation's candidates unless restricted.

**Translation to candidate `j`** (in `SINDy`, raw units, before the solver):

```
G r = 0   ⇒   G ξ = G e_j   →   C = G (column j masked),  d = G[:, j]
```

### Consequences

1. **Degenerate rows are unavoidable.** Every candidate masks its own column
   `j`, so a row applied to all candidates collapses for the candidates it
   mentions:
   - A row mentioning only `j` becomes `0·ξ = G_jj ≠ 0`: the constraint says
     `r_j = 0`, this candidate has `r_j = 1`. The candidate is incompatible and
     must be ruled out (score `inf`), not produce NaN.
   - Rows can become linearly dependent once column `j` is removed, making `S`
     (below) singular for some candidates only.

   Handled without branching under vmap: solve with `S` through lineax
   `AutoLinearSolver(well_posed=False)` instead of Cholesky, and after the
   final refit compute `‖Cξ − d‖`; above a tolerance the pair is marked
   infeasible.
2. **Dead rows.** A zero-right-hand-side row is satisfied when all its terms
   are zero. Forcing its terms into the support would add two noise-fitted
   terms to every candidate whose relation lacks them, hurting sparsity, which
   SINDy-PI ranking relies on. Hence the dead-row rule above. Any row
   mentioning `j` has `d ≠ 0` and always stays on.
3. **Rows touching terms outside an equation's library.** Masking turns them
   into a different constraint; with (1) and (2) they are either switched off
   (all-zero row, `d = 0`) or rule the candidate out. Documented, not guarded.
4. **Printed coefficients.** `_rational` rescales each relation so the
   denominator's largest coefficient is +1, so `r_dx = 0.3·r_{x·dx}` prints as
   `Km = 0.3` only when `x·dx` is the denominator's largest term. Docstring
   note.
5. **Cost.** Unchanged in complexity: per (equation, candidate), one QR
   factorisation, one K×K solve, then the SR3 iterations.

## Explicit mode

Direct form: a row `{"terms": {...}, "rhs": h}` for equation `i` is
`G ξ_i = h`, so `C = G` (masked by the equation's library), `d = h`. `ref` and
`candidates` are not allowed. The same degenerate-row handling, dead-row rule
and feasibility check apply (they cost nothing and cover risk 1 below).

## Implementation

### `src/sindy_optimizers.py` (new)

Split the solver out of `src/sindy_utils.py`, which keeps conditioning
diagnostics and spec normalisation.

- Move `_lstsq`, `_stlsq_one`, `_stlsq` here unchanged in behaviour.
- `_fit_groups(one_fn, thetas, Ys, masks, normalise, *args)`: the existing
  normalise → double vmap → un-scale logic taken out of `_stlsq`, generic over
  the per-target solver. It also scales constraints: with `w̃ = w·col/tgt`,
  `Cw = d` becomes `(C / col) w̃ = d / tgt`. In implicit mode `tgt = col_j`, no
  special case. Skipping this applies constraints in the wrong units silently.
- `_sr3_one`, `_constrained_lstsq`, jitted entry point `_sr3`.

Constraint arrays, padded to a fixed `K` rows:
`C: (groups, targets, K, features)`, `d` and `active: (groups, targets, K)`.
Explicit: `groups = 1`, `targets = n_states`. Implicit: `groups = n_states`,
`targets = features` (candidates). Without constraints `K = 0` and the code
reduces to plain SR3 with no extra branch.

`_sr3_one` sketch:

```python
def _sr3_one(theta, y, allowed, C, d, active, threshold, nu, tol, max_iters):
    theta = theta * allowed[None, :]
    C = C * allowed[None, :]
    n = theta.shape[1]

    # w-update = lstsq on [Θ; I/√ν] w ≈ [y; u/√ν]. QR once, never ΘᵀΘ.
    Q, R = jnp.linalg.qr(jnp.vstack([theta, jnp.eye(n) / jnp.sqrt(nu)]))
    Qy, Qu = Q[:-n].T @ y, Q[-n:].T / jnp.sqrt(nu)
    H_inv = lambda b: solve_triangular(R, solve_triangular(R, b, trans="T"))

    # Equality constraints (KKT), factorised once. Padded rows -> identity rows,
    # zero right-hand side, zero multiplier.
    M = H_inv(C.T)                                   # (features, K)
    S = C @ M + jnp.diag(~active)                    # solved with lineax, well_posed=False

    def w_update(u):
        w0 = solve_triangular(R, Qy + Qu @ u)
        return w0 - M @ solve_S(jnp.where(active, C @ w0 - d, 0.0))

    # lax.while_loop:
    #   cond: (i < max_iters) & (err >= tol)
    #   body: w = w_update(u)
    #         u_new = where(allowed & (|w| >= threshold), w, 0)
    #         err = ‖u_new − u‖ / ν
    # init: u = where(allowed, _lstsq(theta, y), 0), err = inf, i = 0
    #
    # final:
    #   alive = active & ((d != 0) | any(C != 0 on support(u)))   # dead-row rule
    #   keep  = support(u) | any(alive rows' C != 0)
    #   coef  = _constrained_lstsq(theta * keep, y, C * keep, d, alive)
    #   residual = ‖where(alive, C @ coef − d, 0)‖
    return coef, i, residual
```

`_constrained_lstsq` uses the null-space method so the condition number is not
squared (a KKT system on `ΘₛᵀΘₛ` would square it):

- `ξp = pinv(Cₛ) d` over alive rows, `Z` = null-space basis of `Cₛ` from its
  SVD (fixed shape: all `V` columns, weighted by `s < tol`).
- `η = _lstsq(Θₛ Z, y − Θₛ ξp)`, `ξ = where(keep, ξp + Zη, 0)`.
- Zeroing outside `keep` is safe: `Cₛ` and `Θₛ` are zero there, so neither the
  fit nor `Cξ` changes (the null space of `Cₛ` contains those coordinate
  directions, mixed arbitrarily with others).
- Infeasible constraints give a least-squares `ξp`; the residual check flags
  them.

Notes:

- `H` never changes: one factorisation per target, then one triangular solve,
  one K×K solve and an element-wise prox per iteration.
- Masked columns: `H_jj = 1/ν`, nothing drives them, `u_j` stays 0; `C`'s
  masked columns are zeroed.
- `threshold`, `nu`, `tol` traced; `max_iters` static.
- Compatible with `qr_reduction_plan.md` if implemented later: `Θ` can be
  replaced by its `R` factor before stacking.

### `src/sindy.py`

```python
model.solve(Y, dXdt, threshold=0.1, max_iters=None, normalise=True,
            method="stlsq", nu=1.0, tol=1e-5, constraints=None)

constraints = {
    0: [  # equation x0
        {"terms": {"x0": 1.0, "dx0": -1.0}},                           # Vmax = Km, all candidates
        {"terms": {"dx0": 1.0}, "rhs": 0.3, "ref": "x0*dx0"},          # Km = 0.3 in x0*dx0's scale
        {"terms": {"x0": 1.0}, "rhs": 2.0, "candidates": ["x0*dx0"]},  # per pair, own scale
    ],
}
```

- Term keys are relation-coefficient names, the same names `candidates()`
  prints in `term` (explicit mode: `feature_names`).
- `_constraint_arrays(constraints) -> (C, d, active)`: name lookup, reference
  folding (`G ← G − h·e_refᵀ`), candidate broadcast/restriction, the
  `G e_j` translation in implicit mode, padding to `K`. Lives on the class
  because it needs the names.
- `max_iters=None` resolves per method: 20 for STLSQ, ~1000 for SR3. `nu`,
  `tol`, `constraints` are SR3-only (one `assert` on `constraints` with
  `method="stlsq"`, matching the existing asserts in `solve`).
- New attributes: `n_iters_` `(groups, targets)`; `feasible_`
  (`(equations,)` explicit, `(equations, candidates)` implicit), all `True`
  without constraints.
- `scores` returns `inf` where `~feasible_`, as it already does for
  out-of-library candidates; `select`, `candidates`, `equations` unchanged.

### `tests/test_sindy.py`

Unconstrained:

- Lotka–Volterra (explicit): same support as STLSQ, coefficients close to true.
- Michaelis–Menten (implicit): same selected rational form as STLSQ.
- Threshold sweep under jit does not recompile; `n_iters_ < max_iters` on clean
  examples.

Constrained:

- Explicit: `C ξ ≈ d` with `normalise` on and off; constrained terms present
  in the support; all-padded constraints equal the unconstrained result.
- Implicit (Michaelis–Menten):
  - Every feasible candidate satisfies `G r ≈ 0`, `normalise` on and off.
  - A row mentioning only candidate `j` rules that candidate out (`inf`).
  - A reference row with `ref = j` restricted to `j` matches the per-pair row
    with no reference.
  - A zero-right-hand-side row adds no terms when its columns are absent.
  - The selected rational form is unchanged when the constraint matches the
    true model.

### Docs (outside `.docs/`)

- `README.md`: layout row for `src/sindy_optimizers.py`, `sindy_utils.py` row
  trimmed, SR3 section (method, threshold meaning, `nu`, refit) and a
  constraints section (spec format, implicit scale convention with the
  reference term, `feasible_`).
- `CLAUDE.md`: `sr3` branch notes (solver moved, `method` keyword, QR instead
  of normal equations, constraint scaling under `normalise`, implicit scale
  convention, dead-row and feasibility rules).

## Implementation order

1. **Refactor only.** Create `sindy_optimizers.py`, move STLSQ, add
   `_fit_groups`. Existing tests must pass unchanged. Commit point.
2. **Unconstrained `_sr3_one` + `_sr3`** with their tests.
3. **`SINDy.solve(method=...)`, `n_iters_`.** Commit point (equals
   `sr3_plan.md`).
4. **Constraints in the solver:** `C, d, active` in `_sr3_one`,
   `_constrained_lstsq`, dead-row rule, residual, constraint scaling in
   `_fit_groups`.
5. **Constraints in `SINDy`:** `_constraint_arrays`, `feasible_`, `scores`
   update, tests. Commit point.
6. **Docs.**

## Risks, not guarded in code

1. **Constraint on a masked term** (outside the equation's library). Becomes a
   different or infeasible row; surfaces as `feasible_ = False` or a switched-off
   row.
2. **Redundant or contradictory constraints** (rank-deficient `C`). The
   `well_posed=False` solves avoid NaN; contradictions surface as
   `feasible_ = False`.
3. **Feasibility tolerance.** A single absolute tolerance on `‖Cξ − d‖`
   applies in normalised units; float32 may need a looser one than float64.
4. **STLSQ numbers may shift slightly** after the step-1 refactor (XLA
   lowering, ~1e-7 in float32). Check existing tests and the four-decimal
   Lotka–Volterra example output.
5. **`ν` choice.** `ν = 1` with normalised columns. Small `ν` slows
   convergence; large `ν` approaches a single threshold of the least-squares
   fit. Nothing warns on hitting `max_iters`; `n_iters_` exposes it.

## Out of scope

L1/L2 regularisers, trimming, inequality constraints, cross-target constraints
(they would couple targets and lose the target vmap).
