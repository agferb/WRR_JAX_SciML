# QR-reduced STLSQ: cheap float64 for the SINDy-PI sweep

## Context

Every STLSQ solve in the repo runs on the full sample matrix. `_stlsq`
(`src/sindy_utils.py:70`) double-vmaps `_stlsq_one`, which calls `_lstsq` (Lineax
`AutoLinearSolver(well_posed=False)`, i.e. SVD) on a `(samples, features)` matrix
at every iteration of a `lax.while_loop`, for every target of every equation. On
Monod-Herbert that is an SVD of `(1001, 80)`, `3 x 80` times per iteration.

float64 is what makes the Monod-Herbert sweep succeed, and
`examples/monod_herbert.py` now enables it unconditionally at import. The goal
here is to make float64 cheap enough that precision stops being a
performance decision, on CPU or GPU, without changing what gets fitted.

## The idea

For each equation group, factor the matrix once as `A = QR`, with Q orthonormal
of shape `(samples, k)` and R of shape `(k, columns)`, where
`k = min(samples, columns)`. Because Q is orthonormal and every target lies in
the column span of A:

```
|| A_S c - a_j ||  =  || R_S c - r_j ||      for any column subset S and target j
```

So every regression STLSQ runs (any mask, any iteration, any candidate) can be
solved on the `k x columns` matrix R instead of the `samples x columns` matrix A.

- **Same solutions in exact arithmetic.** The minimum-norm least-squares solution
  depends only on `A^T A` and `A^T a_j`, and R preserves both
  (`R^T R = A^T A`, `R^T r_j = A^T a_j`).
- **Doesn't square kappa.** R has exactly the singular values of A, so
  `kappa(R) = kappa(A)`. Forming the Gram matrix `A^T A` would square it (1e16 ->
  1e32 here), which is why that shortcut is not an option.
- **Stable.** Householder QR is backward stable. It costs one `O(samples * columns^2)`
  factorisation per group, done once, outside the loop.
- **Parallelism unchanged.** The reduced problem keeps static shapes, so the
  double vmap, `jit`, and any future `shard_map` over candidates work as they do
  now, just on smaller matrices.

## Measured on Monod-Herbert

Prototype: normalised thetas passed to the current `_stlsq` either raw or as
`jnp.linalg.qr(thetas, mode="r")`, `threshold=0.01`, `max_iters=20`, scored on
the held-out `x0 = [25, 40, 0]` trajectory. 14-core CPU, second (compiled) call
timed.

| dtype | input | shape | solve time | selected LHS | held-out error |
|---|---|---|---|---|---|
| float64 | full | `(3, 1001, 80)` | 91.5 s | `s`, `x`, `s*u` | 1.3e-14, 2.0e-15, 6.0e-15 |
| float64 | R | `(3, 80, 80)` | **21.3 s** | `s*dx_k`, `s*x`, `s*u` | 2.1e-14, 7.8e-15, 3.1e-14 |
| float32 | full | `(3, 1001, 80)` | 155.7 s | `1`, `x`, `s*o*o` | 0.48, 0.36, 0.19 |
| float32 | R | `(3, 80, 80)` | **23.1 s** | `o*u`, `o`, `s*s*o` | 0.35, 0.012, 0.11 |

- **Speedup.** 4.3x in float64 and 6.8x in float32. That is below the 12.5x ratio
  of the matrix sizes, because SVD is not the only cost per iteration (Lineax,
  loop and vmap overhead). float32 is slower than float64 on the full matrix,
  most likely because its noise-driven pruning keeps changing and the loop runs
  to `max_iters`.
- **float64 equivalence holds where it matters.** Implicit-fit residuals agree to
  6.8e-13, and both runs find exact relations at machine precision. Coefficients
  do **not** agree (max difference 1.86), and neither do two of the selected
  left-hand sides. With `kappa ~ 1e16`, several exact rearrangements of the same
  relation are equally good, and rounding differences decide which one
  thresholding keeps.
- **float32 changes completely,** as expected: its results were noise-driven
  before this change too, so they are no reference.

## Implementation

### `src/sindy_utils.py`

1. **`_reduce` helper.** Add a private helper next to `_stlsq`:

   ```python
   def _reduce(
       thetas: Float[Array, "groups samples features"],
       Ys: Float[Array, "groups samples targets"] | None,
   ) -> tuple[Float[Array, "groups k features"], Float[Array, "groups k targets"]]:
   ```

   - `Ys is None` (implicit: the targets are the library columns):
     `R = jnp.linalg.qr(thetas, mode="r")`, return `(R, R)`.
   - Otherwise (explicit): QR of `jnp.concatenate([thetas, Ys], axis=2)`, split
     `R[..., :features]` and `R[..., features:]`. The augmentation is what puts
     the targets in the span.
   - `jnp.linalg.qr` batches over the leading `groups` axis.

2. **`_stlsq` changes.**
   - Accept `Ys: ... | None`. `None` means the targets are the library columns.
     The branch is on a Python `None`, so it resolves at trace time. Normalisation
     then uses `tgt = col`.
   - Add `reduce: bool = True` as the last parameter and to `static_argnames`.
     `solve` does not expose it: it exists for the equivalence test and for
     benchmarking.
   - Order of operations: normalise on the full data (column norms of R equal
     those of A anyway), reduce if requested, run the existing double vmap
     unchanged, un-scale unchanged.

3. **`_stlsq_one` and `_lstsq` are untouched.** Masking by multiplication still
   works on R. Masked R columns are no longer triangular, so Lineax keeps using
   SVD; exploiting the triangular structure is out of scope, see below.

### `src/sindy.py`

- Implicit branch of `solve` (`src/sindy.py:281`): pass `None` as the targets
  instead of `thetas`. Passing `thetas` twice would still be correct, but the
  augmented QR would be twice as wide for nothing.
- Explicit branch: unchanged call. It gets the augmented reduction by default.
- `scores`, `select`, `candidates`, `equations`, `conditioning`: unchanged. They
  need the full samples, and each runs once rather than inside the loop.

### `tests/test_sindy.py`

- **New `test_qr_reduction_matches_full_solve`.** Well-conditioned data only:
  `known_system` explicit (degree 2, one control) and `michaelis_menten`
  implicit. Compare `_stlsq(..., reduce=True)` with `reduce=False`: equal
  supports (`!= 0` patterns) and `allclose` coefficients (rtol ~1e-4 in float32).
  The data must stay well-conditioned. On `kappa ~ 1e16` problems coefficient
  comparisons are meaningless (see measurements); compare residuals there.
- **Existing tests.** They assert physical coefficients on well-conditioned data
  and should pass unchanged. The `_stlsq` calls at `tests/test_sindy.py:304` and
  `:317` pass targets positionally, so they keep working. The implicit one
  (`thetas, thetas`) can switch to `None`.
- **Fewer samples than columns.** One small case where `samples < features + targets`:
  R keeps `samples` rows, so there is no reduction. Assert only that the result
  is finite and has the right shape. STLSQ on an underdetermined system has no
  unique answer to compare against.

### Docs (outside `.docs/`, which stays read-only)

- **`README.md`.**
  - Extend the `src/sindy_utils.py` Layout row with "QR-reduced solves".
  - Add a row for `.docs/qr_reduction_plan.md`.
  - Replace the stale mention of an `X64 = True` flag: the example now enables
    x64 unconditionally at import.
  - Add one "things worth knowing" line: on ill-conditioned implicit libraries,
    compare fits by residual, not coefficients. The heading count changes with
    it.
- **`CLAUDE.md`.** Its branch notes still describe `sindy_pi` vs `master`, but
  `sindy_pi` has been merged into `master` (`2ea6db8`). If this work happens on a
  branch, record the reduction there; whether to retire the old notes is the
  user's call.

## Risks, not guarded in code

- **`Ys=None` vs `Ys=thetas`.** Both are correct; the second silently doubles the
  QR width and cost.
- **`samples <= columns`.** No reduction happens; the only extra cost is the QR.
- **Coefficient-level regressions.** Any future test or benchmark that compares
  coefficients on an ill-conditioned library will flake. Compare residuals or
  held-out errors instead.
- **float32 results change** under this plan (measured). They were noise-driven
  already; not a regression criterion.
- **Global x64 leak.** `examples/monod_herbert.py` enables x64 at import, so any
  script importing it runs in float64. The float32 prototype above had to switch
  it back off. Tests don't import examples, so the suite is unaffected.

## Verification

1. `.venv/bin/python -m pytest tests/ -q`: all pass, plus the new test.
2. `.venv/bin/python -m examples.michaelis_menten_sindy_pi` and
   `.venv/bin/python -m examples.lotka_volterra_ude` must print the same equations
   as the README to four decimals.
3. `.venv/bin/python -m examples.monod_herbert` (float64): held-out errors at
   machine precision (~1e-14) for all three equations. The selected LHS may
   differ from the pre-change run.
4. Timing: call `_stlsq(..., reduce=False)` then `reduce=True` on the Monod-Herbert
   normalised thetas and time the second, compiled, call of each. Expect roughly
   the 4x (float64) to 7x (float32) speedup measured above on this machine.
5. Residual equivalence on Monod-Herbert in float64:
   `norm(theta - theta @ models_)` for both paths agrees to ~1e-12.

## Out of scope

- **float64 as the repo-wide default.** This plan makes that cheaper, but it
  remains a separate decision (test tolerances, Diffrax cost, README numbers).
- **Exploiting R's triangular structure.** Column deletion via Givens downdates
  instead of an SVD per iteration: bigger speedups, but masks change every
  iteration and it needs a custom solver in place of Lineax.
- **Running on the GPU** (`jax[cuda12]`, not installed here) and sharding the
  candidate sweep across devices with `shard_map`. Both are orthogonal and work
  on top of the reduction.
