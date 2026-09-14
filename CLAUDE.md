# CLAUDE.md

## Branch notes

### `sindy_pi` (vs `master`)

Extends `src/sindy.py` with the SINDy-PI parallel-implicit formulation
(Kaheman, Kutz & Brunton, arXiv:2004.02322):

- Library gains `interactions_degree` / `var_interactions_degree`, admitting
  `monomial * dx_k/dt` columns. Index `n_states` inside a monomial tuple is a
  sentinel for "this equation's own derivative", so mixed (`dx0/dt * dx1/dt`) and
  higher-order (`d2x/dt2`) derivative terms cannot be expressed at all.
- `SINDy(..., implicit=True)` sweeps every library column as the left-hand side.
  `solve` writes `models_` instead of `coefficients_`; `scores`, `select`,
  `candidates` and the rational form in `equations()` are implicit-only.
- `_stlsq` is one double-vmapped function (targets inner, equation groups outer)
  and runs `lax.while_loop` to convergence, so `n_iters` is now `max_iters`.
  `while_loop` is not reverse-differentiable; nothing here backprops through it.

- Control variables: `SINDy(..., n_controls=k)` adds exogenous variables that
  build the library but get no equation of their own. `var_names` lists states
  first, then controls; positional tuples (`var_degree`,
  `var_interactions_degree`, `exclude`) are indexed over
  `n_variables = n_states + n_controls`. A short `var_degree` is silently
  truncated by `zip`, leaving trailing controls uncapped -- documented, not
  guarded.
- `candidates()` returns data (one dict per equation with `term` / `error` /
  `explicit_form`), and `equations(payload)` renders it. `top=None` keeps all.
- `solve(..., normalise=True)` (default) scales `_stlsq`'s columns and targets
  to unit 2-norm before fitting and un-scales the result, so `threshold` reads
  as a dimensionless fraction of the target rather than a raw coefficient
  magnitude -- this changes behaviour vs `master`, not just lowering. Pass
  `normalise=False` for `master`'s absolute-magnitude semantics.
- `SINDy.conditioning(Y, dXdt) -> list[dict]` / `.diagnostics(payload) -> str`
  report per-equation rank, condition number and column-scale span on each
  equation's masked columns (`sindy_utils.conditioning`/`format_conditioning`
  do the underlying SVD work over a raw matrix, given whatever columns they
  are handed).

The explicit path is otherwise unchanged in behaviour but **not** bit-identical
to `master`: the extra vmap axis changes XLA's lowering, moving float32
coefficients by ~7e-7 (~2e-7 relative). `examples/lotka_volterra_ude.py` prints
the same values to four decimals.
