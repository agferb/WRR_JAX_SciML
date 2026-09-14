# Implementing the library-sizing recipe from `monod_herbert_limitations.md`

## Context

`examples/monod_herbert.py` does not recover the true model. `.docs/monod_herbert_limitations.md`
diagnosed why and closed with **§6, a six-item recipe** for sizing a library against
the data you actually have. Nothing in that recipe is implemented — the report ends
with "No code was changed in producing this report."

This plan implements it. Re-measured against the working tree today (`.venv`, float32, CPU):

| quantity | doc says | measured now |
|---|---|---|
| union library columns | 80 | **80** ✓ (but only **37 admitted per equation**) |
| column-norm span | 5.06e7 | **5.058e7** ✓ |
| `kappa(Theta)` raw | ~1e15 | **2.14e15** ✓ |
| `kappa(Theta)` normalised | 3.7e8 | **3.56e8** ✓ |
| numerical rank, raw | 9 | **10** |
| numerical rank, normalised (ramp control) | 20 | **20** ✓ |

The figures hold. Two things in the doc do *not*:

1. **§4 is factually wrong.** It says "the Gram-matrix fast path was left as opt-in in
   `sindy_utils`". `grep -rni gram` hits that sentence and nothing else — no such path
   exists. The only solve is `_lstsq` (`src/sindy_utils.py:31`) via Lineax
   `AutoLinearSolver(well_posed=False)`, which (confirmed in the installed lineax 0.1.1
   source) dispatches to **SVD**. So the repo already scales with `kappa`, not `kappa^2`
   — the right thing, for a reason the doc states but misattributes.
2. The "37 of 80 admitted per equation" split is never mentioned, and it is the number
   that matters for recipe item 3 (rank as a ceiling): the ceiling to compare against is
   **37 admitted vs rank 20**, not 80 vs 20.

### Scoping decisions already taken (from the interrupted session)

| question | decision |
|---|---|
| where normalisation lives | **default-on inside `solve`** |
| how far float64 goes | **example-level opt-in flag**, rest of repo stays float32 |
| `examples/monod_herbert.py` | **fix bugs only, no redesign** |
| singular-value spectrum | **text only**, no matplotlib |

Those last two mean recipe items **4 and 6 are documentation + diagnostics, not an
experiment rewrite**: the code should *measure and report* that the excitation is poor
and the library is oversized, and leave changing the experiment to the user.

---

## The three questions, answered

### Why does JAX default to float32?

JAX inherits XLA's accelerator-first design. On a GPU, float64 throughput is typically
1/32 or 1/64 of float32 on consumer NVIDIA parts; TPUs have no native float64 at all.
Halving the element width also halves memory traffic, and bandwidth — not flops — is
usually what binds. NumPy defaults the other way because it targets CPUs, where the
float64 penalty is roughly 2x rather than 32x.

So x64 is opt-in: `jax.config.update("jax_enable_x64", True)` or `JAX_ENABLE_X64=1`,
and **it must be set before any array is created** — the flag is read at array
construction and trace time, not at solve time.

Worth noting for this repo specifically: jaxlib here falls back to CPU (*"An NVIDIA GPU
may be present... a CUDA-enabled jaxlib is not installed"*). The throughput argument
that motivates the float32 default **does not really bite on this machine** — the cost
of x64 is closer to 2x than 32x.

### What are the consequences of enabling float64?

The gain is the whole point of recipe item 5: `eps` goes `1.19e-7` → `2.22e-16`, so at
the normalised `kappa = 3.56e8`, `kappa*eps` goes from **~42** (no guaranteed correct
digits) to **~7.9e-8** (~7 correct digits).

The costs:

- ~2x memory, and on this CPU roughly 2x wall-clock on an SVD-heavy STLSQ path that is
  already the slowest thing in the repo (an `(1001, 80)` SVD per candidate, vmapped
  over 80 candidates x 3 equations, inside a `lax.while_loop`).
- **It is a global, process-wide switch.** Everything downstream moves: Diffrax in
  `src/ude.py` integrates at a different effective tolerance, every
  `pytest.approx(..., abs=1e-3)` in `tests/test_sindy.py` sits on different digits, and
  the console values quoted in `README.md` shift. This is exactly why the chosen scope
  is example-level and not repo-wide.
- Low risk of leakage, verified: no test imports anything under `examples/`, so a
  module-level flag in `examples/monod_herbert.py` cannot contaminate `pytest`.
- No annotation churn — `jaxtyping`'s `Float[Array, ...]` accepts both widths.

### How much would float64 actually improve identifiability?

**Much more than the doc implies — on this sweep it is decisive, not a finishing
touch.** I prototyped normalisation + the SINDy-PI sweep in both precisions and scored
on the held-out `x0 = [25, 40, 0]` trajectory:

```
float32 thr=0.001:  lhs=['s*s*o','s*s*o','s*o*o*u']   err=[0.638,   0.121,   0.0102 ]
float32 thr=0.01:   lhs=['x*o','x*o','s*o*o*u']       err=[0.00397, 5.1e-08, 0.0102 ]
float32 thr=0.05:   lhs=['s*s*x*o','x','s*s*x*o']     err=[0.000689,5.1e-08, 0.003  ]

float64 thr=0.001:  lhs=['s','x','s*u']               err=[1.3e-14, 2.0e-15, 6.0e-15]
float64 thr=0.01:   lhs=['s','x','s*u']               err=[1.3e-14, 2.0e-15, 6.0e-15]
float64 thr=0.05:   lhs=['s*o*dx_k','x','dx_k']       err=[3.8e-14, 2.0e-15, 6.9e-15]
```

Two things jump out. **float64 finds exact implicit relations** — held-out derivative
error at machine precision, five to thirteen orders of magnitude below anything float32
manages. And **its selection is stable**: identical across two decades of `threshold`,
where float32's answer changes completely at every threshold, which is the signature of
thresholding on noise.

This corrects the doc. §2's "the true coefficients are recovered exactly once columns
are normalised" was measured on the **oracle 7-column support** (`kappa = 1.2e3`), and
does not carry to the full 37-column sweep — where `kappa*eps ~ 42` in float32, exactly
as §3 warns. **Normalisation is necessary but not sufficient here.** The `kappa*eps`
bound predicted this outcome correctly; the doc simply did not follow it through.

Two honest caveats:

- Machine-precision error means *an* exact relation was found, not necessarily the true
  one in its nicest form. The true system has three exact implicit relations, and any
  algebraic rearrangement of one is equally exact — which is why `thr=0.05` picks
  different left-hand sides at the same error.
- float64 still **does not raise the rank**. Rank 20 of 37 admitted is a property of the
  data manifold: one trajectory of a 3-state ODE traces a 1-D curve. §4 shows the
  spectrum decays smoothly with no cliff, so a moved tolerance reveals nothing. Richer
  excitation (item 4) remains the only fix for that.

**Revised ranking for this problem:** normalise columns → **enable float64** → enrich
excitation → shrink the library. float64 moves from fourth to second.

> This is new evidence against the settled "example-level opt-in" scope — it suggests
> x64 may deserve to be the default for implicit sweeps generally. I have **not**
> changed the scope; the example flag is what lets you reproduce the table above in one
> run and decide. Worth revisiting once it is in.

---

## Implementation

### 1. Column normalisation, default-on (`recipe item 1`)

**`src/sindy_utils.py`** — normalise inside `_stlsq`, which is the single choke point
both modes already pass through (explicit sends one group, the SINDy-PI sweep sends
`n_states` groups).

- Add `normalise: bool = True` to `_stlsq` and to its `static_argnames` alongside
  `max_iters`.
- Scale both arguments by their own column 2-norms, guarding zero columns with
  `jnp.where(norm > 0, norm, 1.0)`:
  - `col = ||thetas[g, :, f]||` shape `(groups, features)`
  - `tgt = ||Ys[g, :, t]||` shape `(groups, targets)`
- Fit on the scaled pair, then un-scale on return so callers never see scaled numbers:

  ```
  xi[g, f, t] = xi_n[g, f, t] * tgt[g, t] / col[g, f]
  ```

  This one expression is correct for **both** modes. In implicit mode the targets *are*
  library columns, so `tgt is col` and it collapses to `xi_n * col[:, None, :] / col[:, :, None]`.

**`src/sindy.py:236` `solve`** — add `normalise: bool = True`, pass it through both
branches. `coefficients_` and `models_` stay physical, so `scores`, `select`,
`candidates`, `_rational` and `equations` need **no change at all**.

**What this changes for callers:** `threshold` stops being an absolute coefficient
magnitude and becomes a dimensionless fraction-of-target — comparable across problems,
which is the point. Document that in the `solve` docstring and the class docstring.

### 2. Conditioning diagnostics, text-only (`recipe items 2, 3, 4, 5`)

Follow the data/render split the repo already uses for `candidates()` / `equations()`
(noted as a deliberate design in `CLAUDE.md`).

Two layers, as chosen: pure functions in `sindy_utils` over a raw matrix, and thin
`SINDy` methods that build the thetas, apply the mask and delegate.

**`src/sindy_utils.py`** — a pure function `conditioning(theta) -> dict` computing, from
one `jnp.linalg.svd(theta, compute_uv=False)`:

| key | recipe item | note |
|---|---|---|
| `col_scale_span`, `col_scale_min/max` | 1 | the root-cause number |
| `sigma` (normalised `sigma_i / sigma_0`) | 2 | the spectrum itself |
| `kappa_raw`, `kappa_normalised` | 2, 5 | |
| `rank`, `n_admitted` | 3 | the ceiling, and what is being asked of it |
| `eps`, `kappa_eps`, `digits_lost` | 5 | `eps` read from the **active dtype**, not hardcoded |

Use the `numpy.linalg.matrix_rank` tolerance the doc documents in §4
(`sigma > sigma_max * max(m, n) * eps`) computed explicitly, so the number is
reproducible and dtype-aware.

Plus `format_conditioning(report) -> str` alongside it, rendering one report as text:
the spectrum as a `log10(sigma_i / sigma_0)` column with an ASCII bar so the cliff (or
its absence) is visible without matplotlib, truncated around the rank cutoff with the
cutoff row marked, exactly as §4 prints it.

**`src/sindy.py`** — two thin methods mirroring the `candidates`/`equations` split:

- `SINDy.conditioning(Y, dXdt) -> list[dict]` — builds thetas via `_build_thetas`
  (implicit) or `_build_theta` (explicit), slices each equation's admitted columns with
  `library_mask`, and calls the `sindy_utils` function once per equation.
- `SINDy.diagnostics(payload) -> str` — joins `format_conditioning` over the payload,
  labelled per equation.

For **item 4** ("add directions, not samples") the decision is **diagnose only** — no
new pooling API, since concatenating `Y`/`dXdt` before `solve` already works and the
report shows extra samples buy no rank anyway. What the report gains is the statistic
that makes the point measurable rather than advisory: the settled fraction from §3,
`samples with |dx/dt| > 1% of its peak`, plus the index of the first fully-settled
sample. That says "a third of your samples carry no dynamical information" in numbers.
Document the concatenate-to-pool technique in the README rather than wrapping it.

For **item 6** ("prefer structure over degree") the renderer should flag, in one line,
when `n_admitted > rank` — the "you are asking the data to distinguish more columns than
it can" verdict — and when `kappa_eps > 1`, the float64 suggestion.

### 3. float64 opt-in in the example (`recipe item 5`)

**`examples/monod_herbert.py`** — a module-level switch at the very top, before any
array exists:

```python
X64 = False  # recipe item 5: set True to measure the float32 margin
if X64:
    jax.config.update("jax_enable_x64", True)
```

It must sit above the `from src import sindy, ude` import chain's first array
construction. Then call `model.diagnostics(model.conditioning(ys_train, dxs_train))`
before `solve`, so a run prints its own conditioning verdict.

`threshold = 0.001` at `examples/monod_herbert.py:82` was tuned against unnormalised
columns and **must be retuned** once normalisation is on — with unit-norm columns and
targets, a fraction-of-target threshold in the `0.01`-`0.1` range is the right
starting scale.

**The "fix bugs only" bug** (`examples/monod_herbert.py:116`):

```python
for lib, state in enumerate(lib_terms):     # lib_terms is a dict
    print(f"\n{state}:  {lib}")
```

`library_terms()` returns `dict[str, list[str]]`, so `enumerate` yields
`(index, key)` — this prints `ds:  0`, never the terms it claims to show, and it
shadows the `lib` spec dict on the way. Should be `for state, terms in lib_terms.items()`.

### 4. Test updates

`tests/test_sindy.py` asserts *physical* coefficients almost everywhere
(`approx(-2.0)`, `approx(3.0)`, ...), and un-scaling keeps those valid — so most of the
suite should pass untouched. The exception is threshold semantics:

- **`test_stlsq_converges_and_respects_the_threshold` (`tests/test_sindy.py:225`)** —
  lines 246-248 pin `threshold=1.5` pruning `u` and `threshold=0.5` keeping it, which
  relies on the true `u` coefficient being `1.0` in raw units. Under normalisation the
  pruning coefficient is dimensionless; retune both values and keep the *intent* (one
  threshold above, one below) rather than the numbers.
- Add one test pinning the new contract: fitting a badly-scaled library (scale one
  column of `known_system` by `1e6`) recovers the same physical coefficients with
  `normalise=True` and fails to with `normalise=False`. This is the regression test for
  the entire root cause.
- Add one test that `conditioning()` is dtype-aware and that `kappa_eps` on a
  deliberately ill-conditioned `theta` exceeds 1.

### 5. Documentation

- **`.docs/monod_herbert_limitations.md` §4** — delete the Gram-matrix sentence. Replace
  with the verified fact: `_lstsq` uses Lineax `AutoLinearSolver(well_posed=False)`,
  which dispatches to SVD, so the repo already scales with `kappa` rather than `kappa^2`.
- **§2 / §3** — correct the scope of "the true coefficients are recovered exactly once
  columns are normalised": that is the **oracle 7-column** result and does not carry to
  the full sweep, where float32 still fails. Record the measured float32-vs-float64
  table, and promote float64 from fourth to second in the Summary's ordering.
- **§1 / §6** — note that only 37 of the 80 union columns are admitted per equation, and
  that item 3's ceiling comparison is 37-vs-rank, not 80-vs-rank. Add a closing line to
  §6 recording that items 1, 2, 3 and 5 are now implemented, and where.
- **`README.md`** — per the repo's update policy:
  - the Layout table points at `docs/monod_herbert_limitations.md` and
    `docs/noise_models.md`; both actually live in **`.docs/`**. Fix the paths.
  - `examples/monod_herbert.py` is missing from the Layout table entirely. Add it.
  - update the `src/sindy_utils.py` row to mention conditioning diagnostics.
  - add a "Five things worth knowing" entry for column normalisation — that `threshold`
    is now a fraction of the target, why an unnormalised polynomial library silently
    disables sparsity thresholding, and the float32-vs-float64 result above.
  - document pooling trajectories by concatenating `Y`/`dXdt` before `solve`, with the
    §3 caveat that extra samples on the same curve buy no rank.

---

## Verification

1. `.venv/bin/python -m pytest tests/ -q` — 23 tests, all passing, with only the
   threshold test's numbers changed.
2. `.venv/bin/python -m examples.michaelis_menten_sindy_pi` and
   `.venv/bin/python -m examples.lotka_volterra_ude` — both still recover their systems;
   the Lotka-Volterra console values in `README.md` should still match to the precision
   quoted there (`CLAUDE.md` notes they already move at ~2e-7 on this branch).
3. `.venv/bin/python -m examples.monod_herbert` — prints a conditioning report before
   solving. Confirm it reports `col_scale_span ~ 5.06e7`, `kappa_raw ~ 2.14e15`,
   `kappa_normalised ~ 3.56e8`, `rank 20`, `n_admitted 37`, and a `kappa*eps > 1`
   float32 warning.
4. Flip `X64 = True` and re-run. This must reproduce the table in the float64 answer
   above: `kappa*eps` drops below 1, the warning clears, rank stays ~20, and the
   held-out derivative errors fall to `~1e-14` with the selection stable between
   `threshold=0.001` and `0.01`. If it does not, the normalisation un-scaling is wrong —
   this is the sharpest end-to-end check in the plan.
5. Oracle check for the normalisation claim: fit only the true 7-column support with
   `normalise=True` in float32 and confirm the worst coefficient error is ~0%, matching
   §2 of the doc — and note it does *not* generalise to the full sweep.

## Files

| file | change |
|---|---|
| `src/sindy_utils.py` | `normalise` in `_stlsq` (+ static arg); new `conditioning()` and `format_conditioning()` |
| `src/sindy.py` | `normalise=True` on `solve`; thin `conditioning()` / `diagnostics()` methods; docstrings |
| `examples/monod_herbert.py` | `X64` flag; diagnostics call; retuned `threshold` |
| `tests/test_sindy.py` | retune threshold test; +2 tests |
| `.docs/monod_herbert_limitations.md` | §4 correction; 37-of-80 note; implementation status |
| `README.md` | `.docs/` paths; `monod_herbert.py` row; normalisation note |
