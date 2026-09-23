# Flat polynomial library for the SymDer port

Plan for giving `_hidden_vars/sym_models.py` the nested levels of library control
that `src/sindy.py` already has (`degree`, `var_degree`, `exclude`, `bias`),
under a flat feature-vector architecture.

Scope: explicit SINDy only. `interactions_degree` / `var_interactions_degree`
(SINDy-PI) are out of scope and must be rejected, not silently ignored.

## Context

The ported library (`Linear`, `Quadratic`, `Cubic`, `SymModel`) is the reference's:
one dense weight tensor per degree, `(n_out, n_dims, ...)`, with a symmetry mask.
It admits *every* monomial of each degree, so there is no way to say "degree 3
overall but this variable only linear", or to drop individual terms.

`src/sindy.py` already expresses all of that as a predicate over exponent vectors
(`_allows`) evaluated on an explicit combos list, yielding a `(equations, features)`
boolean `library_mask`. This plan adopts that representation in the SymDer model.

Diverging structurally from `symder_ref` is explicitly accepted.

## Measured basis

`n_dims = 14` (10 states + 3 hidden + 1 control), `n_out = 13`, degree <= 3,
680 features. Weights passed as **traced inputs**, as in a jitted training step --
with zero weights baked in as constants XLA folds the contraction away and the
timings are meaningless.

| implementation | batch 1024 | `dfunc` order 2 | trainable params |
|---|---|---|---|
| dense `Linear+Quadratic+Cubic` (current) | 8.71 ms | 0.246 ms | 38 415 |
| flat, `y ** exps` exponent matrix | 6.13 ms | 0.145 ms | 8 840 |
| **flat, gather with sentinel** | **1.34 ms** | **0.030 ms** | **8 840** |

Feature counts: degree <= 2 -> 105, <= 3 -> 560, <= 4 -> 2380 (at `n_dims = 13`).
Dense tensors hold `n_out * n_dims**d` weights, of which only the unique monomials
are useful: 20.7 % at degree 3, 6.4 % at degree 4.

### Why the gather form, not the exponent matrix

`jnp.prod(y ** exps, axis=-1)` is the obvious vectorisation and it is **wrong here**:
`d/dy` of `y ** 0` evaluates `0 * y ** (-1)`, so the gradient is NaN wherever any
component of `y` is 0 -- and `dfunc(f, 2)` returns NaN at such a point.
`examples/monod_herbert.py` starts at `x0 = [0, 15, 0]`, and hidden variables can
cross zero at any time. A `jnp.where` on the *result* does not fix it (the untaken
branch is still evaluated); substituting the base before the power does, at
70 vs 66 jaxpr equations.

The gather form sidesteps the issue entirely -- there is no `pow` at all -- and is
4.6x faster than the guarded power form, because its intermediate is
`(B, F, max_degree)` (2040 elements per sample) rather than `(B, F, n_dims)` (9520).

## Design

```python
class PolynomialLibrary(eqx.Module):
    w:    Float[Array, "n_out n_features"]          # trainable, zero init
    mask: Bool[Array, "n_out n_features"]           # admissibility, per equation
    idxm: Int[Array, "n_features max_degree"]       # sentinel-padded factor indices

    def __call__(
        self, z: Float[Array, "... n_dims"], t: Float[Array, ""] | None = None
    ) -> Float[Array, "... n_out"]:
        y = jnp.concatenate([z, jnp.ones(z.shape[:-1] + (1,))], axis=-1)
        theta = jnp.prod(y[..., self.idxm], axis=-1)
        return jnp.einsum("...f,of->...o", theta, self.mask * self.w)
```

`idxm` holds, for each library term, the indices of the variables multiplied
together, right-padded with a sentinel index `n_dims` that points at an appended
constant `1.0`. With `n_dims = 4` and `max_degree = 3`:

| term | combo | `idxm` row | evaluates |
|---|---|---|---|
| `1` | `()` | `[4, 4, 4]` | `1*1*1` |
| `x0` | `(0,)` | `[0, 4, 4]` | `y0*1*1` |
| `x1*x2` | `(1, 2)` | `[1, 2, 4]` | `y1*y2*1` |
| `x0*x0*u0` | `(0, 0, 3)` | `[0, 0, 3]` | `y0*y0*y3` |

Three properties fall out:

- **The bias is the empty combo** -- its row is all sentinels, so `theta[0] == 1`.
  `bias=False` clears that mask column; nothing special-cases it.
- **Only `w` is trainable.** `mask` is bool and `idxm` is int, so
  `eqx.is_inexact_array` filtering skips both, with no `static=True` (a jnp array
  is unhashable and cannot be static).
- **No redundant metadata.** `n_dims`, `n_features` and `max_degree` are implied by
  the field shapes; term names are recovered from `idxm`.

### Spec compilation

Local, ~10 lines, plus the shared normaliser:

```python
from src.sindy_utils import _normalise_spec

def _allows(spec: dict, combo: tuple[int, ...], n_dims: int) -> bool:
    exponents = tuple(combo.count(i) for i in range(n_dims)) + (0,)   # trailing derivative slot
    if exponents in spec["exclude"]:
        return False
    total = sum(exponents[:-1])
    if total > spec["degree"]:
        return False
    if total == 0:
        return spec["bias"]
    cap = spec["var_degree"]
    return cap is None or all(e <= c for e, c in zip(exponents[:-1], cap))
```

Verified to reproduce `SINDy.library_mask` bit-for-bit on three specs, including an
`inf` wildcard and a short `var_degree`. Notes:

- The trailing `0` matters: `_normalise_spec` pads `exclude` entries to `n_dims + 1`
  because SINDy reserves the last slot for the derivative factor. Take it as a
  parameter defaulting to `0` rather than hard-coding it, so SINDy-PI can reuse this.
- A short `var_degree` leaves trailing variables uncapped, because `zip` truncates.
  Inherited from `src/sindy.py` deliberately; document it, do not guard it.
- `_SPEC_KEYS` still accepts the two SINDy-PI keys, so `_normalise_spec` will not
  complain about them. Reject them explicitly -- they would generate derivative
  columns this library cannot evaluate.

Combos are built exactly as in `src/sindy.py`: `[()]` then
`itertools.combinations_with_replacement(range(n_dims), d)` for `d = 1..max_degree`,
so feature order matches `SINDy` column for column.

### Constructor and API

- `PolynomialLibrary(n_out, spec, n_dims=None, var_names=None)` -- `spec` is one dict
  or a list of `n_out` dicts (per-equation libraries).
- `n_out` is the number of equations (visible + hidden); `n_dims` is the latent width
  `[visible | hidden | controls]`. The control tail gets no equation, which maps onto
  `SINDy`'s `n_states` / `n_controls` split exactly.
- `feature_names(var_names=None)`, `library_terms(var_names=None)` -- derived from
  `idxm` and `mask`, so the description cannot disagree with what is evaluated.
  `library_terms` mirrors `SINDy.library_terms()`'s shape of output.
- `n_terms()` -- admitted terms per equation; the honest denominator for a sparsity
  report, since the raw parameter count includes structural zeros.
- `var_names` is a method argument, not a stored field: storing it would need
  `static=True` and changing names would trigger a recompile. It covers all `n_dims`
  columns in order `[visible | hidden | controls]`.

### What happens to the existing classes

- `Linear`, `Quadratic`, `Cubic` are **removed** -- superseded, no callers, and two
  representations of the same library would drift.
- `SymModel` **stays unchanged**: it is the summing wrapper `dfunc` receives, and it
  is how a non-polynomial term (a rational one, later) gets added alongside.
- Migration: `SymModel([Linear(4), Quadratic(4)])` becomes
  `SymModel([PolynomialLibrary(n_out=4, spec={"degree": 2}, n_dims=4)])`, or the
  library on its own.
- A library with `n_out < n_dims` still cannot be the vector field by itself:
  `odeint_zero` requires `f(y)` to match `y`'s width, so the caller concatenates the
  supplied control derivative, `F(y) = concatenate([library(y), du_dt])`.

## Verification

Validated already, in a prototype:

- mask identical to `SINDy.library_mask`; output matches `_build_theta(Y, 0) @ xi`
  to 9.5e-7 (float32);
- zero init returns zeros; batch axes `(n_dims,)`, `(20, n_dims)`, `(4, 5, n_dims)`;
- `dfunc` order 2 matches `jvp(F, F)`; the control row of `d2` is exactly 0.0;
- only `w` is trainable; masked gradients are exactly zero; `grad.mask`/`grad.exps`
  are `None`;
- **NaN regression:** `d2x/dt2` and gradients are finite at `y = [0, 2, 0, 1]`.

Still to test at implementation time: per-row spec lists with differing dicts;
`inf` vs `True` wildcards; short `var_degree`; `degree=0` with `bias=False`;
that the interactions-key rejection fires; and that `idxm` is an integer dtype
(as float it becomes trainable and the optimizer would update the indices).

## Risks

- `idxm` dtype must be int -- silent and severe if wrong.
- The local `_allows` duplicates `SINDy._allows`'s explicit branch and can drift.
  The equivalence test is the guard; hoisting `_allows` into `src/sindy_utils.py`
  would remove the duplication but requires editing `src/`.
- Init cost grows with degree (Python-level combos and mask construction): ~210 ms
  at `n_dims = 14`, degree 3; roughly 4x that at degree 4. One-time, per construction.
- Passing an `eqx.Module` directly as `odeint_zero`'s `func` works in this JAX
  version despite `hash(SymModel)` raising `TypeError: unhashable type: ArrayImpl`.
  It is version-dependent luck; callers should pass a plain function.
