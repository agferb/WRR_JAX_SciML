# jax-training

Differential equations, sparse regression (SINDy), autoencoders, and universal
differential equations in JAX — built on the **Equinox** ecosystem.

## Why Equinox rather than Flax

An `eqx.Module` *is* a PyTree. That single property is what makes this stack
work: a neural network can be passed straight into a Diffrax solver's inner
loop, or sit as a field alongside mechanistic parameters in one differentiable
object, with no `split`/`merge` bookkeeping at the boundary.

```python
class LotkaVolterraUDE(eqx.Module):
    alpha: Float[Array, ""]   # mechanistic, known
    delta: Float[Array, ""]   # mechanistic, known
    net: eqx.nn.MLP           # neural closure for the unknown physics
```

`eqx.filter_grad` differentiates all of it at once; `eqx.tree_at` freezes any
part independently. Diffrax, Optimistix, and Lineax are all Equinox-native, so
the whole pipeline speaks one language.

Flax NNX is the better choice for large-scale distributed training or when you
need pretrained checkpoints — neither applies here.

## Setup

```bash
python -m venv .venv && ./.venv/bin/pip install -r requirements.txt
```

## Layout

| Path | What's in it |
|---|---|
| `src/models.py` | `Autoencoder` and parameter-counting helpers |
| `src/ude.py` | UDE vector fields, the Diffrax `solve` wrapper, multiple shooting, `freeze_mechanistic` |
| `src/sindy.py` | `SINDy` class: per-equation libraries with five levels of control (total degree, per-variable degree, term exclusions, derivative interactions, per-variable interaction degree), control variables, jittable STLSQ (Lineax solves), the SINDy-PI parallel-implicit sweep, candidate scoring and rational-form recovery |
| `src/sindy_utils.py` | Helpers behind `SINDy`: the jittable STLSQ solver (Lineax least squares, masked pruning, optional column/target normalisation) and conditioning diagnostics (`conditioning`/`format_conditioning`: column scale, singular spectrum, numerical rank, condition number), plus library-spec normalisation |
| `src/train.py` | Optax training loop with partial-freezing support |
| `examples/lotka_volterra_ude.py` | Full pipeline: fit a UDE, then recover its closure symbolically |
| `examples/michaelis_menten_sindy_pi.py` | SINDy-PI: recover a rational ODE that no explicit library can represent |
| `examples/monod_herbert.py` | SINDy-PI on a rational Monod-Herbert model: an oversized, poorly conditioned library where recovery fails in float32 and succeeds in float64 -- prints a conditioning report before solving |
| `tests/` | `test_sindy.py`, `test_ude.py`, `test_models.py` — a small suite chosen by mutation testing: each test pins a decision a future edit could reverse |

## The worked examples

### Learning a UDE closure, then naming it

```bash
python -m examples.lotka_volterra_ude
```

Withholds the interaction terms from Lotka-Volterra, learns them with an
embedded MLP, then recovers them as symbols. Runs in ~18s on CPU:

```
Final multiple-shooting MSE: 1.332e-06
Closure relative error:      1.0%, 0.9%

SINDy recovery of the neural closure:
dclosure_x/dt = -0.8935 x*z
dclosure_z/dt = +0.7955 x*z

Ground truth:  dclosure_x/dt = -0.9000 x*z
               dclosure_z/dt = +0.8000 x*z
```

### Recovering a rational ODE (SINDy-PI)

```bash
python -m examples.michaelis_menten_sindy_pi
```

Integrates `dx/dt = 0.6 - 1.5x/(0.3 + x)`, takes it derivatives and sweeps every 
library term as the left-hand side:

```
Best candidates on held-out data:
[3.02e-07] LHS=1               dx/dt = -(-0.1800 +0.9000*x) / (+0.3000 +1.0000*x)
[5.06e-07] LHS=dx              dx/dt = -(-0.1800 +0.9000*x) / (+0.3000 +1.0000*x)
[5.73e-07] LHS=x               dx/dt = -(-0.1800 +0.9000*x) / (+0.3000 +1.0000*x)
[8.81e-07] LHS=x*dx            dx/dt = -(-0.1800 +0.9000*x) / (+0.3000 +1.0000*x)

Ground truth:                  dx/dt = -(-0.1800 +0.9000 x) / (+0.3000 +1.0000 x)
```

All four candidates are one relation at different scales — which is why
`equations()` normalises the denominator before printing.

## Eight things worth knowing

**The closure gain is trainable, and must not start at zero.** The closure is
`gain * net(y)`, and the fitted gain says how much correction the mechanistic
model needed (0.1 → 0.363 in the example). At `gain=0.0` the product rule zeroes
every gradient into `net`, so the network cannot learn until the gain moves —
start it small but nonzero. Freezing it (`freeze_mechanistic(train_gain=False)`)
costs ~7x in loss. Caveat: `gain * net` is over-parameterised, so read the fitted
value as a soft diagnostic, not a measurement.

**Use multiple shooting, not single shooting.** Fitting one long trajectory
plateaus around MSE `1e0` and the closure never becomes identifiable — the
network fits the trajectory while learning the wrong function (measured 52%
closure error at 2e-3 trajectory MSE). Splitting into short data-initialised
segments via `ude.multiple_shooting_windows` drops the loss to `1e-6` and the
closure error under 1%. This is a property of the loss landscape, not of tuning.

**chex needs `eqx.partition` first.** An `eqx.Module` carries non-array leaves
(its activation function), and chex tree assertions assume array leaves:

```python
chex.assert_tree_all_finite(model)          # TypeError on the activation leaf

params, _ = eqx.partition(model, eqx.is_inexact_array)
chex.assert_tree_all_finite(params)         # fine
```

The failure is loud, not silent. `chex.assert_max_traces` composes fine with
`eqx.filter_jit` (put `filter_jit` outermost). For shape *annotations* prefer
`jaxtyping` over `chex.assert_shape` — it's the ecosystem convention.

**Implicit dynamics need SINDy-PI, and masking makes it nearly free.** A rational
right-hand side like `dx/dt = jx - Vmax x/(Km + x)` has no explicit polynomial
representation. Write it as `Theta(x, dx/dt) xi = 0`, set
`interactions_degree` to admit `monomial * dx_k/dt` columns, and pass
`implicit=True`; `solve` then regresses every library column on the others.
Because STLSQ already prunes by *masking* rather than slicing, excluding a
candidate from its own regression is one `~jnp.eye` on the mask — the sweep reuses
the same jitted solver, no new solver code. That solver now exits as soon as the
active set settles (`lax.while_loop`), which is 2.5-6.1x faster than the old fixed
20 iterations at bit-identical output, so `n_iters` became `max_iters` -- a cap,
not a count. Two things to keep in mind: the sweep runs
`n_states * n_features` regressions rather than `n_states` (measured 0.6 ms at 6
candidates, 5.0 ms at 42), and the best candidates are usually the *same relation
rescaled*, so read the rational form `equations()` prints, not the winning
left-hand-side term.

**Column normalisation turns `threshold` into a fraction, not a magnitude.**
`solve`'s STLSQ scales every library column and target to unit 2-norm before
fitting (`normalise=True`, the default) and un-scales the fitted coefficients,
so the same `threshold` is comparable across problems instead of an absolute
coefficient size. Skip it (`normalise=False`) and a badly-scaled library
silently disables sparsity thresholding: a column six orders of magnitude
larger than the rest carries a physically tiny coefficient that a fixed
absolute threshold prunes regardless of its importance. Normalisation is
necessary but not always sufficient. On `examples/monod_herbert.py` at
`threshold=0.01`, float32 gives held-out derivative errors of `0.19`-`0.48`,
while float64 (`X64 = True`) recovers exact implicit relations at `~1e-14`.

**Read `conditioning()` differently for implicit libraries.** If an implicit
library contains the true relation, that relation is an exact null vector of the
admitted columns. `kappa` is then infinite in principle, and `rank <
n_admitted` is guaranteed. On Monod-Herbert (37 admitted columns per equation)
float64 reports rank 36 with `kappa_eps` of 3-5, and that missing direction *is*
the answer. float32 reports rank 17-18 and `kappa_eps` of 11-16, and that gap is
real precision loss. Both verdict lines fire in both precisions, so compare the
ranks rather than reading either verdict alone.

**Pooling trajectories is concatenation, not a new API.** Fit several
trajectories at once by stacking them before `solve`:
`jnp.concatenate([Y1, Y2])`, `jnp.concatenate([dXdt1, dXdt2])`. This buys more
*samples*, not more *directions* -- `conditioning()`'s `rank` reflects the
manifold the trajectories trace, and more samples along the same curve don't
raise it. The `settled` field in the same report makes this concrete: on the
Monod-Herbert training trajectory, 37%-46% of samples (depending on the
equation) have `|dx/dt|` below 1% of its peak and carry almost no dynamical
information.

**Do not pad an implicit library "to be safe".** This is the opposite of the
habit that is harmless in explicit SINDy. If the library can express the true
relation `R = 0` multiplied by another library term, then `R*(1 + a*x)` is
*exactly* as valid for any `a`, and the solver returns an arbitrary member of 
that family. Measured on Michaelis-Menten with `degree=4, interactions_degree=2`
and `normalise=False`, the recovered form grows spurious quadratic terms that
all match `R*(1 + a*x)` for a single `a`. With the default column normalisation,
thresholding happens to prune that family at `threshold=0.05` and returns the
clean form. The non-uniqueness is still there, though, and a different threshold 
or dataset can expose it again. Only a tighter library removes it. Start tight 
and grow; `library_terms()` prints exactly what each equation will sweep.

## Displaying models

`eqx.tree_pformat(model)` for text. For the rich interactive view that
`nnx.display` gives you, use treescope directly — it supports Equinox natively,
and is in fact what `nnx.display` wraps:

```python
import treescope
treescope.show(model)
```

## Notes

- Runs on CPU. For GPU, install a CUDA-enabled `jaxlib`.
