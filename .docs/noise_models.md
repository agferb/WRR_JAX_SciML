# Adding noise to synthetic trajectories

Design note for `examples/monod_herbert.py:trajectory`. Every number here was
measured on the Monod-Herbert system with `x0 = [0, 15, 0]`, `dt = 0.01`,
`t_span = (0, 10)` and the clipped-ramp control `kLa(t) = clip(6(t-1), 0, 7)`.

## The decision that comes before the three

Where noise enters relative to the derivative computation matters more than
which distribution it is drawn from:

| where | what `dXdt` becomes | derivative error at 5% |
|---|---|---|
| noise, then vector field | true model evaluated at noisy states | **220%** (16% if clipped) |
| noise, then finite differences | what a real experiment gives you | **235%** |
| vector field, then noise on states only | clean derivatives, noisy states | 0% |

Finite differences on *clean* states are 0.3%, so at `dt = 0.01` the
differentiation step amplifies 5% state noise by **47x**.

The third row deserves naming. Adding noise *after* computing `dxs` yields an
inconsistent pair: states that do not produce those derivatives. For explicit
SINDy that is a defensible "the derivatives were measured separately" story. For
**SINDy-PI it is not**, because implicit relations place `x` and `dx/dt` in the
same row (`x*dx_k`), so the inconsistency corrupts the columns the method
depends on.

A related trap: evaluating the true `vector_field` at noisy states is circular —
it uses the model being identified. On Michaelis-Menten that route is also
numerically immune to noise (5% state noise cost it nothing). That immunity does
**not** transfer to Monod, where the same route gives 16-220%. The circularity
argument holds regardless of the numbers.

---

## Alternative 1 — Additive Gaussian, scaled per channel

```python
sigma = noise * xs.std(axis=0)                    # one sigma per state
xs = xs + sigma * jax.random.normal(key, xs.shape)
```

**Methodology.** Classic i.i.d. observation noise, `x_obs = x + eps` with
`eps ~ N(0, sigma_i^2)`, each `sigma_i` tied to that channel's own spread.

**Benefits.** The standard model in the SINDy literature, so results compare
against published work. One knob. Per-channel scaling is not optional here — the
channels differ by **10.8x** in standard deviation:

```
       channel       min       max      mean       std
 S (substrate)     0.000    98.608    66.502    23.553
   X (biomass)     0.311    31.162    17.322    13.179
O (dissolved O2)   -0.000     6.722     2.808     2.178
```

A single global sigma at `noise = 0.05` lands as **6.7% / 11.9% / 72.1%**
relative noise — it obliterates the DO channel while barely touching substrate.

**Drawbacks.**

*It pushes concentrations negative.* 5.7% of DO samples and 2.3% of biomass
samples fall below zero. Worse than cosmetic: the growth term divides by
`(O + K_O)` with `K_O = 0.2`, and **3 samples out of 1001** flip that
denominator's sign. Those three alone drive the derivative error from 16.4% to
**220.4%**.

*47x amplification into derivatives.* Unusable with raw `jnp.gradient`; needs
Savitzky-Golay smoothing, TV-regularised differentiation, or the weak form.

---

## Alternative 2 — Multiplicative (relative) noise

```python
xs = xs * (1 + noise * jax.random.normal(key, xs.shape))
```

**Methodology.** Error proportional to the reading, `x_obs = x * (1 + eps)`. This
is how instrument accuracy is specified for concentration sensors — "±2% of
reading", not "±2 mg/L".

**Benefits.** The right physical model for bioprocess measurement. Handles the
10.8x scale spread automatically with no per-channel bookkeeping. Measured
**0.0% sign flips** where the signal was positive, so concentrations stay
physical and the `(O + K_O)` denominator stays safe — derivative error 32.4%
rather than 220%.

**Drawbacks.** It **vanishes where the state is near zero**. DO starts at exactly
0.0, so sigma there is 0.0000: the early transient arrives noise-free, which is
unrealistic and is precisely the region carrying the most information about the
kinetics. It is also heteroscedastic, so effective SNR drifts along the
trajectory and "5% noise" is harder to reason about as one number. The usual fix
is a floor: `sigma = noise * (abs(x) + x_floor)`.

---

## Alternative 3 — Process noise (SDE), not observation noise

```python
# dx = f(t,x,u) dt + g(x) dW  -- requires extending ude.solve to SDE terms
terms = dfx.MultiTerm(dfx.ODETerm(vector_field), dfx.ControlTerm(diffusion, bm))
```

**Methodology.** Noise enters the *dynamics* rather than the measurement, via
`dfx.VirtualBrownianTree` + `MultiTerm` and an SDE solver (Euler-Maruyama or
Ito-Milstein). Models feed fluctuation, imperfect mixing, temperature drift.

**Benefits.** The trajectory stays a **self-consistent realisation**: the states
genuinely satisfy a dynamical law, so finite differences of the path are that
path's real derivatives. No `x`/`dx` inconsistency — the failure mode that hurts
SINDy-PI most. It is also the only option producing *correlated* deviations; real
process upsets persist across samples, where i.i.d. observation noise does not.

**Drawbacks.** The heaviest option by a distance. It changes what is being
identified — SINDy recovers the **drift**, and the Ito-vs-Stratonovich convention
affects the answer. Brownian paths are nowhere differentiable, so finite
differences of the path do not converge as `dt -> 0`; they *diverge* like
`1/sqrt(dt)`, breaking the usual "sample finer for better derivatives" instinct.
It needs a real change to `ude.solve` (SDE terms, a Brownian tree, a different
solver), costs substantially more compute, and `noise = 0.05` loses its obvious
interpretation.

---

## Recommendation

**Alternative 2 with a floor**, `sigma = noise * (abs(x) + x_floor)`, paired with
**finite-difference derivatives**, applied where the `if noise > 0:` block
already sits.

It is the physically correct measurement model for concentrations, it survives
the `(O + K_O)` hazard that makes Alternative 1 explode, and the floor repairs
its one real weakness. Finite differences rather than the vector-field route
because the point of adding noise is to stop flattering the method, and the
vector-field route cannot show measurement error honestly.

Alternative 3 answers a different question — "how does the method behave under
process variability" rather than "under measurement error". Worth having later;
not worth the machinery now.

To stay with Alternative 1 for comparability with the literature, clip:
`jnp.maximum(xs + sigma * eps, 0.0)` takes the derivative error from 220% to
16.4%.

## Critical mismatches, not guarded in code

- `noise` needs a PRNG key. Hard-coding one makes every call return the same
  realisation, silently ruining held-out scoring — train and test would share a
  noise draw. It should be a parameter.
- `trajectory` is called twice in `main` (train and test); both must get
  *different* keys.
- Controls are normally known exactly, so leave `us` clean. The current ordering
  (noise before `concatenate`) already does this.
- The 47x amplification is a function of `dt`, so changing `dt` silently changes
  the effective derivative noise.
