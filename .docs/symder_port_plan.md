# Porting `symder.py`: observation map, pinning, and the derivative stack

Plan for `_hidden_vars/symder.py`, replacing the reference's `get_symder_apply`
and `get_model_apply` (`symder_ref/symder/symder/symder.py`).

## Context

The reference assumes the observables **are** state variables: `transform` slices
`z[..., :num_visible]`, `hidden_transform` slices `z[..., -num_hidden:]`, and
`concat_visible` pins the measured series into the latent vector.

Two things break that here:

1. **Observables can be sums of states** (total COD, total biomass), often with
   stoichiometric weights. Then no single state is read off a measurement.
2. **The latent layout is `y = [states | controls]`**, so `z[..., -num_hidden:]`
   would silently return the *control* tail. Copying the reference's lambda gives a
   model that trains and reports hidden diagnostics computed from the control signal.

Both measured and summed observables must be supported in the same model: some
applications have directly measured states, some do not.

Only `num_der = 2`. Explicit SINDy only.

## The observation map

Declared by name, compiled to a matrix `C` of shape `(n_obs, n_states)` with
`x_obs = C @ z`:

```python
Observation(
    state_names=["S_S", "S_I", "X_S", "X_H", "X_A", "S_O"],
    observables={
        "COD":   {"S_S": 1.0, "S_I": 1.0, "X_S": 1.6},   # weighted sum
        "X_tot": ["X_H", "X_A"],                          # plain sum, weights 1
        "S_O":   "S_O",                                   # direct measurement
    },
)
```

A matrix rather than a lambda because:

- **It is inspectable.** A **zero column** is a state no measurement touches -- an
  identifiability red flag reportable at construction (`unobserved_states()`)
  instead of after a failed run.
- **It commutes with time differentiation** (verified): for linear `C`,
  applying it inside `dfunc` and after `dfunc` agree exactly, so it can be applied
  once to the stacked derivatives. For a nonlinear observable (`|z|**2`, the
  reference's NLSE case) they differ -- verified -- so a callable observation must
  be passed into `dfunc` as `transform`.
- Slicing is the special case where `C` is a selection matrix, so the reference's
  behaviour is still expressible.

**Rule: linear `Observation` is applied outside `dfunc`; a callable is applied inside.**

## Pinning: `concat_visible`, generalised

The reference concatenates the measured series into the front of `z`. Generalise it
to a scatter, keeping the capability but not the layout assumption.

A row of `C` with a **single** nonzero weight `w` at state `s` measures that state
directly: `z[s] = obs / w`, and `dz[s]/dt = dobs/dt / w` straight from `dvs`.
Those states are **pinned**; the encoder produces only the rest.

- `pin_measured: bool = True` (default reproduces the reference). Set `False` when
  measurements are noisy enough that pinning would inject that noise straight into
  the latent state, and the encoder should infer even measured states.
- Assembly scatters both groups into the full state vector at their declared
  indices -- `zeros.at[..., pinned_idx].set(...).at[..., free_idx].set(encoder_out)`
  -- rather than assuming the measured ones come first.
- **Pinned states get their `dz/dt` from the data**, not from the encoder JVP:
  cheaper and more accurate. Only the free states need the JVP.
- The encoder's output width is `n_states - n_pinned`, which the encoder port must
  be told. Flagged for that file, not decided here.

With every observable a direct measurement, this reduces exactly to the reference.
With none, every state is free and nothing is concatenated.

## `hidden_transform` disappears

It presumed a visible/hidden split inside the latent. With pinning derived from `C`
and controls at the tail, the `dzdt` regulariser applies to the **free** states --
exactly those the encoder had to infer, which is what the term is for. No argument,
no lambda, no layout assumption.

## Structure

`SymDerModel` (`eqx.Module`) holding `encoder`, `library`, `observation`, and static
`n_states` / `n_controls` / pinning index arrays. It replaces both factory functions;
Equinox modules carry their own parameters, so the closure-returning
`get_*_apply` pattern has no purpose.

**Parameters must travel as an explicit argument, never a closure.** Closing over a
traced library inside `dfunc`'s `func` raises
`UnexpectedTracerError: ... wrapped in a LinearizeTracer to escape the scope`,
because `odeint_zero` marks `func` as `nondiff_argnums`. This is why the reference
writes `func(z, t, params)`. With Equinox:

```python
arrays, static = eqx.partition(self.library, eqx.is_inexact_array)
def field(y, t, p):
    return jnp.concatenate([eqx.combine(p, static)(y), du_dt], axis=-1)
```

`du_dt` must be broadcast to `y.shape[:-1] + (n_controls,)`. A bare
`(n_controls,)` array works for one sample and fails for `(batch, time)`, with the
error surfacing from deep inside the JVP -- so the broadcast lives in one place
where a caller cannot get it wrong.

## `dzdt`: the middle option

No `get_dzdt` boolean threaded through the code. The forward takes `dvs` and
computes the encoder JVP only when it is not `None`:

```python
def latent(self, vs, dvs=None):   # -> (z, dzdt or None)
```

`reg_dzdt=None` means the caller passes `dvs=None`. Measured: always-on `dzdt`
costs **+58% per training step** (25.0 -> 39.5 ms at Lorenz scale: 2 visible,
1 hidden, T=10000, 128-channel encoder), essentially all of it the encoder JVP
(2.87 ms forward -> 4.83 ms with JVP; the symbolic `dz/dt` itself is 0.36 ms).
`reg_dzdt=0` costs the same as `reg_dzdt=1e-3` -- multiplying by zero saves nothing.

## What this file does not own

The encoder (its own port), `Scaling` (separate; `symder.py` stays unit-agnostic),
and the loss (`utils.py`).

## Verification

- **Selection matrix reproduces slicing**: `C = I[:n_obs]` matches `z[..., :n_obs]`.
- **Commutation**: linear `Observation` applied inside vs outside `dfunc` agree;
  a callable observation differs (so it must go inside).
- **Pinning**: with all observables direct, pinned values equal the measurements and
  the encoder output is unused; with `pin_measured=False` nothing is pinned; with a
  mix, states land at their declared indices (test with a non-contiguous pinned set).
- **Pinned `dz/dt`** equals `dobs/dt / w` from the data.
- **Shapes**: `(T, n_dims)` and `(batch, T, n_dims)`; batched equals a per-sample loop.
- **Controls**: the control rows of `d2y` are 0 when `du/dt` is constant.
- **No tracer leak** under `eqx.filter_jit(eqx.filter_grad(...))` over the whole model.
- **`dzdt`**: with a linear encoder the JVP value matches the analytic one;
  `dvs=None` skips the JVP.
- **`unobserved_states()`** fires for a state no observable touches.

## Risks

- Summed observables make recovery harder; the zero-column check catches only the
  degenerate case, not weak observability.
- Unpinned states have nothing anchoring their scale or sign, so anchoring
  (mass balances, positivity) matters more here than in the reference.
- The linear/nonlinear split in where `Observation` is applied is a real branch and
  needs a test on each side.
- Pinning propagates measurement noise directly into the latent state; that is what
  `pin_measured=False` is for.

---

## Appendix: two ways to use the measurement *values*

The loss compares only *derivatives* of observables, so for a summed observable the
measured **value** is never used. Nothing forces `C z = v`. Measured with an
untrained encoder on `COD = S_S + S_I + 1.6*X_S`, `X_tot = X_H + X_A`:

```
C @ z  = [[-75.6, 66.1], ...]
v      = [[100.0, 30.0], ...]     <- nothing forces these to agree
```

The only thing tying the latent's level to reality is the indirect requirement that
the resulting dynamics reproduce the observed derivatives. Pinning fixes this for
single-state rows; summed rows are left free. Two ways to close the gap.

### Alternative 1: algebraic elimination (hard constraint)

Per row, solve for one state instead of letting the encoder predict it. With `dep`
the eliminated states, `free` the rest, `A = C[:, dep]`, `B = C[:, free]`:

```
z_dep    = A^-1 (v    - B z_free    - C_u u)
dz_dep/dt = A^-1 (dv/dt - B dz_free/dt - C_u du/dt)
```

- The measurement holds **exactly** at every sample; verified `C @ z == v`.
- The encoder produces `n_states - rank(C)` states instead of `n_states`: a
  structural reduction in degrees of freedom (5 -> 3 in the example above).
- **It subsumes pinning**: a single-state row makes `A` 1x1 and recovers today's
  `z = obs / w`, so this replaces the pinned/free machinery rather than adding to it.
- **It resolves the deferred control-offset case**: `u` is known, so a row mixing a
  state and a control eliminates just as easily. No need to thread `u` into pinning.
- Cost is slightly favourable: the encoder's last layer and its JVP narrow, against
  two small matmuls with a precomputed `A^-1`.

Risks:

- **Measurement noise lands entirely on the eliminated state**, scaled by
  `COD / (w * z)`. Measured on `COD = S_S + S_I + 1.6*X_S` with `S_S = 8`
  (small) and `X_S = 40` (dominant):

  | noise on COD | eliminate `S_S` | eliminate `X_S` |
  |---|---|---|
  | 2 % | 25.4 % relative, P(negative) 0.005 % | 3.2 %, 0 % |
  | 5 % | 64.2 %, P(negative) 5.9 % | 8.0 %, 0 % |
  | 10 % | 127.1 %, P(negative) 21.7 % | 15.9 %, 0 % |

  **Always eliminate the dominant term.** This matters more than `cond(A)`, which
  was 1.0 for both choices above and so blind to the difference.
- A negative eliminated state is unphysical and, inside a Monod denominator
  `K_S + S`, flips its sign -- the failure mode `monod_herbert_limitations.md`
  records, where three sign-flipped samples took the derivative error from 16 % to
  220 %.
- Positivity cannot be enforced on eliminated states (the free ones can still use a
  positive parameterisation, though `softplus` cannot represent an exact zero, and
  Monod-Herbert starts at `x0 = [0, 15, 0]`).
- Requires `rank(C) = n_obs`; redundant observables leave no invertible submatrix.
- Measurement bias lands directly in the states; the encoder cannot absorb it.

### Alternative 2: an MSE penalty on `||C z - v||` (soft constraint)

Add a loss term comparing predicted observables to measured ones.

| | elimination (hard) | MSE penalty (soft) |
|---|---|---|
| uses the measurement value | yes, exactly | yes, weighted |
| degrees of freedom | `n_states - rank(C)` | unchanged (informational only) |
| positivity | impossible for eliminated states | enforceable on all states |
| per-sample noise | injected 1:1 into the state | encoder can average it over its window |
| needs `rank(C) = n_obs` | yes | no |
| choice of eliminated state | required, and consequential | none |
| measurement uncertainty | implicitly zero | a weight per sensor |
| new hyperparameter | no | yes, a weight |

The last row is the principled argument: elimination asserts **infinite confidence**
in every measurement, while a weighted penalty states COD to +-5 % and DO to +-1 %.
Because the encoder sees a time window (delay embedding / conv receptive field), it
can smooth noise across samples; an algebraic per-sample substitution cannot.

### Recommendation

**Start with the penalty.** It is one extra term in `utils.py`, where the loss
already lives, so `symder.py` does not change, and it is reversible. There is
precedent in the same file: `reg_dzdt` is exactly this pattern, a soft coupling
between encoder and data.

Then, once the residual `||C z - v||` can be seen to sit within measurement noise,
consider elimination for rows where it is safe -- a dominant eliminated component and
a well-conditioned `A`. A hybrid is legitimate: eliminate the safe rows, penalise
the rest.

Watch the weight early in training, when the encoder is arbitrary and `||C z - v||`
is large (the `-75.6` vs `100.0` above). Normalising each observable by its scale, or
warming the weight up, will matter more than its final value.
