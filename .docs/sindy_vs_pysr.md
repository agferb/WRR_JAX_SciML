# SINDy vs PySR for equation discovery in this repo

Comparison of two equation-discovery methods against the repo's goals: train a
UDE for the unknown terms, extract sparse, interpretable ODEs, identify hidden
variables with autoencoders and Takens delay embeddings, stay JAX-compatible,
and handle Monod-Herbert-like systems (rational and polynomial terms) with up to
10 variables, up to 3 of them hidden.

The two candidates are SINDy, as implemented in `src/sindy.py` with possible
extensions, and PySR ([Cranmer 2023](https://arxiv.org/abs/2305.01582)).

These are insights, not a verdict. Sources are literature, documentation and
this repo's own Monod-Herbert measurements. PySR was **not** run: the install
succeeded, but its Julia download failed mid-transfer. Every PySR statement
below comes from literature or documentation.

## 1. Integration with the objectives (UDE, hidden variables, JAX)

### SINDy: strong fit

- **Hidden variables are the deciding case.** The autoencoder approaches (Champion
  et al. 2019; Bakarji et al. 2023,
  [deep delay autoencoders](https://arxiv.org/abs/2201.05136)) train the encoder
  and a sparse SINDy model *jointly*. The SINDy loss pushes the latent coordinates
  toward coordinates whose dynamics are sparse. That matters because Takens'
  theorem only guarantees the embedding up to an unknown diffeomorphism, so a
  search run after training cannot shape the coordinates.
- **Already native here.** SINDy runs in JAX in this repo (`jit`, `vmap`), so it
  can sit inside that training loop.
- **Missing piece.** A differentiable sparse regression: STLSQ uses
  `lax.while_loop`, which is not reverse-differentiable. Options are a fixed mask
  per training phase, or an L1 penalty as in Champion et al.
- **Precedent.** The UDE-then-SINDy pipeline is the standard one
  ([Rackauckas et al. 2020](https://arxiv.org/abs/2001.04385)).

### PySR: fits only as an offline stage

- **Not JAX.** It is a Python front-end driving a Julia engine
  (SymbolicRegression.jl), and it cannot live inside a gradient loop.
- **Results can be handed back.** `output_jax_format` returns JAX callables for
  the discovered expressions, so constants can be re-fitted through Diffrax.
- **Hidden variables need an alternating pipeline.** Train the autoencoder and
  latent dynamics, run PySR, freeze, refit. Without a sparsity signal during
  training, the latent coordinates may only admit complicated expressions.
- **ODEs are not native.** The PySR paper's comparison table marks it "×" for
  differential equations. Like SINDy, it regresses on derivative targets.

## 2. Ease of use and modification

### SINDy

- **Fully owned.** About 500 lines of JAX. The library spec, masks,
  normalisation, SINDy-PI sweep and diagnostics can all be changed directly.
- **Library design is manual.** For rational terms it is hard, which is where
  the complexity of `exclude`, `var_degree` and `var_interactions_degree` comes
  from.

### PySR

- **Easy to start.** A scikit-learn-style `PySRRegressor` gives a full Pareto
  front of accuracy against complexity. `model_selection="best"` picks the
  highest score among expressions within 1.5x of the most accurate loss.
- **Customisable via Julia snippets.** Losses (`elementwise_loss`,
  `loss_function`), operators, `constraints`, `nested_constraints` and
  `complexity_of_operators` are configured this way. Changing the search
  algorithm itself means changing Julia code.
- **Structure can be imposed.** `TemplateExpressionSpec` fixes an outer form and
  lets the search fill in sub-expressions; this is how you would encode "a sum of
  Monod-type terms". It replaced `ParametricExpressionSpec` in PySR 2.0.
- **Practical friction.** It needs a Julia runtime (downloaded on first import),
  and searches take minutes to hours per output. ODEFormer's authors report
  "minutes for all other methods except SINDy"
  ([d'Ascoli et al., ICLR 2024](https://arxiv.org/abs/2310.05573)).
- **Multiple outputs.** `y` may be 2-D, but each output is a separate search.

## 3. Extra code needed to reach the goal

### SINDy: more methodology to build

- **Joint autoencoder training:** a differentiable sparse regression.
- **Noise robustness:** weak-form SINDy (Messenger & Bortz 2021, *Multiscale
  Model. Simul.* 19(3):1474-1497) or ensembles
  ([E-SINDy, Fasel et al. 2022](https://arxiv.org/abs/2111.10992)). `vmap` makes
  ensembles cheap.
- **Cost:** the QR-reduced STLSQ (`.docs/qr_reduction_plan.md`).
- **Hardest:** keeping rational libraries tractable at 10 variables (section 5).

### PySR: mostly glue

- **Tasks:** export data from the UDE, write templates, re-import to JAX.
- **Cost:** the pipeline splits across two runtimes, and joint training for
  hidden variables is simply not available.

## 4. Sparse models from noisy data

### Benchmark evidence favours PySR, with caveats

- **ODEBench** (63 ODEs, 1-4 dimensions; d'Ascoli et al.): PySR was the strongest
  classical baseline, "only occasionally outperformed" by ODEFormer on clean
  data, with ODEFormer's lead growing with noise. SINDy variants ranked low. All
  non-transformer baselines used finite-difference derivatives with optional
  Savitzky-Golay smoothing, with hyperparameters searched per trajectory.
- **Brum et al. 2025** ([arXiv:2508.20257](https://arxiv.org/abs/2508.20257)):
  PySR was "the least affected by noise in data, with almost negligible
  differences in R²". PySINDy recovered all nine systems' structure, but less
  accurately.
- **Caveats.** Nearly all benchmark systems are polynomial (Lotka-Volterra,
  Lorenz, SIR variants): SINDy's home turf, not the rational case here. The
  noise-oriented SINDy variants (weak form, E-SINDy) were not the ones tested.

### In the UDE pipeline, noise moves upstream

The regression targets are the network's output evaluated on the fitted
trajectory, so they are smooth and consistent with the states. Regression error
becomes systematic approximation error rather than white noise.

Measured in this repo on Monod-Herbert:

- **Exact data.** The true implicit relation is an exact null vector of the 37
  admitted columns: float64 rank 36, with the smallest singular value 1e-6 to
  1e-7 below the next.
- **Inconsistent state and derivative data.** The null direction is no longer
  isolated: smallest over next singular value is 0.2-0.8, rank is full, and both
  conditioning warnings go quiet.

So expect SINDy-PI's candidate selection to be fragile whenever the network's
function is not exactly rational.

## 5. The deciding issue: rational terms at 10 variables

SINDy-PI multiplies out denominators, so its library degree grows with the
number of **distinct** denominators in an equation.

| case | library needed | columns per equation |
|---|---|---|
| Monod-Herbert (1 denominator `(S+K_S)(O+K_O)`, 3 states + 1 control) | `degree=4`, `interactions_degree=2` | 80 union, 37 admitted |
| 10 variables, same degrees, no priors | C(14,4) + C(12,2) | ~1067 |
| 10 variables, two distinct 2-factor denominators | denominator degree 4, numerator ~6 | ~C(16,6) ≈ 8000 |

- **The sweep grows with the columns.** SINDy-PI runs one regression per library
  column per equation, so the last row means roughly 80,000 candidate regressions
  per iteration for 10 equations.
- **The data can't support it.** A single 3-state trajectory here already
  supported only rank 20 of 37 admitted columns, so libraries of this size are
  unidentifiable, not just slow.
- **PySR scales with expression size instead.** Monod-Herbert's dS/dt is already
  about 20-25 nodes against the default `maxsize=30`, and genetic search gets
  hard as expressions grow, so templates are what make large rational models
  feasible.

To stay with SINDy, the library needs structure priors:

- per-equation libraries built from known stoichiometry (partly supported via
  per-equation specs and `exclude`);
- libraries of Monod-type terms `x / (K + x)` over a grid of `K` values, which
  keeps the regression linear in the unknown coefficients;
- or a bilevel fit of the `K` values (e.g. optimistix) around a linear sparse
  regression.

## Summary

| criterion | SINDy (repo, extended) | PySR |
|---|---|---|
| UDE + hidden variables + JAX | native; joint training possible (needs differentiable regression) | offline only; alternating pipeline; JAX export for refits |
| ease of use / modification | fully owned; library design manual and hard | easy API; deep changes need Julia; separate runtime |
| extra code | substantial (differentiable STLSQ, weak form / ensembles, rational scaling) | mostly glue + templates; pipeline split in two |
| sparse under noise | SINDy-PI fragile; weak / ensemble variants help; strong on polynomial systems | best classical baseline in benchmarks (mostly polynomial ODEs); slow |
| rational terms at 10 variables | library size explodes with distinct denominators | scales with expression size; templates help |

A hybrid is worth considering:

- **SINDy in JAX inside training:** latent coordinates, fast screening,
  polynomial terms.
- **PySR offline for the final rational forms:** seeded with structure from SINDy
  results or known stoichiometry, then exported to JAX and refitted through
  Diffrax.

## Experiment that would settle it (not run)

Run PySR (binary operators only, `maxsize` about 35, with and without a Monod
`TemplateExpressionSpec`) and the repo's SINDy-PI on the same Monod-Herbert data
in three versions:

1. clean derivatives;
2. derivatives evaluated at noisy states (the example's current noise model);
3. noisy states with finite-difference derivatives.

Compare structure recovery, held-out derivative error and runtime. Needs a
working Julia install.

## Sources

- Cranmer 2023, PySR paper: https://arxiv.org/abs/2305.01582
- PySR API reference: https://ai.damtp.cam.ac.uk/pysr/api/
- PySR source (`sr.py` docstrings): https://github.com/MilesCranmer/PySR
- PySR discussion on `TemplateExpressionSpec`: https://github.com/MilesCranmer/PySR/discussions/787
- d'Ascoli et al. 2024, ODEFormer / ODEBench: https://arxiv.org/abs/2310.05573
- Brum et al. 2025, symbolic regression in dynamical systems: https://arxiv.org/abs/2508.20257
- Bakarji et al. 2023, deep delay autoencoders: https://arxiv.org/abs/2201.05136
- Fasel et al. 2022, Ensemble-SINDy: https://arxiv.org/abs/2111.10992
- Messenger & Bortz 2021, Weak SINDy, *Multiscale Model. Simul.* 19(3):1474-1497
- Rackauckas et al. 2020, Universal Differential Equations: https://arxiv.org/abs/2001.04385
- Champion et al. 2019, data-driven discovery of coordinates and governing equations, *PNAS* 116(45):22445-22451

---

## Appendix: hybrid pipeline sketch

Division of labour: JAX owns everything that needs gradients or thousands of
cheap linear fits. PySR runs once, offline, on small, well-posed subproblems that
the JAX stages prepare for it.

```
measured y(t) ──► delay embedding ──► [1] joint training (JAX)
                                          encoder/decoder + latent UDE + sparsity regulariser
                                                   │  frozen model, closure g(z) on the fitted latents
                                                   ▼
                                      [2] screening (JAX, SINDy + linear algebra)
                                          polynomial terms, variable support, process count, stoichiometry
                                                   │  small subproblems: rate_j(few inputs)
                                                   ▼
                                      [3] rate laws (PySR, offline)
                                          Pareto fronts per rate, ~3-4 inputs, + - * / only
                                                   │  candidate expressions → JAX callables
                                                   ▼
                                      [4] trajectory refit + selection (JAX, Diffrax)
                                                   │  accepted terms become "known" mechanistic parts
                                                   └──► back to [1] for what's still unexplained
```

### Stage 1: joint training (JAX only)

Pieces: `src/models.py` `Autoencoder`, a UDE vector field like `src/ude.py`, and
`train.fit`.

- **Embedding and latent state.** Delay-embed the ~7 measured channels, then
  encode them into a 10-dimensional latent state `z` that includes the ~3 hidden
  variables.
- **Latent dynamics.** `dz/dt = f_known(z, u) + g_θ(z, u)`, where `g_θ` is the
  network closure.
- **Losses:**
  - reconstruction;
  - consistency between the encoder's time derivative and the latent vector
    field;
  - trajectory error through Diffrax with `multiple_shooting_windows`;
  - a sparsity regulariser, which is where SINDy enters.

Two choices for the regulariser:

- **Differentiable SINDy on `g_θ`** (as in Bakarji et al.): fit a small polynomial
  library to the closure outputs with a fixed mask or L1, and penalise the
  residual. It pushes the latent coordinates toward ones with sparse dynamics,
  which is SINDy's key advantage for hidden variables. The risk is that a
  polynomial regulariser biases the coordinates toward polynomial-looking
  dynamics, working against real rational terms.
- **Group sparsity on the closure's Jacobian `∂g_i/∂z_k`:** makes each closure
  output depend on few variables without assuming any functional form. That
  dependency sparsity is exactly what PySR needs later, and it does not fight
  rational terms.

Reasonable start: the Jacobian regulariser, adding the SINDy residual penalty
only if the latent coordinates come out poorly identified.

**Anchor the hidden variables.** Latent coordinates are only defined up to a
change of coordinates. To make a hidden variable mean "biomass", anchor it with
known mechanistic terms, mass balances or positivity. Any later symbolic result
depends on these coordinates.

### Stage 2: screening (JAX: SINDy plus linear algebra)

Freeze stage 1 and evaluate `g_θ` on the fitted latent trajectories only; the
network cannot be trusted outside the data it was trained on. This stage is
cheap and produces four outputs:

1. **Polynomial equations settle here.** Run explicit SINDy per equation with a
   small library. When the fit is good and the support stays stable across an
   ensemble (E-SINDy style: `vmap` over bootstrap resamples gives inclusion
   probabilities), accept it; PySR never sees that equation. Use `conditioning()`
   to confirm the library is identifiable on this data.
2. **Variable support per equation:** which `z_k` each closure output depends on.
   Combine SINDy inclusion probabilities with the closure's Jacobian magnitudes.
   - Polynomial support alone is a heuristic: a polynomial fit of `s/(K+s)` still
     "uses" `s`, and collinear variables can sneak in.
   - Be generous: a variable dropped here cannot be recovered by PySR later.
3. **Number of processes.** Stack the closure outputs into `G` (samples ×
   equations) and take its SVD.
   - In bioprocess models each equation is a stoichiometric combination of a few
     shared rates, `G ≈ R νᵀ`. For Monod-Herbert, growth and decay are shared
     across S, X and O.
   - The numerical rank of `G` estimates the number of processes.
4. **Stoichiometry and rates.** Factor `G` into a rate matrix `R` and a
   stoichiometric matrix `ν`.
   - The split is not unique (any rotation of `R` works), so it needs
     constraints: known yields, non-negative rates, or a known zero pattern in `ν`.
   - It turns "10 coupled equations" into "a few independent rate laws, each
     depending on few variables".

### Stage 3: rate laws with PySR (offline)

**Inputs and targets.** One search per rate `r_j`, using only the variables
selected in stage 2 (3-4 instead of 10) and the rate column of `R` as target. This
is the biggest lever for genetic search.

**Settings:**

- binary operators `+ - * /` only;
- `nested_constraints` to stop division from nesting inside division;
- `maxsize` around 30;
- `weights` to down-weight regions where the closure is unreliable;
- multiple seeds, keeping forms that recur (a credibility check similar to
  E-SINDy's inclusion probabilities).

**Size gain.** Monod-Herbert's growth rate `μ s x o / ((s+K_S)(o+K_O))` is about
15 nodes on 3 inputs. The full `dS/dt` that SINDy-PI had to express with 37
columns in float64 is about 20-25 nodes on 4 inputs.

**Priors.** A `TemplateExpressionSpec` can fix an outer form, for example a product
of switching factors times a mass-action term. Not verified: the exact template
syntax in PySR 2.4, and whether it accepts starting expressions (e.g. forms found
by SINDy). Check both before relying on them.

**Output.** A Pareto front per rate, exported as JAX callables via
`output_jax_format` or through SymPy. Keep the top few candidates per rate rather
than only PySR's "best".

### Stage 4: refit and select on trajectories (JAX)

**Build the models.** Combine accepted SINDy terms, candidate rate laws and `ν`
into ODE right-hand sides, with every constant as a trainable Equinox parameter.

**Fit trajectories, not derivatives.** Refit constants against the measured data,
through the decoder for hidden states, using `ude.solve`,
`multiple_shooting_windows` and `train.fit`. Stages 2-3 regressed on network
outputs, so their constants inherit the closure's bias; this refit removes it.

**Selection:**

- **Candidates.** With k candidates per rate and p rates, there are kᵖ models.
  `vmap` over them is cheap in JAX, and selection uses held-out trajectory error
  (other initial conditions or controls).
- **Residual network.** Optionally keep a small, heavily penalised one. If it
  grows during refitting, something is still unexplained.

### Stage 5: peel off and repeat

Accepted terms move into `f_known`, the pattern `freeze_mechanistic` already
encodes. Stage 1 retrains a smaller closure for whatever is left. Each round the
unknown part shrinks, so later screening and PySR searches get easier.

### What each method contributes

| | SINDy's role | PySR's role |
|---|---|---|
| hidden variables | shapes latent coordinates during training (optional regulariser) | none: sees only frozen coordinates |
| polynomial terms | discovers and accepts them directly | not used |
| rational terms | screening: variable support, not the final form | final functional form on small subproblems |
| scale (10 variables) | small per-equation libraries only, no rational sweep | 3-4 inputs per rate instead of 10 outputs × 10 inputs |
| noise | handled upstream by the UDE and the trajectory refit | same |
| JAX | everything except stage 3 | offline; only arrays in, expressions out |

This avoids both failure modes from the comparison: no SINDy-PI library blow-up
from multiple denominators, and no PySR search over large expressions with 10
inputs.

### Costs and risks

- **More moving parts.** Two runtimes, several thresholds (inclusion probability,
  "polynomial fit is good enough", rank cutoff) and more hyperparameters.
- **Screening errors propagate.** A wrongly dropped variable or a badly rotated
  rate matrix gives PySR an unsolvable problem. Staying generous and constraining
  with stoichiometry mitigates this.
- **Latent coordinates.** If the hidden variables are not anchored, PySR may find a
  correct but unphysical rate law for a transformed coordinate.
- **Closure bias.** A network that fits trajectories while learning the wrong
  function (the README's single-shooting warning) poisons stages 2-3. Multiple
  shooting and the stage-4 refit are the safeguards.

### Smallest experiment to try first

Monod-Herbert, no hidden variables, everything measured:

1. Replace the unknown kinetics with a network closure and train the UDE.
2. Take the SVD of the closure outputs: check the rank matches the number of
   processes, and recover `ν` using the yield as a constraint.
3. Run PySR on the growth rate with inputs `s, x, o`.
4. Refit constants on trajectories and compare against SINDy-PI's float64 result,
   on structure, held-out error and runtime.

This tests the core idea (rates plus stoichiometry, then PySR on small problems)
before adding autoencoders.
