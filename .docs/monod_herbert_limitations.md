# Conditioning limits on the Monod-Herbert identification

Why `examples/monod_herbert.py` still does not recover the true model, and the
numerical-linear-algebra background needed to size a SINDy library against the
data you actually have.

Two earlier findings have been fixed and are no longer covered here: the
swapped `select` criterion labels, and `degree=3` being too low to express the
oxygen equation's `s*o*o*u` term (now `degree=4`). A third — `exclude` vectors
of length `n_variables` rather than `n_variables + 1` — was intentional: the
entries drop biomass-times-aeration terms, which is what they do.

All figures below were re-measured against the **current** library:

```python
{"degree": 4, "interactions_degree": 2,
 "var_degree": (2, 2, 2, 1), "var_interactions_degree": (2, 2, 2, 0),
 "bias": True, "exclude": [(0, True, 0, 1), (True, True, 0, 1), (0, True, True, 1)]}
```

with `dt = 0.01`, `t_span = (0, 10)`, `x0 = [0, 15, 0]`, `noise = 0`,
`kLa(t) = clip(6(t-1), 0, 7)` — giving **80 library columns** and 240 candidate
regressions.

> **This document corrects an earlier version.** That version reported the
> problem as two independent causes: a rank deficiency (rank 9 of 80) and a
> threshold-versus-column-scale mismatch. Re-measurement shows they are not
> independent — the scale mismatch *causes* the rank diagnosis. It also claimed
> the true support was "not identifiable"; that was a float32 artifact, and the
> true coefficients are recovered exactly once columns are normalised. The
> corrected analysis follows.

---

## 1. The root cause: column scale spans 5.06e7

The library columns differ enormously in magnitude, because they are monomials
of states that already differ by ~11x in spread (S ~ 66, X ~ 17, O ~ 2.8) and
degree-4 monomials raise that disparity to the fourth power:

```
        column      ||col||
             1    3.164e+01
             o    1.124e+02
             u    2.008e+02
       s*s*s*u    9.813e+07
       s*s*s*x    9.924e+07
       s*s*s*s    1.600e+09

ratio largest/smallest: 5.06e+07
```

This single fact causes three separate failures.

### 1a. It disables the sparsity threshold

STLSQ prunes on raw coefficient magnitude, `|coef| >= threshold`, with one
threshold for every column. But a least-squares fit balances `coef * ||col||`,
so a term sitting on a large column earns a small coefficient regardless of its
importance. For the true `dx/dt` relation:

```
      term      coeff     ||col||  coeff*||col||
      dx_k          4   2.657e+02      1.063e+03
         x      14.48   6.886e+02      9.972e+03
    o*dx_k         20   6.548e+02      1.310e+04
       x*o       72.4   1.223e+03      8.855e+04
    s*dx_k        0.2   1.417e+04      2.834e+03
       s*x      0.724   3.285e+04      2.378e+04
  s*o*dx_k          1   5.213e+04      5.213e+04
     s*x*o      -2.38   7.098e+04      1.689e+05
```

The true coefficients span 0.2 to 72.4 — a factor of 360 — purely from column
scale, not because some terms matter 360x more. `threshold = 0.001` is low
enough to prune nothing, so thresholding is effectively off and STLSQ degenerates
to plain least squares. Raising it to prune meaningfully would remove `s*dx_k`
(coefficient 0.2) before `x*o` (coefficient 72.4) — on scale, not relevance.

### 1b. It inflates the condition number past what float32 can carry

```
  ds/dt    rank 10/80   cond 2.14e+15
  dx/dt    rank  9/80   cond 1.78e+15
  do/dt    rank  9/80   cond 4.55e+15
```

Normalising each column to unit norm — which changes only the parameterisation,
not the subspace being fitted — collapses the condition number by seven orders
of magnitude:

```
setup                                   raw rank   raw cond  norm rank  norm cond
ramp, 1 IC (the example)                       9    1.8e+15         20    3.7e+08
ramp, 6 ICs                                    8    2.9e+13         30    1.2e+08
two-tone, 1 IC                                12    9.5e+14         51    5.9e+07
two-tone, 6 ICs                                8    2.2e+13         51    3.6e+07
```

### 1c. It corrupts the numerical-rank diagnostic itself

Numerical rank counts singular values above a tolerance that is **relative to
the largest**: `sigma_i > sigma_max * max(m, n) * eps`. One enormous column
drags `sigma_max` up and buries every other direction beneath the bar:

```
largest column norm  : 1.600e+09  (s*s*s*s)
sigma_max            : 1.605e+09
rank tolerance       : 1.916e+05
```

So "rank 9 of 80" was never a statement about the information in the data — it
was a statement about `s*s*s*s` being a billion times larger than the constant
column. With unit-norm columns the same data gives rank 20, and with a two-tone
control, rank 51.

---

## 2. The true support IS identifiable — the earlier claim was wrong

The previous version of this document restricted the fit to exactly the seven
true `dx/dt` columns and reported coefficients off by a factor of two at a
residual of 2.4e-04, concluding that the data could not distinguish them. That
conclusion does not survive scrutiny.

Same data, same seven columns, float32, two parameterisations of the *same*
subspace:

```
      term      true    raw cols  unit-norm cols
      dx_k         4       1.951               4
         x     14.48       7.058           14.48
    o*dx_k        20       16.37              20
       x*o      72.4       73.97            72.4
    s*dx_k       0.2      0.2206             0.2
       s*x     0.724      0.7989           0.724
     s*x*o     -2.38       -2.35           -2.38

  worst coefficient error, raw       : 51.3%
  worst coefficient error, normalised:  0.0%
  cond(A) raw 5.39e+04   cond(A) normalised 1.22e+03
```

Normalisation alone recovers every coefficient exactly. Confirmed independently
in float64, where the raw columns also recover exactly:

```
  1 IC ramp      cond 5.39e+04   worst coefficient error 0.0%
  6 ICs ramp     cond 2.33e+03   worst coefficient error 0.0%
  6 ICs 2-tone   cond 1.45e+03   worst coefficient error 0.0%
```

**The data identifies the true relation.** What failed was solving for it in
float32 on columns spanning eight orders of magnitude.

---

## 3. What still limits the fit

Correcting the above leaves two real constraints.

**Rank 51 of 80 is still a deficit.** Even with unit-norm columns and a two-tone
control, 29 directions remain indistinguishable. A single trajectory of a
3-state ODE traces a one-dimensional curve, and polynomial features along a
curve are inherently near-dependent. Adding samples along the same curve does
not help — 6006 samples give no more rank than 1001. Adding *directions* does:
richer excitation took rank from 20 to 51 on identical sample counts.

The trajectory also settles, so a third of the samples carry no dynamical
information:

```
samples with |dx/dt| > 10% of its peak: 454/1001 (45.4%)
samples with |dx/dt| >  1% of its peak: 688/1001 (68.7%)
first fully-settled sample: index 688 (t = 6.88 d)
```

At steady state every derivative-bearing column goes to zero at once.

**float32 remains tight.** At `eps = 1.2e-7`, a normalised condition number of
3.7e8 still leaves `kappa * eps` above 1 — no guaranteed correct digits by the
standard bound. The oracle fit succeeds because the true 7-column submatrix
conditions to 1.2e3; the full 80-column sweep does not have that luxury.

---

## 4. The maths behind the two numbers

Both diagnostics come from the singular values of `Theta` (shape 1001 x 80).

**Condition number** `kappa(Theta) = sigma_max / sigma_min`. The working rule:
solving a least-squares problem loses roughly `log10(kappa)` decimal digits. At
`kappa ~ 1e15` in float32 — which offers about 7 digits — nothing survives. This
is also why solvers matter: normal-equation methods (forming `Theta^T Theta`)
scale with `kappa^2`, while QR- and SVD-based solvers scale with `kappa`. It is
the reason the Gram-matrix fast path was left as opt-in in `sindy_utils`.

**Numerical rank**: count `sigma_i > sigma_max * max(m, n) * eps`. This is the
default tolerance in `numpy.linalg.matrix_rank`. It is a *numerical* judgment,
not an algebraic one — `Theta` is almost surely full-rank algebraically, but
singular values below that bar are indistinguishable from zero at working
precision. **The tolerance is relative to `sigma_max`**, which is exactly why
§1c happens: rescale one column and the reported rank moves.

The singular-value spectrum tells you more than either scalar. For the `dx/dt`
equation, raw columns:

```
   i   sigma_i/sigma_0
   0        1.000e+00
   1        6.451e-02
   2        3.451e-02
   3        1.704e-02
   4        2.419e-03
   5        1.968e-03
   8        2.116e-04   <- numerical rank cutoff
   9        6.462e-05
  10        5.287e-05
  79        5.613e-16
```

A well-posed problem shows a sharp cliff between signal and noise directions.
This decays smoothly, which is the signature of an ill-posed problem rather than
a cleanly rank-deficient one — the distinction Hansen's book (below) is built
around.

---

## 5. References

None of the analysis above was taken from the literature — it is direct
computation using standard numerical linear algebra. These are the references
for the underlying theory, not sources that were consulted.

**Numerical linear algebra**

- **Trefethen & Bau, _Numerical Linear Algebra_** (SIAM, 1997). The best starting
  point. Lectures 4-5 build the SVD; Lectures 12-15 cover conditioning and
  stability, including the digits-lost rule.
- **Golub & Van Loan, _Matrix Computations_** (4th ed., Johns Hopkins, 2013). The
  standard reference for rank-deficient least squares and numerical rank via SVD.
- **Higham, _Accuracy and Stability of Numerical Algorithms_** (2nd ed., SIAM,
  2002). Rigorous floating-point error analysis, including the `kappa` vs
  `kappa^2` distinction between solver families.
- **Hansen, _Rank-Deficient and Discrete Ill-Posed Problems_** (SIAM, 1998). The
  most directly applicable: what to do once you know you are rank-deficient —
  truncated SVD, the Picard condition, the L-curve. Also the source of the
  "cleanly rank-deficient vs ill-posed" distinction in §4.

**SINDy-specific**

- **Zhang & Schaeffer (2019)**, _On the convergence of the SINDy algorithm_,
  Multiscale Model. Simul. 17(3):948-972, [arXiv:1805.06445](https://arxiv.org/abs/1805.06445).
  Conditions on `Theta` under which STLSQ provably recovers the true support —
  the closest thing to a theoretical answer to "how big a library can this data
  support".
- **Champion, Zheng, Aravkin, Brunton & Kutz (2020)**, _A unified sparse
  optimization framework to learn parsimonious physics-informed models from
  data_, IEEE Access 8:169259-169271, [arXiv:1906.10612](https://arxiv.org/pdf/1906.10612).
  SR3 — replaces hard thresholding with a relaxed formulation that handles
  badly-scaled libraries far better than one global threshold (directly relevant
  to §1a).
- **Messenger & Bortz (2021)**, _Weak SINDy: Galerkin-based data-driven model
  selection_, Multiscale Model. Simul. 19(3):1474-1497. The weak form improves
  conditioning substantially and never differentiates the data.
- **Kaheman, Kutz & Brunton (2020)**, _SINDy-PI_, Proc. R. Soc. A 476:20200279.
  The method this repo implements; its case for the parallel formulation over
  null-space implicit-SINDy is a conditioning argument.

**Experiment design — how much data, and how rich**

- **Ljung, _System Identification: Theory for the User_** (2nd ed., Prentice
  Hall, 1999). Chapter 13 on experiment design and *persistency of excitation of
  order n* — the formal version of "the input must be rich enough to identify n
  parameters". The ramp control here is low-order, which is why the two-tone
  signal bought rank.
- **Willems, Rapisarda, Markovsky & De Moor (2005)**, _A note on persistency of
  excitation_, Systems & Control Letters 54(4):325-329. The fundamental lemma,
  tying data richness to a Hankel-matrix rank condition.
- **Foucart & Rauhut, _A Mathematical Introduction to Compressive Sensing_**
  (Birkhäuser, 2013), for `m >~ s*log(N/s)` sample bounds.

  **Caveat**: those bounds assume *incoherent* measurement matrices. Trajectory
  data is the opposite — maximally coherent, confined to a low-dimensional
  manifold. This is why 6006 pooled samples bought no rank over 1001. Do not
  size a library from a sample count.

---

## 6. A recipe for sizing a library against your data

1. **Normalise columns before anything else.** Everything downstream — the
   threshold, the condition number, the rank diagnostic — is meaningless until
   the columns are comparable. This is one line and it recovered the true
   coefficients exactly.
2. **Compute the spectrum, not just the rank.** `jnp.linalg.svd(theta,
   compute_uv=False)` and plot `log10(sigma_i / sigma_0)`. Look for the cliff. A
   smooth decay means ill-posed, and no rank number will summarise it honestly.
3. **Treat numerical rank `r` as a ceiling, and stay well under it.** It is an
   upper bound on how many columns the data can distinguish, not a target.
4. **To raise `r`, add directions — not samples.** Different regions of state
   space, higher-order excitation, horizons that stay in the transient. Measured
   here: two-tone control took rank 20 to 51; sixfold more samples took it
   nowhere.
5. **Check `kappa * eps` against your precision.** In float32
   (`eps = 1.2e-7`), a condition number above ~1e7 means no guaranteed correct
   digits. Either enable float64 (`jax.config.update("jax_enable_x64", True)`)
   or shrink the library until it conditions.
6. **Prefer structure over degree.** A library built from known reaction
   stoichiometry will always beat a full degree-4 sweep, because it buys the
   terms you need without the 80-column conditioning bill.

---

## Summary

| # | finding | status |
|---|---|---|
| 1 | column scale spans 5.06e7, disabling the threshold, inflating `kappa`, and corrupting the rank diagnostic | root cause — fix with column normalisation |
| 2 | the true support is fully identifiable; the earlier "not identifiable" claim was a float32 artifact | corrected |
| 3 | rank 51/80 even normalised and well-excited; a third of samples are at steady state | real, needs richer excitation |
| 4 | float32 leaves no margin at `kappa ~ 3.7e8` | real, consider float64 |

In order of expected effect: **normalise the columns**, then **enrich the
excitation** (multi-tone control, wider initial conditions, horizons inside the
transient), then **shrink the library** using known reaction structure, and only
then consider float64.

No code was changed in producing this report.
