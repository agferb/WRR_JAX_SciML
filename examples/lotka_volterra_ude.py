"""End-to-end UDE pipeline: fit a neural closure, then recover it symbolically.

  1. Generate data from the true Lotka-Volterra system.
  2. Fit a UDE in which the linear terms are known and an MLP stands in for the
     unknown interaction terms, using multiple shooting.
  3. Run SINDy on the trained MLP's outputs to recover those terms as symbols.

Run:  python -m examples.lotka_volterra_ude
"""

import argparse

import jax
import jax.numpy as jnp

from src import sindy, train, ude

TRUE = dict(alpha=1.3, beta=0.9, gamma=0.8, delta=1.8)


def main(steps: int, n_points: int, t_end: float, length: int, stride: int, seed: int):
    key = jax.random.key(seed)

    # --- 1. synthetic data ------------------------------------------------
    ts = jnp.linspace(0.0, t_end, n_points)
    y0 = jnp.array([0.44249296, 4.6280594])
    ys = ude.solve(ude.lotka_volterra(**TRUE), y0, ts)
    tw, yw = ude.multiple_shooting_windows(ts, ys, length, stride)
    print(f"{n_points} observations -> {tw.shape[0]} shooting windows of {length}\n")

    # --- 2. fit the UDE ---------------------------------------------------
    model = ude.LotkaVolterraUDE(
        alpha=TRUE["alpha"],
        delta=TRUE["delta"],
        key=key,
        width_size=32,
        depth=2,
        gain=0.1,
    )

    def loss_fn(model, tw, yw):
        def segment(t_seg, y_seg):
            return jnp.mean((ude.solve(model, y_seg[0], t_seg) - y_seg) ** 2)

        return jnp.mean(jax.vmap(segment)(tw, yw))

    # Mechanistic parameters are known here, so train only the neural closure.
    spec = ude.freeze_mechanistic(model)
    print(f"Fitting UDE for {steps} steps:")
    model, history = train.fit(
        model, loss_fn, tw, yw, steps=steps, lr=5e-3, filter_spec=spec, print_every=250
    )
    print(f"\nFinal multiple-shooting MSE: {history[-1]:.3e}")
    print(f"Fitted gain:                 0.1 -> {float(model.gain):.4f}")

    # How close is the learned closure to the term we withheld?
    xz = ys[:, 0] * ys[:, 1]
    learned = jax.vmap(model.closure)(ys)
    true_closure = jnp.stack([-TRUE["beta"] * xz, TRUE["gamma"] * xz], axis=1)
    rel = jnp.abs(learned - true_closure).mean(0) / jnp.abs(true_closure).mean(0)
    print(f"Closure relative error:      {rel[0]:.1%}, {rel[1]:.1%}\n")

    # --- 3. recover the closure symbolically ------------------------------
    theta, names = sindy.polynomial_library(ys, degree=2, var_names=["x", "z"])
    xi = sindy.stlsq(theta, learned, threshold=0.15)

    print("SINDy recovery of the neural closure:")
    print(sindy.format_equations(xi, names, target_names=["closure_x", "closure_z"]))
    print(f"\nGround truth:  dclosure_x/dt = {-TRUE['beta']:+.4f} x z")
    print(f"               dclosure_z/dt = {TRUE['gamma']:+.4f} x z")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--n-points", type=int, default=80)
    p.add_argument("--t-end", type=float, default=5.0)
    p.add_argument("--length", type=int, default=5, help="shooting window length")
    p.add_argument("--stride", type=int, default=2, help="shooting window stride")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    main(a.steps, a.n_points, a.t_end, a.length, a.stride, a.seed)
