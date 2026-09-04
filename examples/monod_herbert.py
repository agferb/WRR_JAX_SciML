"""SINDy-PI on Monod-Herbert model: recovering a rational ODE.

  1. Integrate the model, a rational right-hand side that no
     explicit polynomial library can represent.
  2. Sweep every library term as the left-hand side (the parallel-implicit step).
  3. Score the candidates on a held-out trajectory and print the rational form.

Run:  python -m examples.michaelis_menten_sindy_pi
"""

import argparse
from ast import Tuple
from typing import Callable

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from src import sindy, ude

TRUE = dict(
    D=3.0,  # d-1
    S_in=100,  # gCOD/m3.d
    DOsat=10,  # mgDO/m3
    Y=0.67,  # gCOD_X / gCOD_S
    mu_max=6.0,  # d-1
    K_S=20,  # gCOD/m3
    K_O=0.2,  # gDO/m3
    b=0.62,  # d-1
)


def vector_field(t, x, args):

    kLa = args(t)  # gDO/m3.d
    growth = (
        TRUE["mu_max"]
        * x[0]
        * x[1]
        * x[2]
        / (x[0] + TRUE["K_S"])
        / (x[2] + TRUE["K_O"])
    )

    dx0 = -growth / TRUE["Y"] + TRUE["D"] * (TRUE["S_in"] - x[0])
    dx1 = growth - (TRUE["D"] + TRUE["b"]) * x[1]
    dx2 = (1 - 1 / TRUE["Y"]) * growth - TRUE["D"] * x[2] + kLa * (TRUE["DOsat"] - x[2])

    return jnp.stack([dx0, dx1, dx2])


def trajectory(
    x0: Float[Array, " states"],
    t_span: Tuple[float],
    dt: float,
    control: Callable = None,
) -> tuple[Float[Array, "samples vars"], Float[Array, "samples states"]]:
    """A trajectory and its derivatives (from vector field)."""

    ts = jnp.arange(t_span[0], t_span[1] + dt, dt)
    xs = ude.solve(vector_field, x0, ts, args=control)
    us = jax.vmap(control)(ts)[:, None]
    dxs = jax.vmap(vector_field, in_axes=(0, 0, None))(ts, xs, control)

    return jnp.concatenate([xs, us], axis=1), dxs


def main():

    dt = 0.01  # d
    t_span = (0.0, 10.0)  # d
    x0_train = jnp.array([0, 15, 0])  # [gCOD/m3]
    x0_test = jnp.array([25, 40, 0])  # [gCOD/m3]
    noise = 0.05
    threshold = 0.05

    kLa = lambda t: jnp.clip(6 * (t - 1), 0, 7)
    ys_train, dxs_train = trajectory(x0_train, t_span, dt, kLa)
    ys_test, dxs_test = trajectory(x0_test, t_span, dt, kLa)

    lib = {
        "degree": 3,
        "interactions_degree": 2,
        "var_degree": (3, 3, 3, 1),
        "exclude": [()],
    }
    model = sindy.SINDy()
    model.solve(ys_train, dxs_train, threshold=threshold)

    print(f"Library swept ({len(model.feature_names)} candidates per equation):")
    print(f"  {model.library_terms()}")

    print("\nBest candidates on held-out data:")
    print(model.equations(model.candidates(ys_test, dxs_test, top=4)))

    selected = int(model.select(ys_test, dxs_test)[0])
    _, deriv_fit, _ = model.scores(ys_test, dxs_test)
    print("\nSelected equation:")
    print(model.equations())

    # REMAKE FOR NEW SYSTEM
    km, jx, vmax = TRUE["km"], TRUE["jx"], TRUE["vmax"]
    print(
        f"Ground truth:      dx/dt = -({-jx * km:+.4f} {vmax - jx:+.4f} x)"
        f" / ({km:+.4f} {1.0:+.4f} x)"
    )
    print(f"Held-out derivative error: {float(deriv_fit[0, selected]):.2e}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dt", type=float, default=0.01)
    p.add_argument("--t-end", type=float, default=10.0)
    p.add_argument("--x0", type=float, default=3.0)
    p.add_argument("--threshold", type=float, default=0.05)
    a = p.parse_args()
    main(a.n_points, a.t_end, a.y0, a.threshold)
