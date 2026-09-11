"""SINDy-PI on Monod-Herbert model: recovering a rational ODE.

  1. Integrate the model, a rational right-hand side that no
     explicit polynomial library can represent.
  2. Sweep every library term as the left-hand side (the parallel-implicit step).
  3. Score the candidates on a held-out trajectory and print the rational form.

Run:  python -m examples.monod_herbert
"""

from typing import Callable

import jax
import jax.numpy as jnp
from jax.numpy import inf
from jaxtyping import Array, Float, PRNGKeyArray

from src import sindy, ude

T = dict(
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
    growth = T["mu_max"] * x[0] * x[1] * x[2] / (x[0] + T["K_S"]) / (x[2] + T["K_O"])

    dx0 = -growth / T["Y"] + T["D"] * (T["S_in"] - x[0])
    dx1 = growth - (T["D"] + T["b"]) * x[1]
    dx2 = (1 - 1 / T["Y"]) * growth - T["D"] * x[2] + kLa * (T["DOsat"] - x[2])

    return jnp.stack([dx0, dx1, dx2])


def trajectory(
    x0: Float[Array, " states"],
    t_span: tuple[float, float],
    dt: float,
    control: Callable = None,
    noise: float = 0.0,
    key: PRNGKeyArray | None = None,
) -> tuple[Float[Array, "samples vars"], Float[Array, "samples states"]]:
    """
    A trajectory and its derivatives (from vector field).

    `noise` is additive Gaussian measurement noise on the states, as a fraction
    of each channel's own standard deviation. Needs argument `key` as well.
    """
    if noise > 0 and key is None:
        raise ValueError("`noise` needs its own `key`; reusing one correlates splits")

    ts = jnp.arange(t_span[0], t_span[1] + dt, dt)
    xs = ude.solve(vector_field, x0, ts, args=control)
    us = jax.vmap(control)(ts)[:, None]

    # Add Gaussian noise (if specified) and clip to positive values
    if noise > 0:
        sigma = noise * xs.std(axis=0)
        xs = jnp.maximum(xs + sigma * jax.random.normal(key, xs.shape), 0.0)

    dxs = jax.vmap(vector_field, in_axes=(0, 0, None))(ts, xs, control)

    return jnp.concatenate([xs, us], axis=1), dxs


def main():

    dt = 0.01  # d
    t_span = (0.0, 10.0)  # d
    x0_train = jnp.array([0, 15, 0])  # [gCOD/m3]
    x0_test = jnp.array([25, 40, 0])  # [gCOD/m3]
    noise = 0.0  # 0.05
    threshold = 0.001

    kLa = lambda t: jnp.clip(6 * (t - 1), 0, 7)
    train_key, test_key = jax.random.split(jax.random.key(0))
    ys_train, dxs_train = trajectory(x0_train, t_span, dt, kLa, noise, train_key)
    ys_test, dxs_test = trajectory(x0_test, t_span, dt, kLa, noise, test_key)

    n_states = 3
    n_controls = 1
    var_names = ["s", "x", "o", "u"]
    lib = {
        "degree": 4,
        "interactions_degree": 2,
        "var_degree": (2, 2, 2, 1),
        "var_interactions_degree": (2, 2, 2, 0),
        "bias": True,
        "exclude": [
            (inf, True, inf, 1),
            (1, 2, 1, 0),
            (2, 2, 0, 0),
            (2, 0, 2, 0),
            (0, 2, 2, 0),
        ],
    }
    model = sindy.SINDy(
        n_states=n_states,
        library=lib,
        n_controls=n_controls,
        var_names=var_names,
        implicit=True,
    )

    lib_terms = model.library_terms()
    print(f"Library swept ({len(model.feature_names)} candidates per equation):")
    for lib, state in enumerate(lib_terms):
        print(f"\n{state}:  {lib}")

    model.solve(ys_train, dxs_train, threshold=threshold)

    # Selection by best derivative error
    _, deriv_fit, _ = model.scores(ys_test, dxs_test)
    selected = model.select(ys_test, dxs_test, criterion="deriv-error")
    n_top = 4
    selected_errors = [float(deriv_fit[i, int(s)]) for i, s in enumerate(selected)]

    jnp.set_printoptions(precision=2, suppress=False)
    print("\nBest candidates on held-out data:")
    print(model.equations(model.candidates(ys_test, dxs_test, top=n_top)))
    print("\nBest equation:")
    print(model.equations())
    print(f"Held-out derivative error:\n {selected_errors}")


if __name__ == "__main__":
    main()
