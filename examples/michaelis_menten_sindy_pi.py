"""SINDy-PI on Michaelis-Menten kinetics: recovering a rational ODE.

  1. Integrate dx/dt = jx - Vmax*x/(Km + x), a rational right-hand side that no
     explicit polynomial library can represent.
  2. Sweep every library term as the left-hand side (the parallel-implicit step).
  3. Score the candidates on a held-out trajectory and print the rational form.

Run:  python -m examples.michaelis_menten_sindy_pi
"""

import argparse

import jax
import jax.numpy as jnp

from src import sindy, ude

TRUE = dict(jx=0.6, vmax=1.5, km=0.3)


def vector_field(t, y, args):
    return jnp.array([TRUE["jx"] - TRUE["vmax"] * y[0] / (TRUE["km"] + y[0])])


def trajectory(y0: float, n_points: int, t_end: float, method: str = "exact"):
    """A trajectory and its derivatives, either exact or by finite differences."""
    ts = jnp.linspace(0.0, t_end, n_points)
    ys = ude.solve(vector_field, jnp.array([y0]), ts)
    if method == "exact":
        return ys, jax.vmap(lambda y: vector_field(0.0, y, None))(ys)
    if method == "finite-diff":
        return ys, jnp.gradient(ys[:, 0], ts)[:, None]
    else:
        raise ValueError("`method` must be either 'exact' or 'finite-diff")


def main(n_points: int, t_end: float, y0: float, threshold: float):
    ys, dys = trajectory(y0, n_points, t_end, method="exact")
    held_out = trajectory(0.85 * y0, 173, t_end, method="exact")

    lib = {
        "degree": 3,
        "interactions_degree": 1,
    }
    model = sindy.SINDy(n_states=1, library=lib, var_names=["x"], implicit=True)
    model.solve(ys, dys, threshold=threshold)

    print(f"Library swept ({len(model.feature_names)} candidates per equation):")
    print(f"  {model.library_terms()}")

    print("\nBest candidates on held-out data:")
    print(model.candidates(*held_out, top=4))

    selected = int(model.select(*held_out)[0])
    _, deriv_fit, _ = model.scores(*held_out)
    print(f"\nSelected LHS term: {model.feature_names_for(0)[selected]}")
    print(model.equations())
    km, jx, vmax = TRUE["km"], TRUE["jx"], TRUE["vmax"]
    print(
        f"Ground truth:      dx/dt = -({-jx * km:+.4f} {vmax - jx:+.4f} x)"
        f" / ({km:+.4f} {1.0:+.4f} x)"
    )
    print(f"Held-out derivative error: {float(deriv_fit[0, selected]):.2e}")

    # Padding the library costs uniqueness -- see the README.
    lib_loose = {
        "degree": 4,
        "interactions_degree": 2,
    }
    loose = sindy.SINDy(n_states=1, library=lib_loose, var_names=["x"], implicit=True)
    loose.solve(ys, dys, threshold=threshold)
    loose_selected = int(loose.select(*held_out)[0])
    _, loose_deriv_fit, _ = loose.scores(*held_out)
    print("\nSame data, library padded by one degree (degree=4, interactions=2):")
    print(f"  {loose.equations()}")
    print(
        f"  error {float(loose_deriv_fit[0, loose_selected]):.2e} -- small, but"
        " the form carries spurious quadratic terms."
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n-points", type=int, default=200)
    p.add_argument("--t-end", type=float, default=12.0)
    p.add_argument("--y0", type=float, default=3.0)
    p.add_argument("--threshold", type=float, default=0.05)
    a = p.parse_args()
    main(a.n_points, a.t_end, a.y0, a.threshold)
