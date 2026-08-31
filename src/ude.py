"""Universal differential equations: mechanistic terms + an embedded neural closure.

The point of this module is the `LotkaVolterraUDE` class below. It holds
mechanistic parameters (`alpha`, `delta`) and a neural network (`net`) as
fields of a *single* PyTree. `eqx.filter_grad` then differentiates through
both at once, and `eqx.tree_at` can freeze either independently -- which is
the whole reason Equinox suits this problem.
"""

from collections.abc import Callable

import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray


def lotka_volterra(
    alpha: float = 1.3, beta: float = 0.9, gamma: float = 0.8, delta: float = 1.8
) -> Callable:
    """Ground-truth vector field, used to generate synthetic data.

    dx/dt =  alpha*x - beta*x*z
    dz/dt =  gamma*x*z - delta*z
    """

    def f(t, y, args):
        x, z = y
        return jnp.stack([alpha * x - beta * x * z, gamma * x * z - delta * z])

    return f


class LotkaVolterraUDE(eqx.Module):
    """Known linear growth/decay, unknown interaction learned by an MLP.

    dx/dt =  alpha*x + gain*net(x, z)[0]
    dz/dt = -delta*z + gain*net(x, z)[1]

    The MLP is standing in for the `-beta*x*z` and `+gamma*x*z` terms, which we
    pretend not to know. After fitting, `sindy.py` recovers them symbolically.
    The 'gain' parameter enables tuning the relevance of the closure to the global
    dynamics.
    """

    alpha: Float[Array, ""]
    delta: Float[Array, ""]
    gain: Float[Array, ""]
    net: eqx.nn.MLP

    def __init__(
        self,
        alpha: float,
        delta: float,
        *,
        key: PRNGKeyArray,
        width_size: int = 32,
        depth: int = 2,
        gain: float = 0.1,
    ):
        self.alpha = jnp.asarray(alpha, dtype=jnp.float32)
        self.delta = jnp.asarray(delta, dtype=jnp.float32)
        self.gain = jnp.asarray(gain, dtype=jnp.float32)
       
        self.net = eqx.nn.MLP(
            2, 2, width_size, depth, activation=jax.nn.softplus, key=key
            )
        
    def closure(self, y: Float[Array, " dim"]) -> Float[Array, " dim"]:
        return self.gain * self.net(y)

    def __call__(self, t, y, args):
        closure = self.closure(y)
        return jnp.stack(
            [
                self.alpha * y[0] + closure[0],
                -self.delta * y[1] + closure[1],
            ]
        )


def solve(
    vector_field: Callable,
    y0: Float[Array, " dim"],
    ts: Float[Array, " time"],
    *,
    rtol: float = 1e-6,
    atol: float = 1e-8,
    max_steps: int = 4096,
) -> Float[Array, "time dim"]:
    """Integrate `vector_field` from `ts[0]` to `ts[-1]`, saving at `ts`."""
    sol = dfx.diffeqsolve(
        dfx.ODETerm(vector_field),
        dfx.Tsit5(),
        t0=ts[0],
        t1=ts[-1],
        dt0=ts[1] - ts[0],
        y0=y0,
        saveat=dfx.SaveAt(ts=ts),
        stepsize_controller=dfx.PIDController(rtol=rtol, atol=atol),
        max_steps=max_steps,
    )
    return sol.ys


def multiple_shooting_windows(
    ts: Float[Array, " time"],
    ys: Float[Array, "time dim"],
    length: int,
    stride: int,
) -> tuple[Float[Array, "win len"], Float[Array, "win len dim"]]:
    """
    Split a trajectory into short overlapping segments, each started from data.
    Single-shooting across a long horizon has a notoriously bad loss landscape.
    Fitting many *short* segments instead keeps every integration easy while
    still covering the full trajectory. The returned arrays are stacked on a
    leading window axis, ready to `jax.vmap`.
    
    Arguments:
    ----------
    ts: input time points
    ys: input timeseries values
    length: length of output timeseries arrays
    stride: # of data points to skip for the beginning of next timeseries array

    """
    starts = jnp.arange(0, len(ts) - length + 1, stride)
    idx = starts[:, None] + jnp.arange(length)[None, :]
    return ts[idx], ys[idx]


def freeze_mechanistic(
    model: LotkaVolterraUDE, *, train_gain: bool = True
) -> LotkaVolterraUDE:
    """Return a filter spec that holds `alpha`/`delta` fixed.

    `train_gain=False` also freezes the gain, which caps how large the closure
    can grow.
    """
    filter_spec = jax.tree_util.tree_map(lambda _: False, model)
    filter_spec = eqx.tree_at(
        lambda m: m.net,
        filter_spec,
        replace=jax.tree_util.tree_map(eqx.is_inexact_array, model.net),
    )
    if train_gain:
        filter_spec = eqx.tree_at(lambda m: m.gain, filter_spec, replace=True)
    return filter_spec
