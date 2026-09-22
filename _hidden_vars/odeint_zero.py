"""Zero-step ODE integrator whose custom JVP yields symbolic time derivatives.

`odeint_zero` is the identity on `y0`; its `custom_jvp` rule substitutes the
ODE right-hand side for the tangent of `t`, so repeated forward-mode
differentiation (via `d_dt`/`dfunc`) produces successive exact time
derivatives `dz/dt, d^2z/dt^2, ...` without ever calling a numerical solver.
"""

from collections.abc import Callable
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PyTree

__all__ = ["odeint_zero", "dfunc", "d_dt"]


@partial(jax.custom_jvp, nondiff_argnums=(0,))
def odeint_zero(func: Callable, y0: PyTree, t: Float[Array, ""], *args: Any) -> PyTree:
    """Identity on `y0`; the ODE dynamics live entirely in the JVP rule below."""
    return y0


@odeint_zero.defjvp
def _odeint_zero_jvp(
    func: Callable, primals: tuple[Any, ...], tangents: tuple[Any, ...]
) -> tuple[PyTree, PyTree]:
    """dy = dy0 + func(y, t, *args) * dt, leafwise over the `y0` pytree.

    Recomputes the primal via `odeint_zero(func, y0, t, *args)` rather than
    reusing `y0` directly. `dfunc` nests this rule by re-differentiating its
    own output, so at order n the recursive call re-enters this same rule
    (n-1) times before `func` is evaluated -- that is what turns repeated
    forward-mode differentiation into successive time derivatives instead of
    the first derivative computed n times over.
    """
    y0, t, *args = primals
    dy0, dt, *_ = tangents

    y = odeint_zero(func, y0, t, *args)
    dydt = func(y, t, *args)
    dy = jax.tree.map(
        lambda dy0_, dydt_: dy0_ + dydt_ * jnp.broadcast_to(dt, dydt_.shape), dy0, dydt
    )
    return y, dy


def dfunc(
    func: Callable, order: int, transform: Callable | None = None
) -> list[Callable]:
    """[z(t), dz/dt, ..., d^order z/dt^order], each callable as (y0, t, *args).

    Element 0 is `odeint_zero(func, ...)`, or composed with `transform` when
    given; each later element applies `d_dt` once more to the previous one.
    """
    func0 = (
        partial(odeint_zero, func)
        if transform is None
        else lambda y0, t, *args: transform(odeint_zero(func, y0, t, *args))
    )

    out = [func0]
    for _ in range(order):
        out.append(d_dt(out[-1]))
    return out


def d_dt(func: Callable) -> Callable:
    """Total time derivative of `g(y0, t, *args)` w.r.t. `t`, via `jax.jvp`."""

    def dfunc_dt(y0: PyTree, t: Float[Array, ""], *args: Any) -> PyTree:
        dy0 = jax.tree.map(jnp.zeros_like, y0)
        dt = jax.tree.map(jnp.ones_like, t)
        dargs = jax.tree.map(jnp.zeros_like, args)
        return jax.jvp(func, (y0, t, *args), (dy0, dt, *dargs))[1]

    return dfunc_dt
