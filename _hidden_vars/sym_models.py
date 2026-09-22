"""Symbolic-library terms for the SymDer framework: `Linear`, `Quadratic`, `Cubic`, `SymModel`.

Ported from `symder_ref/symder/symder/sym_models.py`, dropping everything but
the polynomial terms and the term-summing wrapper. Three departures from the
reference: `SymModel` has no `dt` factor and no `time_dependence` (derivative
data is already a true dz/dt, and controls enter through the input vector, not
`t`); weights are zero-initialised with no `init` or `key` argument; and every
term takes `n_out`, letting it emit fewer rows than the input width.

`n_out < n_dims` means a term cannot serve as the vector field on its own --
`odeint_zero` needs f(y) to match y's width, so the caller concatenates the
supplied control derivative: F(y) = [f_learned(y) | du/dt].
"""

import jax.numpy as jnp
import numpy as np
import equinox as eqx
from jaxtyping import Array, Bool, Float

__all__ = ["Linear", "Quadratic", "Cubic", "SymModel"]


class Linear(eqx.Module):
    """out[..., i] = sum_j w[i, j] * z_j + b[i].

    `n_out` narrower than `n_dims` for the same reason as `Quadratic`/`Cubic`.
    `b` is the library's constant term, not a bias in the neural-net sense.
    """

    w: Float[Array, "n_out n_dims"]
    b: Float[Array, " n_out"]

    def __init__(self, n_dims: int, n_out: int | None = None):
        n_out = n_dims if n_out is None else n_out
        self.w = jnp.zeros((n_out, n_dims))
        self.b = jnp.zeros((n_out,))

    def __call__(
        self, z: Float[Array, " n_dims"], t: Float[Array, ""] | None = None
    ) -> Float[Array, " n_out"]:
        return jnp.einsum("...j,ij->...i", z, self.w) + self.b


class Quadratic(eqx.Module):
    """out[..., k] = sum_ij w[k, i, j] * z_i * z_j, masked to j >= i.

    The mask keeps the upper-triangular half of each output row's (i, j) plane
    so symmetric monomials (z_i*z_j == z_j*z_i) are counted once. `n_out`
    lets the output be narrower than the input `n_dims`, for a latent vector
    `[visible | hidden | controls]` whose control tail gets no equation.
    """

    w: Float[Array, "n_out n_dims n_dims"]
    mask: Bool[Array, "n_dims n_dims"]

    def __init__(self, n_dims: int, n_out: int | None = None):
        n_out = n_dims if n_out is None else n_out
        ind = np.arange(n_dims)
        mesh = np.stack(np.meshgrid(ind, ind), -1)
        self.mask = jnp.array(mesh[..., 0] >= mesh[..., 1])
        self.w = jnp.zeros((n_out, n_dims, n_dims))

    def __call__(
        self, z: Float[Array, " n_dims"], t: Float[Array, ""] | None = None
    ) -> Float[Array, " n_out"]:
        weights = self.mask * self.w
        return (weights * z[..., None, None, :] * z[..., None, :, None]).sum((-2, -1))


class Cubic(eqx.Module):
    """out[..., l] = sum_ijk w[l, i, j, k] * z_i * z_j * z_k, masked to j >= i >= k."""

    w: Float[Array, "n_out n_dims n_dims n_dims"]
    mask: Bool[Array, "n_dims n_dims n_dims"]

    def __init__(self, n_dims: int, n_out: int | None = None):
        n_out = n_dims if n_out is None else n_out
        ind = np.arange(n_dims)
        mesh = np.stack(np.meshgrid(ind, ind, ind), -1)
        self.mask = jnp.array(mesh[..., 0] >= mesh[..., 1]) * jnp.array(
            mesh[..., 1] >= mesh[..., 2]
        )
        self.w = jnp.zeros((n_out, n_dims, n_dims, n_dims))

    def __call__(
        self, z: Float[Array, " n_dims"], t: Float[Array, ""] | None = None
    ) -> Float[Array, " n_out"]:
        weights = self.mask * self.w
        return (
            weights
            * z[..., None, None, None, :]
            * z[..., None, None, :, None]
            * z[..., None, :, None, None]
        ).sum((-3, -2, -1))


class SymModel(eqx.Module):
    """dz/dt = sum_k module_k(z), each module a symbolic term over the latent vector.

    No `dt` scaling and no `time_dependence`: derivative data already arrives
    as a true per-unit-time rate, and any control dependence enters through
    `z`'s own columns rather than through `t`. `__call__` still takes `t` (and
    ignores it) so `dfunc` in `odeint_zero.py` can call `func(y0, t, *args)`.
    """

    module_list: tuple

    def __init__(self, module_list):
        self.module_list = tuple(module_list)

    def __call__(
        self, z: Float[Array, " n_dims"], t: Float[Array, ""] | None = None
    ) -> Float[Array, " n_out"]:
        return sum(module(z) for module in self.module_list)
