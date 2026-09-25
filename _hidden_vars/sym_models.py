"""Polynomial symbolic library and the term-summing wrapper for SymDer models.

`PolynomialLibrary` gives the same nested library control as `src/sindy.py`
(`degree`, `var_degree`, `exclude`, `bias`) over a flat, gather-based feature
vector; `SymModel` sums an arbitrary list of such terms into one vector field.
"""

import itertools
from collections.abc import Sequence

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Bool, Float, Int

from src.sindy_utils import _normalise_spec

__all__ = ["PolynomialLibrary", "SymModel"]


def _allows(spec: dict, combo: tuple[int, ...], n_dims: int, deriv: int = 0) -> bool:
    """Whether `spec` admits `combo`, mirroring `SINDy._allows`'s explicit branch."""
    exponents = tuple(combo.count(i) for i in range(n_dims)) + (deriv,)
    if exponents in spec["exclude"]:
        return False
    total = sum(exponents[:-1])
    if total > spec["degree"]:
        return False
    if total == 0:
        return spec["bias"]
    cap = spec["var_degree"]
    # a short `var_degree` leaves trailing variables uncapped (`zip` truncation),
    # inherited from `src/sindy.py` deliberately -- not guarded
    return cap is None or all(e <= c for e, c in zip(exponents[:-1], cap))


def _combos(n_dims: int, max_degree: int) -> list[tuple[int, ...]]:
    """`[()]` then every combination with replacement, for degree 1..max_degree."""
    combos = [()]
    for d in range(1, max_degree + 1):
        combos += itertools.combinations_with_replacement(range(n_dims), d)
    return combos


def _default_var_names(n_out: int, n_dims: int) -> list[str]:
    """`x0..x{n_dims-1}` then `u0..`, matching `src/sindy.py`'s convention."""
    return [f"x{i}" for i in range(n_out)] + [f"u{j}" for j in range(n_dims - n_out)]


class PolynomialLibrary(eqx.Module):
    """Flat polynomial library: dz/dt terms as a gather over sentinel-padded monomials.

    `idxm` holds, per feature, the variable indices multiplied together, right-
    padded with the sentinel `n_dims`, which points at a constant `1.0` appended
    to the input; the empty combo (the bias) is all sentinels, so it evaluates
    to 1 with no special case. `mask` (bool) and `idxm` (int) are skipped by
    `eqx.is_inexact_array` filtering, so only `w` trains. `idxm` dtype must be
    integer.
    """

    coeffs: Float[Array, "n_dims n_features"]
    mask: Bool[Array, "n_dims n_features"]
    idxm: Int[Array, "n_features max_degree"]
    n_dims: int = eqx.field(static=True)

    def __init__(
        self,
        n_states: int,
        spec: dict | Sequence[dict],
        n_controls: int = 0,
    ):
        n_dims = n_states + n_controls
        self.n_dims = n_dims
        specs = [spec] * n_states if isinstance(spec, dict) else list(spec)
        normalised = [_normalise_spec(s, n_dims) for s in specs]

        max_degree = max(s["degree"] for s in normalised)
        combos = _combos(n_dims, max_degree)

        self.mask = jnp.array(
            [[_allows(s, c, n_dims) for c in combos] for s in normalised]
        )
        self.idxm = jnp.array(
            [c + (n_dims,) * (max_degree - len(c)) for c in combos], dtype=jnp.int32
        )
        self.coeffs = jnp.zeros((n_states, len(combos)))

    def __call__(
        self, z: Float[Array, "... n_dims"], t: Float[Array, ""] | None = None
    ) -> Float[Array, "... n_out"]:
        z_sentinel = jnp.concatenate([z, jnp.ones(z.shape[:-1] + (1,))], axis=-1)
        theta = jnp.prod(z_sentinel[..., self.idxm], axis=-1)
        return jnp.einsum("...f,of->...o", theta, self.mask * self.coeffs)

    def feature_names(self, var_names: list[str] | None = None) -> list[str]:
        """One name per column, e.g. "1", "x0", "x1*x2", recovered from `idxm`."""
        n_states, n_dims = self.coeffs.shape[0], self.n_dims
        names = var_names or _default_var_names(n_states, n_dims)
        return [
            "*".join(names[i] for i in row if i != n_dims) or "1"
            for row in np.asarray(self.idxm).tolist()
        ]

    def library_terms(self, var_names: list[str] | None = None) -> dict[str, list[str]]:
        """Admitted term names per equation, keyed `d{name}`, mirroring `SINDy.library_terms()`."""
        n_states = self.coeffs.shape[0]
        names = var_names or _default_var_names(n_states, self.n_dims)
        features = self.feature_names(var_names)
        mask = np.asarray(self.mask)
        return {
            f"d{names[i]}": [n for n, ok in zip(features, mask[i]) if ok]
            for i in range(n_states)
        }

    def n_terms(self) -> Int[Array, " n_out"]:
        """Admitted term count per equation."""
        return self.mask.sum(axis=1)


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
        self, z: Float[Array, "... n_dims"], t: Float[Array, ""] | None = None
    ) -> Float[Array, "... n_out"]:
        return sum(module(z) for module in self.module_list)


class Encoder(eqx.Module):
    """
    Builds encoder based on provided list of layers.
    """

    layers: list

    def __init__(self, layers: list):
        self.layers = layers

    def __call__(
        self,
        x: Float[Array, "... n_in_dims"],
        t: Float[Array, ""] | None = None,
    ) -> Float[Array, "... n_inferred"]:

        for layer in self.layers:
            x = layer(x)
        return x


if __name__ == "__main__":

    spec = {
        "degree": 3,
        "var_degree": (2, 1, 3),
        "exclude": [(3, 0, 0), (0, 1, 2)],
        "bias": True,
    }

    lib = PolynomialLibrary(
        n_states=2,
        spec=spec,
        n_controls=1,
    )

    coeffs = lib.coeffs
    mask = lib.mask
    idxm = lib.idxm

    z = jnp.arange(30).reshape(10, 3)
    out = lib(z)

    pass
