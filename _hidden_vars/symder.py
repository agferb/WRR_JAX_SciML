"""Observation map and the SymDer model, generalising `get_symder_apply`/`get_model_apply`.

Reference (`symder_ref/symder/symder/symder.py`) assumes observables *are*
state slices and a `[visible | hidden]` latent layout. Here observables can be
weighted sums of states (`Observation`), and the latent is `[states |
controls]`, so pinning and the free/pinned split are derived from the
observation matrix instead of assumed from position.
"""

from collections.abc import Callable, Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, PyTree

from _hidden_vars.odeint_zero import dfunc
from _hidden_vars.sym_models import PolynomialLibrary, SymModel

__all__ = ["Observation", "SymDerModel"]


class Observation(eqx.Module):
    """By-name observation map `x_obs = C @ z`, compiled to a matrix at construction.

    `matrix` is a float array field and so *is* picked up by
    `eqx.is_inexact_array` filtering -- training code that partitions a
    `SymDerModel` into trainable/static must exclude it explicitly (e.g. a
    custom filter spec), since it cannot be marked `static=True`: a jnp array
    is unhashable and breaks under `jit`.

    Building e.g.:
        state_names=["S_S", "S_I", "X_S", "X_H", "X_A", "S_O", "S_NH"],
        observables={
            "COD":   {"S_S": 1.0, "S_I": 1.0, "X_S": 1.6},  # weighted sum
            "X_tot": ["X_H", "X_A"],                        # plain sum
            "S_O":   "S_O",                                 # direct measurement
        }                                                   # no S_NH observation
    """

    matrix: Float[Array, "n_obs n_states"]
    names: tuple[str, ...] = eqx.field(static=True)
    state_names: tuple[str, ...] = eqx.field(static=True)

    def __init__(
        self,
        state_names: Sequence[str],
        observables: dict[str, dict[str, float] | Sequence[str] | str],
    ):
        self.state_names = tuple(state_names)
        self.names = tuple(observables.keys())
        index = {name: i for i, name in enumerate(self.state_names)}

        rows = []
        for spec in observables.values():
            row = np.zeros(len(self.state_names))
            if isinstance(spec, dict):
                for name, weight in spec.items():
                    row[index[name]] = weight
            elif isinstance(spec, str):
                row[index[spec]] = 1.0
            else:
                for name in spec:
                    row[index[name]] = 1.0
            rows.append(row)
        self.matrix = jnp.asarray(np.stack(rows))

    def __call__(self, z: Float[Array, "... n_states"]) -> Float[Array, "... n_obs"]:
        """x_obs = C @ z, batch axes carried through."""
        return jnp.einsum("...s,os->...o", z, self.matrix)

    def unobserved_states(self) -> tuple[str, ...]:
        """State names with an all-zero column in `matrix`: touched by no observable."""
        zero = np.all(np.asarray(self.matrix) == 0, axis=0)
        return tuple(name for name, is_zero in zip(self.state_names, zero) if is_zero)

    def _pinned(self) -> list[tuple[int, int, float]]:
        """Rows with exactly one nonzero weight: `(obs_idx, state_idx, weight)`.

        Such a row measures its state directly (`z[s] = obs / w`), so that
        state can be pinned from data instead of inferred by the encoder.
        """
        matrix = np.asarray(self.matrix)
        pinned = []
        for obs_idx, row in enumerate(matrix):
            nonzero = np.flatnonzero(row)
            if nonzero.size == 1:
                pinned.append((obs_idx, int(nonzero[0]), float(row[nonzero[0]])))
        return pinned


class SymDerModel(eqx.Module):
    """Encoder -> latent assembly -> symbolic derivative stack.

    `encoder` is a plain callable `v -> free_states`, mapping the observed
    series to the free (non-pinned) states only -- last axis width
    `n_states - n_pinned` -- and carrying its own parameters as an Equinox
    submodule; this file does not define it.
    """

    encoder: Callable
    sym_model: PolynomialLibrary | SymModel
    observation: Observation
    n_states: int = eqx.field(static=True)
    n_controls: int = eqx.field(static=True)
    pin_measured: bool = eqx.field(static=True)
    pinned_idx: tuple[int, ...] = eqx.field(static=True)
    pinned_obs_idx: tuple[int, ...] = eqx.field(static=True)
    pinned_weight: tuple[float, ...] = eqx.field(static=True)
    free_idx: tuple[int, ...] = eqx.field(static=True)

    def __init__(
        self,
        encoder: Callable,
        sym_model: PolynomialLibrary | SymModel,
        observation: Observation,
        n_states: int,
        n_controls: int = 0,
        pin_measured: bool = True,
    ):
        self.encoder = encoder
        self.sym_model = sym_model
        self.observation = observation
        self.n_states = n_states
        self.n_controls = n_controls
        self.pin_measured = pin_measured

        pinned = (
            observation._pinned()
            if pin_measured and isinstance(observation, Observation)
            else []
        )
        self.pinned_obs_idx = tuple(o for o, _, _ in pinned)
        self.pinned_idx = tuple(s for _, s, _ in pinned)
        self.pinned_weight = tuple(w for _, _, w in pinned)
        self.free_idx = tuple(i for i in range(n_states) if i not in self.pinned_idx)

    def latent(
        self,
        v: Float[Array, "... n_obs"],
        dvdt: Float[Array, "... n_obs"] | None = None,
    ) -> tuple[Float[Array, "... n_states"], Float[Array, "... n_states"] | None]:
        """Assemble the full state from pinned measurements + encoder output;
        `dzdt` only when `dvdt` is given.

        Pinned states take `z = obs / w` and `dz/dt = dobs/dt / w` straight
        from the data; the encoder JVP (run only when `dvdt is not None`)
        supplies the free states and their derivative instead. Both groups
        are scattered into their declared indices, not assumed contiguous.
        """
        free = self.encoder(v)
        dfree = None if dvdt is None else jax.jvp(self.encoder, (v,), (dvdt,))[1]

        batch = v.shape[:-1]
        free_idx = jnp.asarray(self.free_idx, dtype=jnp.int32)
        z = jnp.zeros(batch + (self.n_states,), dtype=v.dtype).at[..., free_idx].set(free)
        if self.pinned_idx:
            obs_idx = jnp.asarray(self.pinned_obs_idx, dtype=jnp.int32)
            pinned_idx = jnp.asarray(self.pinned_idx, dtype=jnp.int32)
            weight = jnp.asarray(self.pinned_weight, dtype=v.dtype)
            z = z.at[..., pinned_idx].set(v[..., obs_idx] / weight)

        if dfree is None:
            return z, None

        dzdt = jnp.zeros(batch + (self.n_states,), dtype=v.dtype).at[..., free_idx].set(dfree)
        if self.pinned_idx:
            dzdt = dzdt.at[..., pinned_idx].set(dvdt[..., obs_idx] / weight)
        return z, dzdt

    def _field(
        self,
        z: Float[Array, "... n_states"],
        dudt: Float[Array, "... n_controls"],
    ) -> tuple[Callable, PyTree]:
        """
        Build `field(y, t, p) = concatenate([library(y), dudt], -1)` with `p` explicit, never closed over.
        """
        trainable, static = eqx.partition(self.sym_model, eqx.is_inexact_array)
        dudt = jnp.broadcast_to(dudt, z.shape[:-1] + (self.n_controls,))

        def field(y: Float[Array, "... n_dims"], t: Float[Array, ""], p: PyTree):
            return jnp.concatenate([eqx.combine(p, static)(y), dudt], axis=-1)

        return field, trainable

    def field_derivatives(
        self,
        z: Float[Array, "... n_states"],
        u: Float[Array, "... n_controls"],
        dudt: Float[Array, "... n_controls"],
        t: Float[Array, ""] | None = None,
    ) -> list[Float[Array, "... n_dims"]]:
        """
        [dy/dt, d^2y/dt^2] of the augmented latent `y = [z | u]` (`n_dims = n_states + n_controls`).
        """
        t = jnp.asarray(0.0) if t is None else t
        field, trainable = self._field(z, dudt)
        y0 = jnp.concatenate([z, u], axis=-1)
        return [f(y0, t, trainable) for f in dfunc(field, 2)[1:]]

    def obs_derivatives(
        self,
        z: Float[Array, "... n_states"],
        u: Float[Array, "... n_controls"],
        dudt: Float[Array, "... n_controls"],
        t: Float[Array, ""] | None = None,
    ) -> Float[Array, "... n_obs 2"]:
        """
        Stack [dx_obs/dt, d^2x_obs/dt^2]. Only valid for linear observations.
        """
        t = jnp.asarray(0.0) if t is None else t
        raw_derivs = self.field_derivatives(z, u, dudt, t)
        outs = [self.observation(dy[..., : self.n_states]) for dy in raw_derivs]
        return jnp.stack(outs, axis=-1)

    def __call__(
        self,
        v: Float[Array, "... n_obs"],
        dvdt: Float[Array, "... n_obs"] | None = None,
        u: Float[Array, "... n_controls"] | None = None,
        dudt: Float[Array, "... n_controls"] | None = None,
        t: Float[Array, ""] | None = None,
    ) -> tuple[
        Float[Array, "... n_obs 2"],
        Float[Array, "... n_states"],
        Float[Array, "... n_states"] | None,
    ]:
        """Encode `v` to the latent, then predict [dx_obs/dt, d^2x_obs/dt^2] as `(sym_deriv, z, dzdt)`."""
        batch = v.shape[:-1]
        u = jnp.zeros(batch + (self.n_controls,)) if u is None else u
        dudt = jnp.zeros(batch + (self.n_controls,)) if dudt is None else dudt
        z, dzdt = self.latent(v, dvdt)
        sym_deriv = self.obs_derivatives(z, u, dudt, t)
        return sym_deriv, z, dzdt



if __name__ == '__main__':

    state_names = ["S_S", "S_I", "X_S", "X_H", "X_A", "S_O", "S_NH"]
    observables = {
        "COD":   {"S_S": 1.0, "S_I": 1.0, "X_S": 1.6},   # weighted sum
        "X_tot": ["X_H", "X_A"],                          # plain sum, weights 1
        "S_O":   "S_O",                                   # direct measurement
        "X_S":   {"X_S": 2.0},
    }

    obs_matrix = Observation(
        state_names=state_names,
        observables=observables
    )

    z = jnp.ones((10,7)) * (jnp.arange(7) + 1)
    matrix = obs_matrix.matrix 
    obss = obs_matrix(z)
    unob_states = obs_matrix.unobserved_states()
    pinned_states = obs_matrix._pinned()

    pass

