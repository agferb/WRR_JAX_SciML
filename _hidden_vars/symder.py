"""Observation map and the SymDer model, generalising `get_symder_apply`/`get_model_apply`.

Reference (`symder_ref/symder/symder/symder.py`) assumes observables *are*
state slices and a `[visible | hidden]` latent layout. Here observables (`y`)
can be weighted sums of states (`x`) or controls (`u`) (`Observation`), and
the augmented latent the symbolic library sees is `z = [x | u]`, so the
visible/lumped/hidden partition and the inferred set are derived from the
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
    """By-name observation map `y = C @ [x | u]`,
    compiled to a matrix at construction.

    `matrix` is frozen by default (`trainable=False`): its weights are
    stoichiometry fixed physics rather than free parameters, and mass
    balance would constrain them anyway, so `__call__` stops the gradient
    through it internally. There should be no control line in the observables
    spec (even though it is measured) to not leak into gradients and loss.

    Building e.g.:
        state_names=["S_S", "X_S", "X_H", "X_A", "S_O", "S_NH"],
        control_names=["S_I"]
        observables={
            "COD":   {"S_S": 1.0, "S_I": 1.0, "X_S": 1.6},  # weighted sum
            "X_tot": ["X_S", "X_H", "X_A"],                 # plain sum
            "S_O":   "S_O",                                 # direct measurement
        }                                                   # no S_NH observation
    """

    matrix: Float[Array, "n_obs n_dims"]
    names: tuple[str, ...] = eqx.field(static=True)
    state_names: tuple[str, ...] = eqx.field(static=True)
    control_names: tuple[str, ...] = eqx.field(static=True)

    def __init__(
        self,
        state_names: Sequence[str],
        observables: dict[str, dict[str, float] | Sequence[str] | str],
        control_names: Sequence[str] = (),
    ):
        self.state_names = tuple(state_names)
        self.control_names = tuple(control_names)
        self.names = tuple(observables.keys())
        names = self.state_names + self.control_names
        index = {name: i for i, name in enumerate(names)}

        rows = []
        for spec in observables.values():
            row = np.zeros(len(names))
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

    def __call__(self, z: Float[Array, "... n_dims"]) -> Float[Array, "... n_obs"]:
        """
        `y = C @ z` for the augmented latent `z = [x | u]`.
        Batch axes carried through.
        """
        matrix = jax.lax.stop_gradient(self.matrix)
        return jnp.einsum("...d,od->...o", z, matrix)

    def visible_states(self) -> tuple[str, ...]:
        """
        State names directly observed in data: they appear alone in at least one row.
        """
        idx = {s for _, s, _ in self._visible()}
        return tuple(name for i, name in enumerate(self.state_names) if i in idx)

    def lumped_states(self) -> tuple[str, ...]:
        """
        State names lumped into at least one observable but not directly measured.
        """
        n_states = len(self.state_names)
        touched = np.any(np.asarray(self.matrix)[:, :n_states] != 0, axis=0)
        visible = set(self.visible_states())
        return tuple(
            name
            for name, is_touched in zip(self.state_names, touched)
            if is_touched and name not in visible
        )

    def hidden_states(self) -> tuple[str, ...]:
        """State names not observed at all through observables."""
        n_states = len(self.state_names)
        zero = np.all(np.asarray(self.matrix)[:, :n_states] == 0, axis=0)
        return tuple(name for name, is_zero in zip(self.state_names, zero) if is_zero)

    def _visible(self) -> list[tuple[int, int, float]]:
        """
        Rows with exactly one nonzero state weight and no nonzero control weight:
        `(obs_idx, state_idx, weight)`.

        Such a row measures its state directly (`x[s] = y / w`), so that state is
        visible and can be read from data instead of inferred by the encoder.
        """
        matrix = np.asarray(self.matrix)
        n_states = len(self.state_names)
        visible = []
        for obs_idx, row in enumerate(matrix):
            nonzero = np.flatnonzero(row[:n_states])
            if nonzero.size == 1 and not np.any(row[n_states:]):
                visible.append((obs_idx, int(nonzero[0]), float(row[nonzero[0]])))
        return visible


class SymDerModel(eqx.Module):
    """Encoder -> latent assembly -> symbolic derivative stack.

    `encoder` is a plain callable `v -> inferred_states`, mapping the
    assembled encoder input `v = [y | u]` (last axis width `n_obs +
    n_controls`) to the inferred (non-visible, when `pin_visible=True`)
    states only -- last axis width `n_states - len(visible_idx)` -- and
    carrying its own parameters as an Equinox submodule; this file does not
    define it.
    """

    encoder: eqx.Module
    sym_model: PolynomialLibrary | SymModel
    observation: Observation
    n_states: int = eqx.field(static=True)
    n_controls: int = eqx.field(static=True)
    pin_visible: bool = eqx.field(static=True)
    visible_idx: tuple[int, ...] = eqx.field(static=True)
    visible_obs_idx: tuple[int, ...] = eqx.field(static=True)
    visible_weight: tuple[float, ...] = eqx.field(static=True)
    inferred_idx: tuple[int, ...] = eqx.field(static=True)

    def __init__(
        self,
        encoder: eqx.Module,
        sym_model: PolynomialLibrary | SymModel,
        observation: Observation,
        n_states: int,
        n_controls: int = 0,
        pin_visible: bool = True,
    ):
        self.encoder = encoder
        self.sym_model = sym_model
        self.observation = observation
        self.n_states = n_states
        self.n_controls = n_controls
        self.pin_visible = pin_visible

        visible = (
            observation._visible()
            if pin_visible and isinstance(observation, Observation)
            else []
        )
        self.visible_obs_idx = tuple(o for o, _, _ in visible)
        self.visible_idx = tuple(s for _, s, _ in visible)
        self.visible_weight = tuple(w for _, _, w in visible)
        self.inferred_idx = tuple(i for i in range(n_states) if i not in self.visible_idx)

    def latent(
        self,
        y: Float[Array, "... n_obs"],
        dydt: Float[Array, "... n_obs"] | None = None,
        u: Float[Array, "... n_controls"] | None = None,
        dudt: Float[Array, "... n_controls"] | None = None,
    ) -> tuple[Float[Array, "... n_states"], Float[Array, "... n_states"] | None]:
        """Assemble the full state from visible measurements + encoder output;
        `dxdt` only when `dydt` is given.

        Builds the encoder input `v = [y | u]` here (and `dvdt = [dydt | dudt]`
        for the JVP) -- the only place either concatenation happens, so
        `__call__` just forwards its own `y, dydt, u, dudt` through. Visible
        states take `x = y_obs / w` and `dx/dt = dy_obs/dt / w` straight from
        the observable block of `y`; `visible_obs_idx` indexes into `y`
        (0..n_obs-1), which stays valid inside `v` only because `y` is
        concatenated first -- do not reorder `[y | u]`. The encoder JVP (run
        only when `dydt is not None`) supplies the inferred states and their
        derivative instead. Both groups are scattered into their declared
        indices, not assumed contiguous.
        """
        batch = y.shape[:-1]
        u = jnp.zeros(batch + (self.n_controls,), dtype=y.dtype) if u is None else u
        v = jnp.concatenate([y, u], axis=-1)

        if dydt is None:
            inferred = self.encoder(v)
        else:
            dudt = (
                jnp.zeros(batch + (self.n_controls,), dtype=y.dtype)
                if dudt is None
                else dudt
            )
            dvdt = jnp.concatenate([dydt, dudt], axis=-1)
            inferred, d_inferred = jax.jvp(self.encoder, (v,), (dvdt,))

        inferred_idx = jnp.asarray(self.inferred_idx, dtype=jnp.int32)
        x = (
            jnp.zeros(batch + (self.n_states,), dtype=y.dtype)
            .at[..., inferred_idx]
            .set(inferred)
        )
        if self.visible_idx:
            obs_idx = jnp.asarray(self.visible_obs_idx, dtype=jnp.int32)
            visible_idx = jnp.asarray(self.visible_idx, dtype=jnp.int32)
            weight = jnp.asarray(self.visible_weight, dtype=y.dtype)
            x = x.at[..., visible_idx].set(y[..., obs_idx] / weight)

        dxdt = None
        if dydt is not None:
            dxdt = (
                jnp.zeros(batch + (self.n_states,), dtype=y.dtype)
                .at[..., inferred_idx]
                .set(d_inferred)
            )
            if self.visible_idx:
                dxdt = dxdt.at[..., visible_idx].set(dydt[..., obs_idx] / weight)

        return x, dxdt


    def _field(
        self,
        x: Float[Array, "... n_states"],
        dudt: Float[Array, "... n_controls"],
    ) -> tuple[Callable, PyTree]:
        """
        Build `field(z, t, p) = concatenate([library(z), dudt], -1)` with `p` explicit, never closed over.
        """
        trainable, static = eqx.partition(self.sym_model, eqx.is_inexact_array)
        dudt = jnp.broadcast_to(dudt, x.shape[:-1] + (self.n_controls,))

        def field(z: Float[Array, "... n_z"], t: Float[Array, ""], p: PyTree):
            return jnp.concatenate([eqx.combine(p, static)(z), dudt], axis=-1)

        return field, trainable

    def field_derivatives(
        self,
        x: Float[Array, "... n_states"],
        u: Float[Array, "... n_controls"],
        dudt: Float[Array, "... n_controls"],
        t: Float[Array, ""] | None = None,
    ) -> list[Float[Array, "... n_z"]]:
        """
        [dz/dt, d^2z/dt^2] of the augmented latent `z = [x | u]` (`n_z = n_states + n_controls`).
        """
        t = jnp.asarray(0.0) if t is None else t
        field, trainable = self._field(x, dudt)
        z0 = jnp.concatenate([x, u], axis=-1)
        return [f(z0, t, trainable) for f in dfunc(field, 2)[1:]]

    def obs_derivatives(
        self,
        x: Float[Array, "... n_states"],
        u: Float[Array, "... n_controls"],
        dudt: Float[Array, "... n_controls"],
        t: Float[Array, ""] | None = None,
    ) -> Float[Array, "... n_obs 2"]:
        """
        Stack [dy/dt, d^2y/dt^2]. Only valid for linear observations.
        """
        t = jnp.asarray(0.0) if t is None else t
        raw_derivs = self.field_derivatives(x, u, dudt, t)
        outs = [self.observation(dz) for dz in raw_derivs]
        return jnp.stack(outs, axis=-1)

    def __call__(
        self,
        y: Float[Array, "... n_obs"],
        dydt: Float[Array, "... n_obs"] | None = None,
        u: Float[Array, "... n_controls"] | None = None,
        dudt: Float[Array, "... n_controls"] | None = None,
        t: Float[Array, ""] | None = None,
    ) -> tuple[
        Float[Array, "... n_obs 2"],
        Float[Array, "... n_states"],
        Float[Array, "... n_states"] | None,
    ]:
        """Encode `v = [y | u]` to the latent, then predict [dy/dt, d^2y/dt^2] as `(sym_deriv, x, dxdt)`."""
        batch = y.shape[:-1]
        u = jnp.zeros(batch + (self.n_controls,)) if u is None else u
        dudt = jnp.zeros(batch + (self.n_controls,)) if dudt is None else dudt
        x, dxdt = self.latent(y, dydt, u, dudt)
        sym_deriv = self.obs_derivatives(x, u, dudt, t)
        return sym_deriv, x, dxdt


if __name__ == "__main__":

    state_names = ["S_S", "X_S", "X_H", "X_A", "S_O", "S_NH"]
    control_names = ["S_I"]
    observables = {
        "COD": {"S_S": 1.0, "S_I": 1.0, "X_S": 1.6},
        "X_tot": ["X_S", "X_H", "X_A"],
        "S_O": "S_O",
    }

    obs_matrix = Observation(
        state_names=state_names, control_names=control_names, observables=observables
    )

    z = jnp.ones((10, 7)) * (jnp.arange(7) + 1)
    matrix = obs_matrix.matrix
    y = obs_matrix(z)
    hidden_states = obs_matrix.hidden_states()
    lumped_states = obs_matrix.lumped_states()
    visible_states = obs_matrix.visible_states()
    _vis = obs_matrix._visible()

    pass
