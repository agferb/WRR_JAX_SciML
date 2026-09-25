"""Physical-unit rescaling for the loss, not the model.

The library, `Observation` matrix and predicted derivatives are all physical
throughout `symder.py`; `Scaling` converts a prediction and its target into
comparable units at the loss, one division per channel.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

__all__ = ["Scaling"]


class Scaling(eqx.Module):
    """
    Per-channel standard deviations for `dy/dt`, `d^2y/dt^2` and controls.

    Fields are float arrays, so their gradients are stopped inside the methods
    so they become not-trainable.
    """

    value: Float[Array, " n_obs"]
    deriv: Float[Array, "n_obs 2"]
    control: Float[Array, " n_controls"]
    dt: float = eqx.field(static=True)

    @classmethod
    def fit(
        cls,
        y: Float[Array, "... n_obs"],
        dydt: Float[Array, "... n_obs"],
        d2y_dt2: Float[Array, "... n_obs"],
        u: Float[Array, "... n_controls"] | None = None,
        dt: float = 1.0,
    ) -> "Scaling":
        """
        Standard deviation of each channel over all leading (sample) axes.
        Offsets from the mean are preserved.
        """
        axes = tuple(range(y.ndim - 1))

        def std(x):
            s = jnp.std(x, axis=axes)
            return jnp.where(s > 0, s, 1.0)

        value = std(y)
        deriv = jnp.stack([std(dydt), std(d2y_dt2)], axis=-1)
        control = std(u) if u is not None else jnp.zeros((0,))
        return cls(value=value, deriv=deriv, control=control, dt=dt)

    def encode_value(self, y: Float[Array, "... n_obs"]) -> Float[Array, "... n_obs"]:
        """Physical -> unit-scale observable."""
        return y / jax.lax.stop_gradient(self.value)

    def decode_value(self, y: Float[Array, "... n_obs"]) -> Float[Array, "... n_obs"]:
        """Unit-scale -> physical observable."""
        return y * jax.lax.stop_gradient(self.value)

    def encode_derivs(
        self, d: Float[Array, "... n_obs 2"]
    ) -> Float[Array, "... n_obs 2"]:
        """Physical -> unit-scale [dy/dt, d^2y/dt^2]."""
        return d / jax.lax.stop_gradient(self.deriv)

    def decode_derivs(
        self, d: Float[Array, "... n_obs 2"]
    ) -> Float[Array, "... n_obs 2"]:
        """Unit-scale -> physical [dy/dt, d^2y/dt^2]."""
        return d * jax.lax.stop_gradient(self.deriv)

    def encode_control(
        self, u: Float[Array, "... n_controls"]
    ) -> Float[Array, "... n_controls"]:
        """Physical -> unit-scale control."""
        return u / jax.lax.stop_gradient(self.control)

    def decode_control(
        self, u: Float[Array, "... n_controls"]
    ) -> Float[Array, "... n_controls"]:
        """Unit-scale -> physical control."""
        return u * jax.lax.stop_gradient(self.control)
