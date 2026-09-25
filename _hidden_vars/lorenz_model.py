"""
lorenz_model.py
================

SymDer training script for the Lorenz dynamical system.

Purpose
-------
This script is the main entry point for training a SymDer model on the
classic three-variable Lorenz system. It demonstrates the full SymDer
pipeline for an ODE (non-PDE) system with partial observability:

1. Load (or generate) a normalized Lorenz dataset from ``data/lorenz.py``.
2. Build a model consisting of:
   - A 1-D convolutional encoder that maps a *window* of visible measurements
     to a latent state z = [visible variables | hidden variables].
   - A sparse symbolic ODE model f(z) = Linear(z) + Quadratic(z) that
     represents the right-hand side of the latent ODE dz/dt = f(z).
3. Train with AdaBelief gradient descent and periodic L1 sparsification to
   push symbolic model coefficients toward zero.
4. Save the best-loss parameters and the sparse mask to disk.

Partial observability
---------------------
The Lorenz system has three state variables (x, y, z_L).  By default only
x and y are observed (``--visible 0 1``). The encoder reconstructs the
hidden variable z_L from the time history of x and y.

Architecture
------------
Encoder:
  Conv1D(128, kernel=9, VALID) → ReLU → Conv1D(128, kernel=1) → ReLU
  → Conv1D(num_hidden, kernel=1)
  then concatenated with the (trimmed) visible inputs.
  The 9-point kernel spans 9 consecutive time steps; "VALID" padding shrinks
  the sequence by 4 on each side (``pad=4``), which must be trimmed from the
  targets too.

Symbolic model:
  SymModel(1e2*dt, [Linear(n_dims), Quadratic(n_dims)])
  rescaled by the input variable standard deviations.

The model is JIT-compiled via ``jax.jit`` for speed.

Sparsification
--------------
Every ``sparse_interval`` gradient steps, all symbolic parameters whose
absolute value is below ``sparse_thres`` are masked to zero.  The mask is
enforced by the optimizer (see ``init_optimizers`` in utils.py).

CLI Usage
---------
    python lorenz_model.py [--output DIR] [--dataset PATH] [--visible 0 1]

Dependencies
------------
- jax, haiku, optax     : autodiff, neural networks, optimizers
- data.utils            : dataset caching layer
- data.lorenz           : Lorenz ODE integrator and normalizer
- encoder.utils         : append_dzdt, concat_visible wrappers
- symder.sym_models     : SymModel, Quadratic, rescale_z
- symder.symder         : get_symder_apply, get_model_apply
- utils                 : loss_fn, init_optimizers, save_pytree
"""

from collections.abc import Callable

import jax
import jax.numpy as jnp
import jax.random as jrn
import numpy as np

import equinox as eqx
import haiku as hk  # Functional neural network framework (parameter management)
import optax  # Gradient-based optimizers

import os.path
import argparse
from functools import partial  # For binding arguments to functions

from _hidden_vars.scaling import Scaling
from _hidden_vars.sym_models import PolynomialLibrary, SymModel, Encoder
from _hidden_vars.symder import Observation, SymDerModel
from scratch1 import out_dim
from symder_ref.symder.lorenz_model import num_visible
from symder_ref.test import n_dims


if __name__ == "__main__":

    # Reproducible PRNG key sequence
    _key = jrn.key(33)

    # Problem statement
    state_names = ("S_S", "X_S", "X_H", "X_A", "S_O", "S_NH")
    control_names = ("S_I",)
    vars_names = state_names + control_names
    observables = {
        "COD": {"S_S": 1.0, "S_I": 1.0, "X_S": 1.6},  # weighted sum
        "X_tot": ["X_H", "X_A"],  # plain sum
        "S_O": "S_O",  # direct measurement
    }  # no S_NH observation

    observation_matrix = Observation(
        state_names=state_names,
        observables=observables,
        control_names=control_names,
    )

    visible_states = observation_matrix.visible_states()
    lumped_states = observation_matrix.lumped_states()
    hidden_states = observation_matrix.hidden_states()

    # System dimensions
    n_states = len(state_names)  # x
    n_controls = len(control_names)  # u
    n_observables = len(observables)  # y
    n_visible = len(visible_states)  # x_vis
    n_lumped = len(lumped_states)  # x_lum
    n_hidden = len(hidden_states)  # x_hid

    n_in_dims = n_observables + n_controls  # v
    n_dims = n_states + n_controls  # z
    n_inferred = n_lumped + n_hidden

    # Dataset (test case, must be pulled in final version)
    dt = 0.01  # d
    t_span = (0.0, 10.0)  # d
    t_arr = jnp.arange(t_span[0], t_span[1] + dt, dt)
    n_samples = len(t_arr)

    y = jrn.uniform((n_samples, n_observables))
    dydt = jrn.uniform((n_samples, n_observables))
    d2y_dt2 = jrn.uniform((n_samples, n_observables))
    u_data = jrn.uniform((n_samples, n_controls))
    dudt = jrn.uniform((n_samples, n_controls))

    v = jnp.hstack((y, u_data))
    dvdt = jnp.hstack((dydt, dudt))

    # Scaling object
    scale = Scaling.fit(y, dydt, d2y_dt2, u=u_data, dt=dt)

    # Build encoder
    kernel_in_size = 2 * n_states
    padding = kernel_in_size // 2
    n_layers = 3
    layer_sizes = [128, 128]
    *keys, _key = jrn.split(_key, n_layers + 1)

    layers = [
        eqx.nn.Conv1d(
            in_channels=n_in_dims,
            out_channels=layer_sizes[0],
            kernel_size=kernel_in_size,
            padding="VALID",
            key=keys[0],
        ),
        jax.nn.relu,
        eqx.nn.Conv1d(
            in_channels=layer_sizes[0],
            out_channels=layer_sizes[1],
            kernel_size=1,
            padding="VALID",
            key=keys[1],
        ),
        jax.nn.relu,
        eqx.nn.Conv1d(
            in_channels=n_in_dims,
            out_channels=n_inferred,
            kernel_size=1,
            padding="VALID",
            key=keys[2],
        ),
        jax.nn.softplus,  # Guarantees positive output
    ]

    encoder = Encoder(layers)

    # Build symbolic model
    spec = {
        "degree": 3,
        "var_degree": 4 * (2,) + 2 * (3,) + (1,),
        # "exclude": [(3, 0, 0), (0, 1, 2)],
        "bias": True,
    }

    sym_model = PolynomialLibrary(
        n_states=n_states,
        spec=spec,
        n_controls=n_controls,
    )

    # Build full model
    full_model = SymDerModel(
        encoder=encoder,
        sym_model=sym_model,
        observation=observation_matrix,
        n_states=n_states,
        n_controls=n_controls,
    )

    # ------------------------------------------------------------------ #
    # Training hyperparameters                                             #
    # ------------------------------------------------------------------ #
    n_steps = 50000  # Total gradient steps
    sparse_thres = 1e-3  # Mask coefficients below this magnitude
    sparse_interval = 5000  # Sparsify every 5000 steps

    # AdaBelief optimizer with very small eps for stability on sparse problems
    optimizers = {
        "encoder": optax.adabelief(1e-3, eps=1e-16),
        "sym_model": optax.adabelief(1e-3, eps=1e-16),
    }

    # Loss function configuration
    loss_fn_args = {
        "deriv_weight": jnp.array(
            [1.0, 1.0]
        ),  # Equal weight on 1st and 2nd derivatives
        "observables_weight": 1.0,  # To match observables values
        "reg_dzdt": 0,  # dz/dt regularization weight (0 = disabled)
        "reg_l1_sparse": 0,  # L1 sparsity weight (0 = pure MSE loss)
    }

    # ------------------------------------------------------------------ #
    # Training                                                             #
    # ------------------------------------------------------------------ #
    # best_loss, best_params, sparse_mask = train(
    #     n_steps,
    #     model_apply,
    #     params,
    #     scaled_data,
    #     loss_fn_args=loss_fn_args,
    #     data_args={"pad": model_args["pad"]},
    #     optimizers=optimizers,
    #     sparse_thres=sparse_thres,
    #     sparse_interval=sparse_interval,
    #     key_seq=key_seq,
    # )

    # # ------------------------------------------------------------------ #
    # # Save results                                                         #
    # # ------------------------------------------------------------------ #
    # print(f"Saving best model parameters in output folder: {args.output}")
    # save_pytree(
    #     os.path.join(args.output, "best.pt"),
    #     {"params": best_params, "sparse_mask": sparse_mask},
    # )
