"""Optax training loop for Equinox models.

Note the `eqx.filter_*` calls: they are what let a model containing non-array
leaves (activation functions) pass through `jit` and `grad` untouched.
"""

from collections.abc import Callable

import equinox as eqx
import optax


def fit(
    model: eqx.Module,
    loss_fn: Callable,
    *args,
    steps: int = 1000,
    lr: float = 1e-3,
    print_every: int = 100,
    filter_spec=eqx.is_inexact_array,
) -> tuple[eqx.Module, list[float]]:
    """Fit `model` by minimising `loss_fn(model, *args)`.

    `filter_spec` selects the trainable leaves -- pass a PyTree of bools (e.g.
    from `ude.freeze_mechanistic`) to train only part of the model.
    """
    
    trainable_model, static_model = eqx.partition(model, filter_spec)
    optim = optax.adam(lr)
    opt_state = optim.init(trainable_model)

    @eqx.filter_jit
    def step(trainable_model, opt_state, *args):
        
        @eqx.filter_value_and_grad
        def _loss(tm):
            return loss_fn(eqx.combine(tm, static_model), *args)

        loss, grads = _loss(trainable_model)
        updates, opt_state = optim.update(grads, opt_state, trainable_model)
        trainable_model = eqx.apply_updates(trainable_model, updates)
        return trainable_model, opt_state, loss

    history = []
    for i in range(steps):
        trainable_model, opt_state, loss = step(trainable_model, opt_state, *args)
        history.append(float(loss))
        if print_every and (i % print_every == 0 or i == steps - 1):
            print(f"  step {i:5d}   loss {float(loss):.6e}")
    return eqx.combine(trainable_model, static_model), history
