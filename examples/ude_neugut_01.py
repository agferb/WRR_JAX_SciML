from collections.abc import Callable
import csv
import re

import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

import os
from pathlib import Path
from dotenv import load_dotenv

SEED = 32
IN_DIM = 8
OUT_DIM = 1
N_SAMPLES = 100


class UDE_MLP(eqx.Module):

    layers: list
    activations: tuple = eqx.field(static=True)

    def __init__(
        self,
        sizes,
        activations,
        *,
        key: PRNGKeyArray,
    ):

        depth = len(sizes) - 1
        keys = jax.random.split(key, depth)
        self.layers = [
            eqx.nn.Linear(i, o, key=k) for i, o, k in zip(sizes[:-1], sizes[1:], keys)
        ]
        self.activations = tuple(activations)

    def __call__(self, X: Float[Array, "IN_DIM"], args=None) -> Float[Array, "OUT_DIM"]:

        for layer, act in zip(self.layers, self.activations):
            X = act(layer(X))
        return X

def load_flat_params_csv(path: str, column: str = "value") -> Float[Array, "N"]:
    """Read parameter values from a CSV, keeping file order (or sorting by `[i]` in the name column)."""

    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    idx = [re.search(r"\[(\d+)\]$", r.get("name", "")) for r in rows]
    if all(idx):
        rows = [r for _, r in sorted(zip((int(m[1]) for m in idx), rows), key=lambda t: t[0])]
    return jnp.asarray([float(r[column]) for r in rows], dtype=jnp.float32)


def set_flat_params(model: UDE_MLP, flat: Float[Array, "N"], order: str = "F") -> UDE_MLP:
    """Fill the MLP weights from a flat vector [W1, b1, W2, b2, ...].

    Each W is (out, in). Use order="F" for vectors flattened column-major in
    Julia (Lux / ModelingToolkitNeuralNets), order="C" for row-major.
    """
    flat = jnp.asarray(flat)

    weights, biases, i = [], [], 0
    for layer in model.layers:
        n = layer.weight.size
        weights.append(flat[i : i + n].reshape(layer.weight.shape, order=order))
        i += n
        n = layer.bias.size
        biases.append(flat[i : i + n].reshape(layer.bias.shape))
        i += n

    model = eqx.tree_at(lambda m: [l.weight for l in m.layers], model, weights)
    return eqx.tree_at(lambda m: [l.bias for l in m.layers], model, biases)


# Build MLP layers model
gkey = jax.random.PRNGKey(SEED)
key1, key2, gkey = jax.random.split(gkey, 3)
UDE_denit = UDE_MLP(
    sizes=(IN_DIM, 16, 4, OUT_DIM),
    activations=[jax.nn.tanh, jax.nn.leaky_relu, jax.nn.relu],
    key=key1,
)

# Load parameters
load_dotenv()
params_path = Path(os.environ['PARAMS_DIR']) / "nn_params.csv"
params_arr = load_flat_params_csv(params_path)
UDE_denit_full = set_flat_params(
    model=UDE_denit, flat=params_arr, order='F'
)

# Test with dummies
X = jax.random.uniform(key2, (N_SAMPLES, IN_DIM))
Y = jax.vmap(UDE_denit_full)(X)

pass
