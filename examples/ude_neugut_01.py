from collections.abc import Callable
import csv
import re

import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray
import pandas as pd

import os
from pathlib import Path
from dotenv import load_dotenv

# Enable float64
jax.config.update("jax_enable_x64", True)

# Global variables
SEED = 32
IN_DIM = 8
OUT_DIM = 1
T0 = pd.Timestamp("2022-07-01 00:00:00", tz="Europe/Zurich")
UDE_INPUTS = ['X_H', 'X_A', 'X_STO', 'S_S', 'S_NOX', 'S_N2', 'S_O2', 'S_ALK']
UDE_OUTPUTS = ['S_ALK', 'S_N2', 'S_NOX', 'S_S', 'X_A', 'X_H', 'X_STO', 'X_SS']
SINDY_INPUTS = ['X_H', 'X_STO', 'S_S', 'S_NOX', 'S_O2', 'S_NH4', 'q']
gkey = jax.random.PRNGKey(SEED)

# Load paths
load_dotenv()
DATA_DIR = Path(os.environ['DATA_DIR']) #/ "nn_params.csv"
DATA_DIR = Path(os.environ['UDE_DIR'])
states_data_dir = DATA_DIR / "reactor_states"
params_path = DATA_DIR / "model_results/parameters/nn_params.csv"


# Load data
def load_ude_states_file(file: str) -> pd.DataFrame:
    """Read single file with values from state variables of the system."""

    df = pd.read_csv(file, header=0)
    df.columns.str.strip()
    df['reactor'] = os.path.basename(file).split(".")[0].split("_", maxsplit=1)[1]

    return df


def load_flat_params_csv(path: str, column: str = "value") -> Float[Array, "N"]:
    """Read parameter values from a CSV, keeping file order (or sorting by `[i]` in the name column)."""

    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    idx = [re.search(r"\[(\d+)\]$", r.get("name", "")) for r in rows]
    if all(idx):
        rows = [r for _, r in sorted(zip((int(m[1]) for m in idx), rows), key=lambda t: t[0])]
    return jnp.asarray([float(r[column]) for r in rows], dtype=jnp.float32)


ude_states_files = sorted(states_data_dir.glob("*.csv"))
ude_states = pd.concat(load_ude_states_file(f) for f in ude_states_files)
ude_states.reset_index(drop=True, inplace=True)
ude_states['t'] = T0 + pd.to_timedelta(ude_states['t'], unit="D").dt.round('s')
ude_states['S_NOX'] = ude_states['S_NOX'].clip(lower=0)

params_arr = load_flat_params_csv(params_path)


# MLP class for the UDE
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

    def set_params(self, flat_params: Float[Array, 'N'], order: str = 'F'):
        """Fill the MLP weights from a flat vector [W1, b1, W2, b2, ...].

        Each W is (out, in). Use order="F" for vectors flattened column-major in
        Julia (Lux / ModelingToolkitNeuralNets), order="C" for row-major.
        """
        flat = jnp.asarray(flat_params)

        new_layers, i = [], 0
        for layer in self.layers:

            n = layer.weight.size
            weights = flat[i : i + n].reshape(layer.weight.shape, order=order)
            layer = eqx.tree_at(lambda l: l.weight, layer, weights)
            i += n

            n = layer.bias.size
            biases = flat[i : i + n].reshape(layer.bias.shape)
            layer = eqx.tree_at(lambda l: l.bias, layer, biases)
            i += n

            new_layers.append(layer)
        
        self.layers = new_layers


# Build MLP layers model
key, gkey = jax.random.split(gkey, 2)
UDE_denit = UDE_MLP(
    sizes=(IN_DIM, 16, 4, OUT_DIM),
    activations=[jax.nn.tanh, jax.nn.leaky_relu, jax.nn.relu],
    key=key,
)

# Update with trained parameters
UDE_denit.set_params(params_arr)

# Simulate UDE
X = ude_states[UDE_INPUTS].to_numpy().T
dX = jax.vmap(UDE_denit)(X).T

# Set SINDy problem
n_vars = len(SINDY_INPUTS)




if __name__ == '__main__':
    
    # Test with dummies
    # X = jax.random.uniform(key2, (N_SAMPLES, IN_DIM))
    # Y = jax.vmap(UDE_denit_full)(X)

    pass
