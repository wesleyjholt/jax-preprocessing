import os
import json
import equinox as eqx
from .factory import generate_transform

def save_transform(
    filename, 
    pytree, 
    generator_name=None, 
    hyperparams=None
):
    with open(filename, "wb") as f:
        if hyperparams is None:
            hyperparams = {}
        if generator_name is None:
            hyperparams['generator_name'] = pytree.__class__.__name__
        else:
            hyperparams['generator_name'] = generator_name
        hyperparam_str = json.dumps(hyperparams)
        f.write((hyperparam_str + "\n").encode())
        eqx.tree_serialise_leaves(f, pytree)

def load_transform(filename):
    with open(filename, "rb") as f:
        hyperparam_str = f.readline().decode().strip()
        hyperparams = json.loads(hyperparam_str)
        name = hyperparams.pop('generator_name')
        like = generate_transform(name, hyperparams)
        pytree = eqx.tree_deserialise_leaves(f, like)
        return pytree
