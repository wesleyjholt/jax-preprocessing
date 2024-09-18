from . import (
    StandardScaler, 
    MinMaxScaler, 
    RobustScaler
)

# This is a mapping from a unique string to a corresponding
# function which generates a transform.
# 
# I.e., each key should have a function with signature: 
#   f: Callable[[**hyperparams], AbstractTransform]
transform_factory = {
    'StandardScaler': lambda *_: StandardScaler(),
    'MinMaxScaler': lambda *_: MinMaxScaler(),
    'RobustScaler': lambda *_: RobustScaler()
    # TODO: Add more transforms here (as they are implemented).
}

def generate_transform(name, hyperparams):
    return transform_factory[name](**hyperparams)