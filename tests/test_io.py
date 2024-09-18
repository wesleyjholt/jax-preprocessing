import os
import equinox as eqx
from src.jax_preprocessing import StandardScaler, save_transform, load_transform

def test_standard(normal_data):
    data = normal_data
    original = StandardScaler.fit(data)
    try: 
        save_transform("test_standard_scaler.eqx", original)
        loaded = load_transform("test_standard_scaler.eqx")
    except Exception as e:
        os.remove("test_standard_scaler.eqx")
        print(e)
    os.remove("test_standard_scaler.eqx")
    assert eqx.tree_equal(original, loaded)