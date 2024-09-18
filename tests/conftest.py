import pytest
import jax.random as jr

@pytest.fixture
def random_key():
    seed = 123
    return jr.PRNGKey(seed)

@pytest.fixture
def normal_data(random_key):
    return jr.normal(random_key, shape=(100,))*2 + 1.0