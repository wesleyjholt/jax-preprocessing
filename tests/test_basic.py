import jax.numpy as jnp

from src.jax_preprocessing import (
    StandardScaler,
    MinMaxScaler,
    RobustScaler
)

def test_standard(normal_data):
    original = normal_data
    scaler = StandardScaler.fit(original)
    transformed = scaler.transform(original)
    assert jnp.allclose(transformed.mean(), 0, atol=1e-6)
    assert jnp.allclose(transformed.std(), 1, atol=1e-6)
    untransformed = scaler.inverse_transform(transformed)
    assert jnp.allclose(original, untransformed, atol=1e-6)

def test_minmax(normal_data):
    original = normal_data
    scaler = MinMaxScaler.fit(original)
    transformed = scaler.transform(original)
    assert jnp.allclose(transformed.min(), 0, atol=1e-6)
    assert jnp.allclose(transformed.max(), 1, atol=1e-6)
    untransformed = scaler.inverse_transform(transformed)
    assert jnp.allclose(original, untransformed, atol=1e-6)

def test_robust(normal_data):
    original = normal_data
    scaler = RobustScaler.fit(original)
    transformed = scaler.transform(original)
    assert jnp.allclose(jnp.median(transformed), 0, atol=1e-6)
    assert jnp.allclose(jnp.percentile(transformed, 75) - jnp.percentile(transformed, 25), 1, atol=1e-6)
    untransformed = scaler.inverse_transform(transformed)
    assert jnp.allclose(original, untransformed, atol=1e-6)