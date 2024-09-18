import jax.numpy as jnp
import equinox as eqx
from .abstract import AbstractTransform

class StandardScaler(AbstractTransform):
    """Standardize input data by removing the mean and scaling to unit variance."""
    mean: float = eqx.field(default_factory=lambda: jnp.array(0.0))
    std: float = eqx.field(default_factory=lambda: jnp.array(1.0))

    @classmethod
    def fit(cls, data, axis=None):
        mean = data.mean(axis=axis)
        std = data.std(axis=axis)
        return cls(mean, std)

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return data * self.std + self.mean

class MinMaxScaler(AbstractTransform):
    """Scale input data to the unit interval."""
    max: float = eqx.field(default_factory=lambda: jnp.array(1.0))
    min: float = eqx.field(default_factory=lambda: jnp.array(0.0))

    @classmethod
    def fit(cls, data, axis=None):
        min = data.min(axis=axis)
        max = data.max(axis=axis)
        return cls(min, max)

    def transform(self, data):
        return (data - self.min) / (self.max - self.min)

    def inverse_transform(self, data):
        return data * (self.max - self.min) + self.min

class RobustScaler(AbstractTransform):
    """Scale input data using median and interquartile range (more robust to outliers than e.g. StandardScaler)."""
    median: float = eqx.field(default_factory=lambda: jnp.array(0.0))
    iqr: float = eqx.field(default_factory=lambda: jnp.array(1.0))

    @classmethod
    def fit(cls, data, axis=None):
        median = jnp.median(data, axis=axis)
        iqr = jnp.percentile(data, 75, axis=axis) - jnp.percentile(data, 25, axis=axis)
        return cls(median, iqr)

    def transform(self, data):
        return (data - self.median) / self.iqr

    def inverse_transform(self, data):
        return data * self.iqr + self.median

