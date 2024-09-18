from abc import abstractmethod
import equinox as eqx

class AbstractTransform(eqx.Module):
    """Abstract base class for a data transform."""
    @classmethod
    @abstractmethod
    def fit(cls, data):
        pass

    @abstractmethod
    def transform(self, data):
        pass

    @abstractmethod
    def inverse_transform(self, data):
        pass