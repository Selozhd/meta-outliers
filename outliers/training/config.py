"""Tools for characterisation of all supported models and model parameters."""

import dataclasses
import enum
import pprint
from typing import Any, Callable, Dict, List, Tuple

import tensorflow as tf
from sklearn.base import BaseEstimator, TransformerMixin

from outliers.autoencoder import AutoencoderAnomaly
from outliers.cumulative_anomaly import CumulativeAnomaly
from outliers.orderstats_anomaly import OrderStatsAnomaly
from outliers.preprocesor import sample_with_anomalies

_sk_prep = Tuple[BaseEstimator, TransformerMixin]


class AlgorithmType(enum.Enum):
    """All supported outlier algorithms."""

    OrderStats = enum.auto()
    Cumulative = enum.auto()
    Autoencoder = enum.auto()

    def get(self):
        return Outlier_Models[self]

    @classmethod
    def count(cls):
        return len(Outlier_Models)

    @classmethod
    def from_name(cls, name):
        return cls.__dict__.get(name)


Outlier_Models = {
    AlgorithmType.OrderStats: OrderStatsAnomaly,
    AlgorithmType.Cumulative: CumulativeAnomaly,
    AlgorithmType.Autoencoder: AutoencoderAnomaly,
}


@dataclasses.dataclass
class TrainConfig:
    """All parameters for training."""

    model: AlgorithmType
    model_kwargs: Dict[str, Any]
    autoencoder: tf.keras.Model
    sampler: sample_with_anomalies
    seed: int
    cts_transforms: List[_sk_prep]
    cat_transforms: List[_sk_prep]
    loss: Callable
    optimizer: tf.optimizers.Optimizer
    metrics: List[tf.metrics.Metric]
    callbacks: List[tf.keras.callbacks.Callback]
    batch_size: int
    epochs: int

    def __str__(self):
        result = dataclasses.asdict(self)
        return pprint.pformat(result)
    
    def set(self, key, value):
        setattr(self, key, value)

    @classmethod
    def from_dict(cls, contents):
        """Instantiate TrainConfig instance from a dictionary."""
        result = cls(**contents)
        return result

    def to_dict(self):
        """Serialize TrainConfig as a dictionary."""
        # Not using dataclasses.asdict() as deepcopy doesn't work with optimizer
        return self.__dict__
