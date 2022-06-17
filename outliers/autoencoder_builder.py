"""Autoencoder builder.

Code partially taken from [1].

References:
    [1] https://github.com/yzhao062/pyod/blob/master/pyod/models/auto_encoder.py
"""

import dataclasses
import pprint
from typing import Callable, List, Union

import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import l2

Activation_Fn = Union[Callable, str]

_Autoencoder_Defaults = {
    "hidden_neurons": [128, 64, 32, 64, 128],
    "hidden_activation": 'relu',
    "output_activation": None,
    "dropout_rate": 0.2,
    "l2_regularizer": 0.1,
}


@dataclasses.dataclass
class AutoencoderParams:

    hidden_neurons: List[int]
    hidden_activation: Activation_Fn
    output_activation: Activation_Fn
    dropout_rate: float
    l2_regularizer: float

    def __str__(self):
        result = dataclasses.asdict(self)
        return pprint.pformat(result)

    def set(self, key, value):
        setattr(self, key, value)

    @classmethod
    def from_dict(cls, contents):
        """Instantiate AutoencoderParams from a dictionary."""
        result = cls(**contents)
        return result

    @classmethod
    def from_defaults(cls):
        """Instantiate AutoencoderParams with defaults."""
        result = cls(**_Autoencoder_Defaults)
        return result

    def to_dict(self):
        """Serialize AutoencoderParams as a dictionary."""
        return dataclasses.asdict(self)


def build_autoencoder(params, x):
    """Builds an Autoencoder composed of Dense layers."""
    params.set("n_features_", x.shape[1])
    params.set("hidden_neurons_", [x.shape[1], *params.hidden_neurons])
    model = tf.keras.models.Sequential()
    # Input layer
    model.add(
        Dense(params.hidden_neurons_[0],
              activation=params.hidden_activation,
              input_shape=(params.n_features_,),
              activity_regularizer=l2(params.l2_regularizer)))
    model.add(Dropout(params.dropout_rate))
    # Additional layers
    for i, hidden_neurons in enumerate(params.hidden_neurons_, 1):
        model.add(
            Dense(hidden_neurons,
                  activation=params.hidden_activation,
                  activity_regularizer=l2(params.l2_regularizer)))
        model.add(Dropout(params.dropout_rate))
    # Output layers
    model.add(
        Dense(params.n_features_,
              activation=params.output_activation,
              activity_regularizer=l2(params.l2_regularizer)))
    return model
