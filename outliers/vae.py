r"""$\beta$-VAE for Unsupervised Outlier Detection.

Code partially taken from [2].

References:
    [1] https://arxiv.org/pdf/1804.03599.pdf
    [2] https://github.com/yzhao062/pyod/blob/master/pyod/models/vae.py
"""

import dataclasses
import pprint
from typing import Callable, List, Union

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2

Activation_Fn = Union[Callable, str]
Loss_Fn = Callable
Optimizer = Union[str, tf.optimizers.Optimizer]

_VAE_Defaults = {
    "encoder_neurons": [128, 64, 32],
    "decoder_neurons": [32, 64, 128],
    "latent_dim": 2,
    "hidden_activation": 'relu',
    "output_activation": None,
    "loss": tf.losses.mse,
    "dropout_rate": 0.2,
    "l2_regularizer": 0.1,
    "gamma": 1.0,
    "capacity": 0.0,
}


@dataclasses.dataclass
class VAEParams:

    encoder_neurons: List[int]
    decoder_neurons: List[int]
    latent_dim: int
    hidden_activation: Activation_Fn
    output_activation: Activation_Fn
    loss: Loss_Fn
    dropout_rate: float
    l2_regularizer: float
    gamma: float
    capacity: float

    def __str__(self):
        result = dataclasses.asdict(self)
        return pprint.pformat(result)

    def set(self, key, value):
        setattr(self, key, value)

    @classmethod
    def from_dict(cls, contents):
        """Instantiate VAEParams from a dictionary."""
        result = cls(**contents)
        return result

    @classmethod
    def from_defaults(cls):
        """Instantiate VAEParams with defaults."""
        result = cls(**_VAE_Defaults)
        return result

    def to_dict(self):
        """Serialize VAEParams as a dictionary."""
        return dataclasses.asdict(self)


class Sampling(tf.keras.layers.Layer):
    """Variational autoencoder sampling layer."""

    def call(self, inputs):
        """Reparametrisation by sampling from Gaussian, N(0,I).

        To sample from epsilon = Norm(0,I) instead of from likelihood Q(z|X)
        with latent variables z: z = z_mean + sqrt(var) * epsilon

        Args:
            args: tensor. Mean and log of variance of Q(z|X).
        Returns:
            z: tensor. Sampled latent variable.
        """
        z_mean, z_log = inputs
        batch = tf.shape(z_mean)[0]  # batch size
        dim = K.int_shape(z_mean)[1]  # latent dimension
        epsilon = tf.random.normal(shape=(batch, dim))  # mean=0, std=1.0
        return z_mean + tf.exp(0.5*z_log)*epsilon


def build_vae(params, x):
    """Build VAE = encoder + decoder + vae_loss."""
    params.n_features_ = x.shape[1]

    def vae_loss(params, inputs, outputs, z_mean, z_log):
        """VAE Loss.
        Loss = Recreation loss + Kullback-Leibler loss
        for probability function divergence (ELBO).
        gamma > 1 and capacity != 0 for beta-VAE
        """
        reconstruction_loss = params.loss(inputs, outputs)
        reconstruction_loss *= params.n_features_
        kl_loss = 1 + z_log - tf.square(z_mean) - tf.exp(z_log)
        kl_loss = -0.5*tf.reduce_sum(kl_loss, axis=-1)
        kl_loss = params.gamma*tf.abs(kl_loss - params.capacity)
        return tf.reduce_mean(reconstruction_loss + kl_loss)

    # Build Encoder
    inputs = Input(shape=(params.n_features_,))
    # Input layer
    layer = Dense(params.n_features_,
                  activation=params.hidden_activation)(inputs)
    # Hidden layers
    for neurons in params.encoder_neurons:
        layer = Dense(neurons,
                      activation=params.hidden_activation,
                      activity_regularizer=l2(params.l2_regularizer))(layer)
        layer = Dropout(params.dropout_rate)(layer)
    # Create mu and sigma of latent variables
    z_mean = Dense(params.latent_dim)(layer)
    z_log = Dense(params.latent_dim)(layer)
    # Use parametrisation sampling
    z = Sampling()([z_mean, z_log])
    # Instantiate encoder
    encoder = Model(inputs, [z_mean, z_log, z])

    # Build Decoder
    latent_inputs = Input(shape=(params.latent_dim,))
    # Latent input layer
    layer = Dense(params.latent_dim,
                  activation=params.hidden_activation)(latent_inputs)
    # Hidden layers
    for neurons in params.decoder_neurons:
        layer = Dense(neurons, activation=params.hidden_activation)(layer)
        layer = Dropout(params.dropout_rate)(layer)
    # Output layer
    outputs = Dense(params.n_features_,
                    activation=params.output_activation)(layer)
    # Instatiate decoder
    decoder = Model(latent_inputs, outputs)
    # Generate outputs
    outputs = decoder(encoder(inputs)[2])
    # Instantiate VAE
    vae = Model(inputs, outputs)
    vae.add_loss(vae_loss(params, inputs, outputs, z_mean, z_log))
    return vae
