"""Anomaly Detection Algorithm using Multiple Generator Adversarial Training.

References:
    [1] https://arxiv.org/pdf/1809.10816.pdf
    [2] https://github.com/leibinghe/GAAL-based-outlier-detection
"""

import math

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.python.keras.engine import data_adapter


def get_multigan_defaults():
    k = 10
    lr_d = 0.01
    lr_g = 1e-4
    decay = 1e-6
    momentum = 0.9
    return locals()


def get_multigan_dataset_defaults(name):
    defaults = get_multigan_defaults()
    onecluster = {'epochs': 1500}
    annthyroid = {'epochs': 25}
    WDBC = {'epochs': 200}
    annthyroid = {'epochs': 25}
    waveform = {'epochs': 30}
    spam_base = {'epchs': 40}
    defaults.update(locals()[name])
    return defaults


def create_generator(latent_dim):
    generator = tf.keras.Sequential()
    generator.add(
        layers.Dense(latent_dim,
                     input_dim=latent_dim,
                     activation='relu',
                     kernel_initializer=tf.initializers.Identity(gain=1.0)))
    generator.add(
        layers.Dense(latent_dim,
                     activation='relu',
                     kernel_initializer=tf.initializers.Identity(gain=1.0)))
    return generator


def create_discriminator(data_size, latent_dim):
    discriminator = tf.keras.Sequential()
    discriminator.add(
        layers.Dense(math.ceil(math.sqrt(data_size)),
                     input_dim=latent_dim,
                     activation='relu',
                     kernel_initializer=tf.initializers.VarianceScaling(
                         scale=1.0,
                         mode='fan_in',
                         distribution='normal',
                         seed=None)))
    discriminator.add(
        layers.Dense(1,
                     activation='sigmoid',
                     kernel_initializer=tf.initializers.VarianceScaling(
                         scale=1.0,
                         mode='fan_in',
                         distribution='normal',
                         seed=None)))
    return discriminator


def _quantile(x, q):
    return np.asarray(np.quantile(x, q), dtype=np.float32)


@tf.function(input_signature=[tf.TensorSpec(None, tf.float32),
                              tf.TensorSpec(None, tf.float64)]) # yapf: disable
def tf_quantile(input, q):
    return tf.numpy_function(_quantile, [input, q], tf.float32)


class MultiGAN(tf.keras.Model):

    def __init__(self, discriminator, generator_func, n_generators, latent_dim):
        super(MultiGAN, self).__init__()
        self.latent_dim = latent_dim
        self.n_generators = n_generators
        self.discriminator = discriminator
        self.generators = [
            generator_func(latent_dim) for _ in range(n_generators)
        ]

    def _generator_optimizer_config(self, optimizer_dict):
        if not isinstance(optimizer_dict, dict):
            return optimizer_dict
        name = optimizer_dict.get('name', 'SGD')
        config = {
            'class_name': name,
            'config': {
                'name': name,
                'learning_rate': optimizer_dict.get('lr_g', 1e-4),
                'decay': optimizer_dict.get('decay', 1e-6),
                'momentum': optimizer_dict.get('momentum', 0.9)
            }
        }
        return config

    @tf.function
    def _get_split_size(self, batch_size):
        k = tf.cast(self.n_generators, tf.float32)
        range_k = tf.range(self.n_generators - 1, dtype=tf.float32)
        block = ((1 + k)*k) // 2
        idx = (k + (k - range_k))*(range_k + 1)/2
        idx = idx*tf.cast(batch_size // tf.cast(block, tf.int32), tf.float32)
        idx = tf.concat([[0], idx, [batch_size]], axis=0)
        split_size = tf.cast(tf.experimental.numpy.diff(idx), tf.int32)
        return split_size

    def compile(self, d_optimizer, g_optimizer_dict, loss, metrics=None):
        super(MultiGAN, self).compile(loss=loss, metrics=metrics)
        self.d_optimizer = d_optimizer
        optimizer_config = self._generator_optimizer_config(g_optimizer_dict)
        [
            gen.compile(optimizer=tf.optimizers.get(optimizer_config))
            for gen in self.generators
        ]

    @tf.function
    def train_step(self, data):
        x, y_true, sample_weight = data_adapter.unpack_x_y_sample_weight(data)
        batch_size = tf.shape(x)[0]
        noise = tf.random.uniform(shape=(batch_size, self.latent_dim))

        # Create generated data
        split_size = self._get_split_size(batch_size)
        noises = tf.split(noise, split_size)
        generated_data = [gen(n) for n, gen in zip(noises, self.generators)]
        x = tf.concat([x, *generated_data], axis=0)
        y = tf.concat([tf.ones((batch_size, 1)),
                       tf.zeros((batch_size, 1))],
                      axis=0)

        # Train discriminator
        with tf.GradientTape() as tape:
            y_pred = self.discriminator(x)
            d_loss = self.compiled_loss(y, y_pred)
        self.d_optimizer.minimize(d_loss,
                                  self.discriminator.trainable_variables,
                                  tape=tape)

        # Train generators with p_value scores from the loss
        full_pred = self.discriminator(self._X)
        generator_p = [
            tf_quantile(full_pred, i/self.n_generators)
            for i in range(self.n_generators)
        ]
        p_values = [tf.fill((batch_size, 1), p) for p in generator_p]
        new_noise = tf.random.uniform(shape=(batch_size, self.latent_dim))
        generator_losses = []
        for p_value, generator in zip(p_values, self.generators):
            with tf.GradientTape() as tape:
                predictions = self.discriminator(generator(new_noise))
                generator_loss = self.compiled_loss(p_value, predictions)
            grads = tape.gradient(generator_loss, generator.trainable_weights)
            generator.optimizer.apply_gradients(
                zip(grads, generator.trainable_weights))
            generator_losses.append(generator_loss)

        # Metrics
        train_logs = {"discriminator_loss": d_loss}
        train_logs.update(
            {f'generator{i}': loss for i, loss in enumerate(generator_losses)})
        orig_pred, _ = tf.split(y_pred, 2)
        self.compiled_metrics.update_state(y_true, orig_pred, sample_weight)
        return_metrics = {}
        for metric in self.metrics:
            result = metric.result()
            if isinstance(result, dict):
                return_metrics.update(result)
            else:
                return_metrics[metric.name] = result
        train_logs.update(return_metrics)
        return train_logs

    @tf.function
    def test_step(self, data):
        """The logic for one evaluation step.
        Args:
            data: A nested structure of `Tensor`s.
        Returns:
            A `dict` containing values that will be passed to
            `tf.keras.callbacks.CallbackList.on_train_batch_end`. Typically, the
            values of the `Model`'s metrics are returned.
        """
        x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)
        y_pred = self.discriminator(x, training=False)
        # Updates stateful loss metrics.
        self.compiled_loss(y,
                           y_pred,
                           sample_weight,
                           regularization_losses=self.losses)
        self.compiled_metrics.update_state(y, y_pred, sample_weight)
        # Collect metrics to return
        return_metrics = {}
        for metric in self.metrics:
            result = metric.result()
            if isinstance(result, dict):
                return_metrics.update(result)
            else:
                return_metrics[metric.name] = result
        return return_metrics