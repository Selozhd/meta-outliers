"""Anomaly Detection Algorithm using Single Generator Adversarial Training.

References:
    [1] https://arxiv.org/pdf/1809.10816.pdf
    [2] https://github.com/leibinghe/GAAL-based-outlier-detection
"""

import math

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.python.keras.engine import data_adapter


def get_gan_defaults():
    lr_d = 0.01
    lr_g = 1e-4
    decay = 1e-6
    momentum = 0.9
    return locals()


def get_gan_dataset_defaults(name):
    defaults = get_gan_defaults()
    onecluster = {'epochs': 1000}
    annthyroid = {'epochs': 20}
    WDBC = {'epochs': 240}
    waveform = {'epochs': 10}
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


class GAN(tf.keras.Model):

    def __init__(self, discriminator, generator, latent_dim):
        super(GAN, self).__init__()
        self.latent_dim = latent_dim
        self.discriminator = discriminator
        self.generator = generator

    def compile(self, d_optimizer, g_optimizer, loss, metrics=None):
        super(GAN, self).compile(loss=loss, metrics=metrics)
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer

    @tf.function
    def train_step(self, data):
        x, y_true, sample_weight = data_adapter.unpack_x_y_sample_weight(data)
        batch_size = tf.shape(x)[0]
        noise = tf.random.uniform(shape=(batch_size, self.latent_dim))
        generated_data = self.generator(noise)
        x = tf.concat([x, generated_data], axis=0)
        y = tf.concat([tf.ones((batch_size, 1)),
                       tf.zeros((batch_size, 1))],
                      axis=0)

        with tf.GradientTape() as tape:
            y_pred = self.discriminator(x)
            d_loss = self.compiled_loss(y, y_pred)
        self.d_optimizer.minimize(d_loss,
                                  self.discriminator.trainable_variables,
                                  tape=tape)

        trick = tf.ones((batch_size, 1))
        with tf.GradientTape() as tape:
            predictions = self.discriminator(self.generator(noise))
            generator_loss = self.compiled_loss(trick, predictions)
        grads = tape.gradient(generator_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(
            zip(grads, self.generator.trainable_weights))

        loss_logs = {
            "discriminator_loss": d_loss,
            "generator_loss": generator_loss
        }
        orig_pred, _ = tf.split(y_pred, 2)
        self.compiled_metrics.update_state(y_true, orig_pred, sample_weight)
        return_metrics = {}
        for metric in self.metrics:
            result = metric.result()
            if isinstance(result, dict):
                return_metrics.update(result)
            else:
                return_metrics[metric.name] = result
        loss_logs.update(return_metrics)
        return loss_logs

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