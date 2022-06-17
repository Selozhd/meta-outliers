"""Autoencoder anomaly detection model and utility functions and layers."""

import copy
import os

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import CallbackList
from tensorflow.python.keras.engine import data_adapter
from tensorflow.python.keras.utils import tf_utils

from outliers.utils import pickle_dump, pickle_load


def get_matrix_dimensions(dimensions):
    """Returns the sizes for weights matricies for an auto-encoder."""
    encoder_matricies = list(zip(dimensions[:-1], dimensions[1:]))
    dimensions = dimensions[::-1]
    decoder_matricies = list(zip(dimensions[:-1], dimensions[1:]))
    return (encoder_matricies, decoder_matricies)


def get_lazy_dimensions(dimensions):
    """Returns inputs for an autoencoder of `keras.layers.Dense`."""
    dims = [*dimensions[1:], *dimensions[::-1][1:]]
    return dims


def get_autoencoder(dimensions, dtype=tf.float32):
    dims = get_lazy_dimensions(dimensions)
    model = tf.keras.Sequential(
        [tf.keras.layers.Dense(i, dtype=dtype) for i in dims])
    return model


def get_autoencoder_shapes(autoencoder):
    """Returns the shapes of W matricies for an autoencoder."""
    return [tuple(layer.weights[0].shape) for layer in autoencoder.layers]


class NotFittedModelError(Exception):
    pass


class Mask2d(tf.keras.layers.Layer):
    """Masks a 2d tensor by using a mask value to skip certain entries."""

    def __init__(self, mask_value=0., **kwargs):
        super(Mask2d, self).__init__(**kwargs)
        self.supports_masking = True
        self.mask_value = mask_value
        self._compute_output_and_mask_jointly = True

    def compute_mask(self, inputs, mask=None):
        return tf.not_equal(inputs, self.mask_value)

    def call(self, inputs):
        boolean_mask = tf.not_equal(inputs, self.mask_value)
        outputs = inputs*tf.cast(boolean_mask, inputs.dtype)
        # Compute the mask and outputs simultaneously.
        outputs._keras_mask = boolean_mask  # pylint: disable=protected-access
        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {'mask_value': self.mask_value}
        base_config = super(Mask2d, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class ApplyMask2d(tf.keras.layers.Layer):
    """Applies mask generated by Mask2d before loss is calculated."""

    def call(self, inputs, mask=None):
        if mask is None:
            return inputs
        mask = tf.cast(mask, inputs.dtype)
        output = tf.math.multiply(inputs, mask)
        return output


class AutoencoderAnomaly(tf.keras.Model):
    """A wrapper `tf.keras.Model` for autoencoder anomaly detection models.

    Attributes:
        model: An autoencoder `tf.keras.Model`.
        alpha: Percentage of anomalies in the data.
        threshold: The loss threshold point which seperates anomalous instances
            from normal observations. threshold is used dynamically during
            training to calculate metrics for binary classification.
    """

    def __init__(self, model, alpha, *args, **kwargs):
        super(AutoencoderAnomaly, self).__init__(*args, **kwargs)
        self._model = model
        self.alpha = alpha
        self._threshold = np.Inf

    @property
    def training_scores(self):
        self._check_fitted()
        return self.epoch_errors[-1]

    def _get_threshold(self):
        return np.quantile(self.training_scores, 1 - self.alpha/100)

    @property
    def threshold(self):
        return self._threshold

    @property
    def training_results(self):
        self._check_fitted()
        return np.asarray(self.training_scores > self.threshold, int)

    def on_train_begin(self, X):
        self._X = X
        self.epoch_errors = []
        self._loss = self.loss

    def on_epoch_end(self, epoch, epoch_logs):
        X = self._X
        epoch_loss = self._loss(X, self(X))
        self.epoch_errors.append(epoch_loss)
        self._threshold = self._get_threshold()

    def on_train_end(self, logs):
        del self._X

    def call(self, inputs, training=None, mask=None):
        return self._model(inputs, training=training, mask=mask)

    def predict_scores(self, inputs, training=False, mask=None):
        preds = self(inputs, training=training, mask=mask)
        loss = self.loss(inputs, preds)
        return loss

    def _check_fitted(self):
        if not hasattr(self, 'epoch_errors'):
            raise NotFittedModelError("Call `.fit()` before.")

    def _get_config(self):
        config = {
            "epoch_errors": self.epoch_errors,
            "alpha": self.alpha,
            "_threshold": self._threshold,
            "loss": self.loss,
        }
        return config

    def save(self, path):
        config = self._get_config()
        tf.keras.models.save_model(self, os.path.join(path, 'model_weights'))
        pickle_dump(os.path.join(path, 'Autoencoder'), config)

    @staticmethod
    def _get_custom_obj(config):
        custom_objects = {}
        if config.get('loss') == 'PercentileLoss':
            from outliers.cumulative_anomaly import PercentileLoss
            custom_objects.update({"PercentileLoss": PercentileLoss})
        return custom_objects

    @classmethod
    def load_from_path(cls, path):
        config = pickle_load(os.path.join(path, 'Autoencoder'))
        custom_objects = cls._get_custom_obj(config)
        _model = tf.keras.models.load_model(os.path.join(path, 'model_weights'),
                                            custom_objects)
        model = cls(model=_model, alpha=config.get('alpha'))
        model.epoch_errors = config.get("epoch_errors")
        model._threshold = config.get("_threshold")
        model.loss = config.get("loss")
        return model

    @tf.function
    def train_step(self, data):
        """The logic for one training step.
        Args:
            data: A nested structure of `Tensor`s.
        Returns:
            A `dict` containing values that will be passed to
            `tf.keras.callbacks.CallbackList.on_train_batch_end`. Typically, the
            values of the `Model`'s metrics are returned. Example:
            `{'loss': 0.2, 'accuracy': 0.7}`.
        """
        x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)
        # Run forward pass.
        with tf.GradientTape() as tape:
            x_pred = self(x, training=True)
            loss = self.compiled_loss(x,
                                      x_pred,
                                      sample_weight,
                                      regularization_losses=self.losses)
        if self.loss and y is None:
            raise TypeError(
                f'Target data is missing. Your model has `loss`: {self.loss}, '
                'and therefore expects target data to be passed in `fit()`.')
        # Run backwards pass.
        self.optimizer.minimize(loss, self.trainable_variables, tape=tape)
        y_pred = tf.cast(self._loss(x, x_pred) > self.threshold, dtype=tf.int32)
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
        x_pred = self(x, training=False)
        # Updates stateful loss metrics.
        self.compiled_loss(x,
                           x_pred,
                           sample_weight,
                           regularization_losses=self.losses)
        y_pred = tf.cast(self._loss(x, x_pred) > self.threshold, dtype=tf.int32)
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

    def custom_fit(self,
                   X,
                   y,
                   batch_size=None,
                   epochs=1,
                   callbacks=None,
                   verbose=1,
                   validation_data=None):
        """Custom `Model.fit()` method for autoencoder training.
        
        The `custom_fit()` is an alternative API for `Model.fit()` which works
        specifically for autoencoder networks. X is used as input and as target
        data, while y is used in conjuction with `threshold` for the creation of
        evaluation metrics dynamically during training.

        Args:
            X: Input data, training an target data for the autoencoder network.
            y: Target data for evaluation. Like the input data `X`, it could be
                either Numpy array(s) or TensorFlow tensor(s). It should be
                consistent with `x` (you cannot have Numpy inputs and tensor
                targets, or inversely). If `x` is a dataset, generator, or
                `keras.utils.Sequence` instance, `y` should not be specified
                (since targets will be obtained from `x`).
            batch_size: Integer. See `Model.fit()`
            epochs: Integer. See `Model.fit()`
            callbacks: List of `keras.callbacks.Callback` instances.
                See `Model.fit()`
            verbose: 0, 1, or 2. Verbosity mode, passed to `Callback`s.
                0 = silent, 1 = progress bar, 2 = one line per epoch.
            validation_data: A tuple of (X, y, sample_weight) for evaluation.
                Evaluation is done at the end of each epoch using `test_step()`
                method.
        """
        dataset = tf.data.Dataset.from_tensor_slices((X, y))
        N = len(dataset)
        dataset = dataset.shuffle(N, reshuffle_each_iteration=True)
        dataset = dataset.batch(batch_size)

        if not isinstance(callbacks, CallbackList):
            callbacks = CallbackList(callbacks,
                                     add_history=True,
                                     add_progbar=verbose != 0,
                                     model=self,
                                     verbose=verbose,
                                     epochs=epochs,
                                     steps=int(tf.math.ceil(N/batch_size)))

        self.stop_training = False
        self._train_counter.assign(0)
        self.on_train_begin(X)
        callbacks.on_train_begin()
        training_logs = None
        for epoch in range(epochs):
            self.reset_metrics()
            callbacks.on_epoch_begin(epoch)

            for step, (x_batch_train, y_batch_train) in enumerate(dataset):
                callbacks.on_train_batch_begin(step)
                logs = self.train_step((x_batch_train, y_batch_train))
                callbacks.on_train_batch_end(step, logs)

            logs = tf_utils.sync_to_numpy_or_python_type(logs)
            epoch_logs = copy.copy(logs)

            if validation_data:
                val_logs = self.test_step(validation_data)
                val_logs = {
                    'val_' + name: val for name, val in val_logs.items()
                }
                epoch_logs.update(val_logs)

            self.on_epoch_end(epoch, epoch_logs)
            callbacks.on_epoch_end(epoch, epoch_logs)
            training_logs = epoch_logs
            if self.stop_training:
                break
        callbacks.on_train_end(logs=training_logs)
        self.on_train_end(logs=training_logs)
        return self.history
