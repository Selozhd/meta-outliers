"""Meta-algorithms for outlier detection discussed in [1].

Implements percentile loss function, early stopping by kneecap detection,
and cumulative error scoring.

References:
    [1] N. Merrill and A. Eskandarian, "Modified Autoencoder Training and Scoring for
        Robust Unsupervised Anomaly Detection in Deep Learning," in IEEE Access, vol. 8,
        pp. 101824-101833, 2020, doi: 10.1109/ACCESS.2020.2997327.
"""

import copy
import os

import numpy as np
import tensorflow as tf
from kneed import KneeLocator
from tensorflow.keras.callbacks import CallbackList
from tensorflow.python.keras.engine import data_adapter
from tensorflow.python.keras.utils import tf_utils

from outliers.autoencoder import NotFittedModelError
from outliers.utils import name_of, pickle_dump, pickle_load


def get_average_loss(epoch_errors):
    w = np.mean(np.array(epoch_errors), axis=1)
    w = np.cumsum(w)/np.arange(1, len(w) + 1)
    return (np.arange(1, len(w) + 1, dtype=np.float32), w)


def get_elbow(x, y):
    try:
        kneedle = KneeLocator(x,
                              y,
                              S=5.0,
                              curve='convex',
                              direction='decreasing')
    except:
        return None
    elbows = list(kneedle.all_elbows)
    return elbows[0] if len(elbows) > 0 else None


class PercentileLoss(tf.keras.losses.Loss):
    """Modifies a loss function to mask values above a percentile."""

    def __init__(self,
                 loss_func,
                 percentile=.95,
                 name='percentile_loss',
                 dtype=tf.float32,
                 *args,
                 **kwargs):
        super(PercentileLoss, self).__init__(name=name, *args, **kwargs)
        self.loss_func = loss_func
        self.dtype = dtype
        self.percentile = percentile
        self.tf_quantile = self._get_quantile_func()

    def _get_quantile_func(self):
        """Returns a compiled `np.quantile` function compatible with `dtype`."""

        def _quantile(x, q):
            return np.asarray(np.quantile(x, q), dtype=self.dtype.name)

        @tf.function(input_signature=[tf.TensorSpec(None, self.dtype),
                                      tf.TensorSpec(None, tf.float64)]) # yapf: disable
        def tf_quantile(input, q):
            return tf.numpy_function(_quantile, [input, q], self.dtype)

        return tf_quantile

    def get_config(self):
        config = super(PercentileLoss, self).get_config()
        config.update({
            "loss_func": self.loss_func,
            "percentile": self.percentile,
            "dtype": self.dtype,
        })
        return config

    def call(self, y_true, y_pred):
        loss_value = self.loss_func(y_true, y_pred)
        quantile_95 = self.tf_quantile(loss_value, self.percentile)
        mask = tf.cast(loss_value < quantile_95, self.dtype)
        loss_value = mask*loss_value
        return loss_value


class KneeStopping(tf.keras.callbacks.Callback):
    """Early Stopping using Kneecap Detection."""

    def __init__(self,
                 X=None,
                 stopping_coef=1.2,
                 patience=1,
                 loss_func=tf.losses.MSE,
                 verbose=0,
                 baseline=None):
        super(KneeStopping, self).__init__()
        self.X = X
        self.patience = patience
        self.verbose = verbose
        self.loss_func = loss_func
        self.baseline = baseline
        self.stopping_coef = abs(stopping_coef)
        self.wait = 0
        self.stopped_epoch = 0
        self.monitor_op = np.less

    def on_train_begin(self, logs=None):
        # Allow instances to be re-used
        self.wait = 0
        self.stopped_epoch = 0
        self.best = np.Inf if self.monitor_op == np.less else -np.Inf
        self.best_weights = None
        self.best_epoch = 0
        self.epoch_errors = []
        self.is_calculate_errors = not self._has_model_epoch_errors()

    def on_epoch_begin(self, epoch, logs=None):
        if self.is_calculate_errors:
            self.model.epoch_errors = self.epoch_errors

    def on_epoch_end(self, epoch, logs=None):
        current = self.get_knee_epoch(logs)
        if current is None:
            return

        self.wait += 1
        if np.not_equal(current, self.best):
            self.best = current
            self.best_epoch = epoch
            self.wait = 0

        # Restart in order to move past the baseline.
        if self.baseline is not None and np.less(current, self.baseline):
            self.wait = 0

        # Only check after the first epoch.
        if self.wait >= self.patience and self._stopping_cond(current, epoch):
            self.stopped_epoch = epoch
            self.model.stop_training = True

    def on_train_end(self, logs=None):
        if self.is_calculate_errors:
            self.model.epoch_errors = self.epoch_errors
        if self.stopped_epoch > 0 and self.verbose > 0:
            print('Epoch %05d: early stopping' % (self.stopped_epoch + 1))

    def _has_model_epoch_errors(self):
        return self.model.epoch_errors == []

    def get_knee_epoch(self, logs):
        X = self.model._X if self.X is None else self.X
        if self.is_calculate_errors:
            epoch_loss = self.loss_func(X, self.model(X))
            self.epoch_errors.append(epoch_loss)
        else:
            self.epoch_errors = self.model.epoch_errors
        x, y = get_average_loss(self.epoch_errors)
        return get_elbow(x, y)

    def _stopping_cond(self, knee_epoch, epoch):
        return self.monitor_op(knee_epoch * self.stopping_coef, epoch)

    def get_config(self):
        config = {
            'X': self.X,
            'stopping_coef': self.stopping_coef,
            'patience': self.patience,
            'loss_func': name_of(self.loss_func),
            'verbose': self.verbose,
            'baseline': self.baseline
        }
        return config


class CumulativeAnomaly(tf.keras.Model):

    def __init__(self, model, alpha=5, *args, **kwargs):
        super(CumulativeAnomaly, self).__init__(*args, **kwargs)
        self._model = model
        self.alpha = alpha
        self._X = None
        self._threshold = None
        self.epoch_errors = None

    @property
    def cumulative_error(self):
        return np.sum(np.array(self.epoch_errors), axis=0)

    @property
    def training_scores(self):
        self._check_fitted()
        return self.cumulative_error

    def _get_threshold(self):
        return np.quantile(self.training_scores, 1 - self.alpha/100)

    @property
    def threshold(self):
        if self._threshold is None:
            return np.Inf
        else:
            return self._threshold

    @property
    def training_results(self):
        self._check_fitted()
        return np.asarray(self.training_scores > self.threshold, int)

    def on_train_begin(self, X):
        self._X = X
        self.epoch_errors = []
        if isinstance(self.loss, PercentileLoss):
            self._loss = self.loss.loss_func
        else:
            self._loss = self.loss

    def on_epoch_end(self, epoch, logs=None):
        X = self._X
        epoch_loss = self._loss(X, self(X))
        self.epoch_errors.append(epoch_loss)
        self._threshold = self._get_threshold()

    def on_train_end(self, logs):
        del self._X

    def call(self, inputs, training=None, mask=None):
        return self._model(inputs, training=training, mask=mask)

    def predict_scores(self, inputs, training=False, mask=None):
        raise NotImplementedError("CumulativeAnomaly Model can not be used "
                                  "for inference after training.")

    def _check_fitted(self):
        if not hasattr(self, 'epoch_errors'):
            raise NotFittedModelError("Call `.fit()` before.")

    def _get_config(self):
        config = {
            "epoch_errors": self.epoch_errors,
            "alpha": self.alpha,
            "_threshold": self._threshold,
            "loss": name_of(self.loss),
        }
        return config

    def save(self, path):
        config = self._get_config()
        tf.keras.models.save_model(self, os.path.join(path, 'model_weights'))
        pickle_dump(os.path.join(path, 'Cumulative'), config)

    @staticmethod
    def _get_custom_obj(config):
        custom_objects = {}
        if config.get('loss') == 'PercentileLoss':
            custom_objects.update({"PercentileLoss": PercentileLoss})
        return custom_objects

    @classmethod
    def load_from_path(cls, path):
        config = pickle_load(os.path.join(path, 'Cumulative'))
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
                   batch_size,
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
