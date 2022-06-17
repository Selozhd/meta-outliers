"""OrderStatsAnomaly Model and related functions and algorithms.

Includes tensorflow versions of functions in `orderstats.distributions`,
the OrderStatsAnomaly Model which uses custom loss weighting based on kappa
statistic threshold, and also an early stopping algorithm based on the kappa
statistic.
"""

import copy
import os

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import Callback, CallbackList
from tensorflow.python.keras.engine import data_adapter
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.platform import tf_logging

from order_statistics import scoring
from outliers.autoencoder import NotFittedModelError
from outliers.utils import pickle_dump, pickle_load

logger = tf_logging.get_logger()


# yapf: disable
def get_selection_vec(n, m):
    return tf.concat((tf.ones(m), tf.zeros(n - m)), axis=0)


def calculate_S_m(obs, m):
    return tf.reduce_sum(obs[:m])


def calculate_T_nm(obs, m):
    n = len(obs)
    return tf.reduce_sum(tf.gather(obs, range(m, n)))


def calculate_kappa(obs, m):
    n = len(obs)
    selection_vec = get_selection_vec(n, m)
    S_m = (1 / m) * tf.tensordot(obs, selection_vec, axes=1)
    unscaled_T_nm = tf.tensordot(obs, 1. - selection_vec, axes=1)
    T_n_m = tf.cast(tf.divide(1, n - m), dtype=tf.float32) * unscaled_T_nm
    return tf.divide(S_m, T_n_m)


def calculate_unscaled_kappa(obs, m):
    n = len(obs)
    selection_vec = get_selection_vec(n, m)
    S_m = tf.tensordot(obs, selection_vec, axes=1)
    T_n_m = tf.tensordot(obs, 1. - selection_vec, axes=1)
    return tf.divide(S_m, T_n_m)


def calculate_asymptotic_stat(obs, k):
    """$S_{n,k}/X_{(n-k+1)}$ ratio of k-trimmed sum to next order statistic."""
    n = len(obs)
    selection_vec = get_selection_vec(n, n - k)
    S_nk = tf.tensordot(obs, selection_vec, axes=1)
    X_k1 = tf.gather(obs, n - k)
    asymptotic_stat = tf.divide(S_nk, X_k1)
    return asymptotic_stat


def moving_average_kappa(obs):
    n = len(obs)
    cum_sum = tf.cumsum(obs)
    lens = tf.range(n, dtype=tf.float32)
    ma_kappa = (cum_sum / lens) / ((cum_sum[-1] - cum_sum) / (lens[-1] - lens))
    return ma_kappa


def moving_average_unscaled_kappa(obs):
    cum_sum = tf.cumsum(obs, axis=0)
    return tf.divide(cum_sum, cum_sum[-1] - cum_sum)


def moving_average_asymptotic_stat(obs):
    n = len(obs)
    cum_sum = tf.cumsum(obs)
    obs_inc = obs[::-1]
    X_k1 = tf.concat(
        (tf.gather(obs_inc, range(1)), tf.gather(obs_inc, range(n - 1))),
        axis=0)
    ma_asymptotic_stat = cum_sum[::-1] / X_k1
    return ma_asymptotic_stat


def get_anomaly_scores(X):
    X = tf.convert_to_tensor(X)
    sorting_idx = tf.argsort(X)
    sample_sorted = tf.gather(X, sorting_idx)
    scores_sorted = moving_average_unscaled_kappa(sample_sorted)
    scores = tf.gather(scores_sorted, tf.argsort(sorting_idx))
    return scores, scores_sorted


def moving_average(x, window_size):
    """Moving average of vector `x` on window_size."""
    return np.convolve(x, np.ones(window_size), 'valid') / window_size


# yapf: enable
class KappaThresholdStopping(Callback):
    """EarlyStopping `keras.callbacks.Callback` monitoring `kappa_threshold`."""

    def __init__(self,
                 sensitivity=0.1,
                 min_delta=1.3,
                 patience=4,
                 verbose=0,
                 baseline=None,
                 restore_best_weights=False):
        super(KappaThresholdStopping, self).__init__()
        self.monitor = 'kappa_threshold'
        self.sensitivity = sensitivity
        self.patience = patience
        self.verbose = verbose
        self.baseline = baseline
        self.min_delta = abs(min_delta)
        self.wait = 0
        self.stopped_epoch = 0
        self.restore_best_weights = restore_best_weights
        self.best_weights = None
        self.monitor_op = np.less
        self.min_delta *= -1

    def on_train_begin(self, logs=None):
        # Allow instances to be re-used
        self.wait = 0
        self.stopped_epoch = 0
        self.best = np.Inf if self.monitor_op == np.less else -np.Inf
        self.best_weights = None
        self.best_epoch = 0
        self.kappa_values = []
        self.still_increasing = True
        self.cnt_increasing = 0

    def on_epoch_end(self, epoch, logs=None):
        current = self.get_monitor_value(logs)
        if current is None:
            return
        if self.restore_best_weights and self.best_weights is None:
            # Restore the weights after first epoch if no progress is ever made.
            self.best_weights = self.model.get_weights()

        self.wait += 1
        # Do not stop at the initial increase
        if self.still_increasing:
            self.wait = 0
            if not self._is_increasing(current, self.kappa_values[0]):
                self.cnt_increasing += 1
                self.still_increasing = self.cnt_increasing < 10

        logger.debug("still increasing: ", self.still_increasing)
        logger.debug("count increasing: ", self.cnt_increasing)
        logger.debug("wait: ", self.wait)
        logger.debug("stable: ", self._is_stable())
        if not self._is_stable():
            self.best = current
            self.best_epoch = epoch
            if self.restore_best_weights:
                self.best_weights = self.model.get_weights()
            # Only restart wait if we beat both the baseline and our previous best.
            if self.baseline is None or self._is_improvement(
                    current, self.baseline):
                self.wait = 0

        # Only check after the first epoch.
        if self.wait >= self.patience and epoch > 0:
            self.stopped_epoch = epoch
            self.model.stop_training = True
            if self.restore_best_weights and self.best_weights is not None:
                if self.verbose > 0:
                    print(
                        'Restoring model weights from the end of the best epoch: '
                        f'{self.best_epoch + 1}.')
                self.model.set_weights(self.best_weights)

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0 and self.verbose > 0:
            print('Epoch %05d: early stopping' % (self.stopped_epoch + 1))

    def get_monitor_value(self, logs):
        logs = logs or {}
        monitor_value = logs.get(self.monitor)
        if monitor_value is None:
            tf_logging.warning(
                'Early stopping conditioned on metric `%s` '
                'which is not available. Available metrics are: %s',
                self.monitor, ','.join(list(logs.keys())))
        self.kappa_values.append(monitor_value)
        return monitor_value

    def _is_stable(self):
        kappa_window = moving_average(self.kappa_values, 3)
        last_ten = kappa_window[-10:]
        range_ = (np.max(last_ten) - np.min(last_ten))/10
        return np.less(range_, self.sensitivity)

    def _is_improvement(self, monitor_value, reference_value):
        return np.greater(monitor_value - self.min_delta, reference_value)

    def _is_increasing(self, monitor_value, reference_value):
        """Checks if a statistic is increasing within a degree of error."""
        return np.greater(monitor_value + self.min_delta, reference_value)

    def get_config(self):
        config = {
            'sensitivity': self.sensitivity,
            'min_delta': self.min_delta,
            'patience': self.patience,
            'verbose': self.verbose,
            'baseline': self.baseline,
            'restore_best_weights': self.restore_best_weights
        }
        return config


@tf.function
def inverse_score_weights(losses, scores, errors, threshold):
    """Inverse anomaly score loss weights for samples above the threshold.
    Args:
        losses: Losses before weighting, shape: [batch_size,].
        scores: Outlier scores of the previous epoch, shape: [val_size].
        errors: Validation errors of the previous epoch, shape: [val_size].
        threshold: Kappa threshold for outlier score.
    Returns:
        Sample_weights for the compiled_loss, shape: [batch_size,].
    """
    if scores is None:
        return tf.ones(tf.size(losses))
    batch_size = tf.size(losses)
    val_size = tf.size(scores)
    below_threshold = tf.less(losses, threshold)
    _errors = tf.reshape(tf.repeat(errors, batch_size), shape=(-1, batch_size))
    idx = tf.reduce_sum(tf.cast(tf.less(_errors, losses), tf.int32), axis=0)
    idx = tf.clip_by_value(idx, clip_value_max=val_size - 1, clip_value_min=0)
    loss_scores = tf.gather(scores, idx)
    inv_scores = tf.divide(1, loss_scores)
    weights = tf.where(below_threshold, tf.ones(batch_size), inv_scores)
    return weights


@tf.function
def no_weights_above_threshold(loss, threshold):
    return tf.cast(tf.less(loss, threshold), tf.float32)


class OrderStatsAnomaly(tf.keras.Model):

    def __init__(self, model, loss_weights=True, *args, **kwargs):
        super(OrderStatsAnomaly, self).__init__(*args, **kwargs)
        self._model = model
        self._errors = None
        self._scores = None
        self._kappa_threshold = None
        self.loss_weights = loss_weights

    @property
    def error_threshold(self):
        """Threshold value for errors corresponding to $\kappa$ value."""
        if self._errors is None:
            return np.Inf
        return self._errors[np.sum(self._scores < self.threshold)]

    @property
    def training_scores(self):
        self._check_fitted()
        return np.nan_to_num(self._scores)

    @property
    def threshold(self):
        """Chosen $\kappa$ value for threshold."""
        if self._errors is None:
            return np.Inf
        return self._kappa_threshold

    @property
    def training_results(self):
        self._check_fitted()
        return self.training_scores > self.threshold

    def call(self, inputs, training=None, mask=None):
        return self._model(inputs, training=training, mask=mask)

    def predict_scores(self, inputs, training=False, mask=None):
        preds = self(inputs, training=training, mask=mask)
        loss = self.loss(inputs, preds)
        scores, _ = get_anomaly_scores(loss)
        scores = np.nan_to_num(scores)
        return scores

    def _check_fitted(self):
        if not hasattr(self, '_scores'):
            raise NotFittedModelError("Call `.fit()` before.")

    def _get_config(self):
        config = {
            "_errors": self._errors,
            "_scores": self._scores,
            "_kappa_threshold": self._kappa_threshold,
            "loss_weights": self.loss_weights,
            "loss": self.loss,
        }
        return config

    def save(self, path):
        config = self._get_config()
        tf.keras.models.save_model(self, os.path.join(path, 'model_weights'))
        pickle_dump(os.path.join(path, 'OrderStats'), config)

    @staticmethod
    def _get_custom_obj(config):
        custom_objects = {}
        if config.get('loss') == 'PercentileLoss':
            from outliers.cumulative_anomaly import PercentileLoss
            custom_objects.update({"PercentileLoss": PercentileLoss})
        return custom_objects

    @classmethod
    def load_from_path(cls, path):
        config = pickle_load(os.path.join(path, 'OrderStats'))
        custom_objects = cls._get_custom_obj(config)
        _model = tf.keras.models.load_model(os.path.join(path, 'model_weights'),
                                            custom_objects)
        model = cls(model=_model, loss_weights=config.get('loss_weights'))
        model._errors = config.get("_errors")
        model._scores = config.get("_scores")
        model._kappa_threshold = config.get("_kappa_threshold")
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
        scores, errors, threshold = sample_weight
        # Run forward pass.
        with tf.GradientTape() as tape:
            x_pred = self(x, training=True)
            with tape.stop_recording():
                if self.loss_weights:
                    losses = self.loss(x, x_pred)
                    sample_weight = inverse_score_weights(
                        losses, scores, errors, threshold)
                else:
                    sample_weight = None
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
        y_pred = tf.cast(self.loss(x, x_pred) > threshold, dtype=tf.int32)
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
        x, y, threshold = data_adapter.unpack_x_y_sample_weight(data)
        x_pred = self(x, training=False)
        # Updates stateful loss metrics.
        self.compiled_loss(x, x_pred, None, regularization_losses=self.losses)
        y_pred = tf.cast(self.loss(x, x_pred) > threshold, dtype=tf.int32)
        self.compiled_metrics.update_state(y, y_pred)
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
        """Custom alternative for `keras.Model.fit`."""
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
                                     steps=int(np.ceil(N/batch_size)))

        self.stop_training = False
        self._train_counter.assign(0)
        callbacks.on_train_begin()
        training_logs = None
        self.epoch_erros = []
        for epoch in range(epochs):
            self.reset_metrics()
            callbacks.on_epoch_begin(epoch)

            for step, (x_batch_train, y_batch_train) in enumerate(dataset):
                callbacks.on_train_batch_begin(step)
                logs = self.train_step(
                    (x_batch_train, y_batch_train, (self._scores, self._errors,
                                                    self.error_threshold)))
                callbacks.on_train_batch_end(step, logs)

            logs = tf_utils.sync_to_numpy_or_python_type(logs)
            epoch_logs = copy.copy(logs)

            # End of each epoch
            X_pred = self(X, training=False)
            self._errors = self.loss(X, X_pred)
            _, self._scores = get_anomaly_scores(self._errors)
            self._kappa_threshold = scoring.get_kappa_threshold(self._scores)

            if validation_data:
                val_logs = self.test_step(
                    (*validation_data, self.error_threshold))
                val_logs = {
                    'val_' + name: val for name, val in val_logs.items()
                }
                epoch_logs.update(val_logs)

            epoch_logs.update({"kappa_threshold": self.threshold})
            callbacks.on_epoch_end(epoch, epoch_logs)
            training_logs = epoch_logs
            if self.stop_training:
                break
        callbacks.on_train_end(logs=training_logs)
        return self.history