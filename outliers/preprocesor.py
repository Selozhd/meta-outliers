"""Pipeline for compact preprocessing."""

from functools import reduce
from typing import List, Tuple

import numpy as np
import pandas as pd
from numpy import random
from sklearn.base import BaseEstimator, TransformerMixin

from order_statistics import utils as stat_utils
from outliers.utils import clean_null_from_list, name_of

_sk_prep = Tuple[BaseEstimator, TransformerMixin]


def pd_wrapper(func):
    """Preserves `pd.DataFrame` inputs for `sample_with_anomalies`."""

    def pandas_func(*args, **kwargs):
        df = args[0]
        if isinstance(df, pd.DataFrame):
            columns = df.columns
            result = func(*args, **kwargs)
            pd_result = pd.DataFrame(result[0], columns=columns)
            return (pd_result, *result[1:])
        else:
            return func(*args, **kwargs)

    return pandas_func


def get_random_states(n_state, seed=42):
    """Repeatable random states from a single seed."""
    rng = np.random.default_rng(seed)
    seeds = rng.integers(0, 1000000, n_state)
    return [np.random.default_rng(i) for i in seeds]


class sample_with_anomalies:
    """Samples a single label data set with added anomalies.

    Attributes:
        normal_label: Integer. The label to choose for normal instances.
        anomaly_label: Integer, List or 'all'. If Integer, or List anomalies are
        sampled only from those labels. If 'all' anomalies are sampled randomly.
        percentage: Float in [0, 1). Percentage of anomalies in the final data.
        ratio: Integer or float. Ratio of normal instances to anomalies.
        p: List. Weights of each anomaly label.
    Raises:
        ValueError: If both or none of percentage and ratio is provided.
    """

    def __init__(self,
                 normal_label,
                 anomaly_label,
                 percentage=None,
                 ratio=None,
                 p=None):
        self.label = normal_label
        self.anomaly_label = anomaly_label
        self.p = p if p is not None else np.ones(np.shape(normal_label))
        self.seed = None
        if percentage and ratio is not None:
            raise ValueError('Please provide only one of percentage and ratio.')
        elif percentage is None and ratio is None:
            raise ValueError('Please provide one of percentage and ratio.')
        elif percentage is not None:
            self.ratio = self._get_ratio(percentage)
        else:
            self.ratio = ratio
        self._check_p()
        self._check_ratio()

    def get_params(self, deep=False):
        params = {
            'normal_label': self.label,
            'anomaly_label': self.anomaly_label,
            'ratio': self.ratio,
            'p': self.p.tolist(),
            'seed': self.seed
        }
        return params

    def _get_ratio(self, percentage):
        return percentage/(1 - percentage)

    def _get_anomaly_idx(self, labels):
        if self.anomaly_label is 'all':
            anomaly_idx = np.where(np.not_equal(labels, self.label))[0]
            p = np.ones(len(anomaly_idx))/len(anomaly_idx)
        elif isinstance(self.anomaly_label, list):
            anomaly_idx = [
                np.where(np.equal(labels, label))[0]
                for label in self.anomaly_label
            ]
            anomaly_idx, p = stat_utils.mix_arrays(anomaly_idx, self.p, axis=0)
            p = p/np.sum(p)
        else:
            anomaly_idx = np.where(np.equal(labels, self.anomaly_label))[0]
            p = None
        return anomaly_idx, p

    def _check_p(self):
        if isinstance(self.anomaly_label,
                      list) and (len(self.anomaly_label) != len(self.p)):
            raise ValueError('Length of anomaly_label and p must match.')

    def _check_ratio(self):
        if self.ratio < 0:
            raise ValueError('percentage has to be in [0, 1), or '
                             'ratio can not be negative.')

    @classmethod
    def from_config(cls, config):
        sampler = cls(
            normal_label=config.get('normal_label'),
            anomaly_label=config.get('anomaly_label'),
            ratio=config.get('ratio'),
            p=config.get('p'),
        )
        sampler.seed = config.get('seed')
        return sampler

    def __call__(self, data, labels, size=None, seed=None):
        """Samples new data with anomaly labels from the data.
        Args:
            data: A `np.ndarray` data set.
            labels: Set of categorical labels corresponding to the data.
            size: Integer. Sample size for normal data.
            seed: Integer seed for reproducibility.
        Returns:
            A sampled `np.ndarray`, and a label array of anomalies.
        """
        data = np.asarray(data)
        label_idx = np.equal(labels, self.label)
        n = np.sum(label_idx) if size is None else size
        n_anomaly = int(n*self.ratio)
        label_idx = np.where(label_idx)[0]
        anomaly_idx, p = self._get_anomaly_idx(labels)
        if seed is None:
            normal_idx = random.choice(label_idx, n, replace=False, p=None)
            sample_idx = random.choice(anomaly_idx,
                                       n_anomaly,
                                       replace=False,
                                       p=p)
        else:
            self.seed = seed
            rng1, rng2 = get_random_states(n_state=2, seed=seed)
            normal_idx = rng1.choice(label_idx, n, replace=False, p=None)
            sample_idx = rng2.choice(anomaly_idx, n_anomaly, replace=False, p=p)
        normal_sample = data[normal_idx]
        anomaly_sample = data[sample_idx]
        x, y = stat_utils.mix_arrays([normal_sample, anomaly_sample], axis=0)
        return x, y


class Centralizor(TransformerMixin, BaseEstimator):

    def fit(self, X):
        self.means = np.mean(X)
        return self

    def transform(self, X):
        X_ = X.copy()
        X_ -= self.means
        return X_


class ApplyFunction(TransformerMixin, BaseEstimator):

    def __init__(self, func, **kwargs):
        super(ApplyFunction, self).__init__(**kwargs)
        self.func = func

    def fit(self, X):
        return self

    def transform(self, X):
        return self.func(X)

    def get_params(self, deep=False):
        return {'function': name_of(self.func)}


def get_rows(X, rows):
    if rows is None:
        return None
    if isinstance(X, pd.DataFrame):
        return X[rows]
    elif isinstance(X, np.ndarray):
        return np.take(X, rows, axis=-1)


def _fit_transform(X, sk_transformer):
    return sk_transformer.fit_transform(X)


def _transform(X, sk_transformer):
    return sk_transformer.transform(X)


def fit_transform(sk_transformers, X):
    if X is None:
        return
    return reduce(_fit_transform, sk_transformers, X)


def transform(sk_transformers, X):
    if X is None:
        return
    return reduce(_transform, sk_transformers, X)


class Pipeline:
    """Preprocessing pipeline for tabular data."""

    def __init__(self, cts_cols, cat_cols):
        self.cts_cols = cts_cols
        self.cat_cols = cat_cols

    def build(self, cts_transforms: List[_sk_prep],
              cat_transforms: List[_sk_prep]):
        self.cts_transforms = cts_transforms if cts_transforms else []
        self.cat_transforms = cat_transforms if cat_transforms else []
        return self

    def fit_transform(self, X):
        X_cts = fit_transform(self.cts_transforms, get_rows(X, self.cts_cols))
        X_cat = fit_transform(self.cat_transforms, get_rows(X, self.cat_cols))
        X_concat = clean_null_from_list([X_cts, X_cat])
        return np.concatenate(X_concat, axis=-1)

    def transform(self, X):
        X_cts = transform(self.cts_transforms, get_rows(X, self.cts_cols))
        X_cat = transform(self.cat_transforms, get_rows(X, self.cat_cols))
        X_concat = clean_null_from_list([X_cts, X_cat])
        return np.concatenate(X_concat, axis=-1)

    @classmethod
    def from_config(cls, config):
        pipeline = cls(config.cts_cols, config.cat_cols)
        return pipeline.build(config.cts_transforms, config.cts_transforms)
