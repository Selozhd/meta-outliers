"""Unittests for `tf` variants of `orderstats.distributions` scoring functions."""

import unittest

import numpy as np
import tensorflow as tf
from numpy import random
from order_statistics import distributions, scoring
from outliers.orderstats_anomaly import (calculate_asymptotic_stat,
                                         calculate_kappa, calculate_S_m,
                                         calculate_T_nm,
                                         calculate_unscaled_kappa,
                                         get_anomaly_scores, get_selection_vec,
                                         inverse_score_weights,
                                         moving_average_asymptotic_stat,
                                         moving_average_kappa,
                                         moving_average_unscaled_kappa,
                                         no_weights_above_threshold)

Epsilon = 1e-2


def get_orderstats_distibutions_functs(tf_distributions):
    names = [f.__name__ for f in tf_distributions]
    np_functions = [distributions.__dict__[name] for name in names]
    return np_functions


class TestTFScoring(unittest.TestCase):

    def setUp(self):
        N = 10000
        m = random.randint(N)
        tf_X = tf.sort(tf.abs(tf.random.normal(shape=(N,), dtype=tf.float32)))
        np_X = np.sort(np.abs(random.normal(0, 1, (N,)), dtype=np.float32))
        self.tf_X = tf_X
        self.np_X = np_X
        self.tf_args = ((N, m),
                        (tf_X, m),
                        (tf_X, m),
                        (tf_X, m),
                        (tf_X, m),
                        (tf_X, m),
                        (tf_X,),
                        (tf_X,),
                        (tf_X,),
                        )
        self.np_args = ((N, m),
                        (np_X, m),
                        (np_X, m),
                        (np_X, m),
                        (np_X, m),
                        (np_X, m),
                        (np_X,),
                        (np_X,),
                        (np_X,),
                        )

    def test_tf_np_modules(self):
        tf_distributions = [get_selection_vec,
                            calculate_S_m,
                            calculate_T_nm,
                            calculate_kappa,
                            calculate_unscaled_kappa,
                            calculate_asymptotic_stat,
                            moving_average_kappa,
                            moving_average_unscaled_kappa,
                            moving_average_asymptotic_stat,
                            ]
        np_distributions = get_orderstats_distibutions_functs(tf_distributions)
        
        result1 = [f(*args) for f, args in zip(tf_distributions, self.np_args)]
        result2 = [f(*args) for f, args in zip(tf_distributions, self.tf_args)]
        result3 = [f(*args) for f, args in zip(np_distributions, self.tf_args)]
        result4 = [f(*args) for f, args in zip(np_distributions, self.np_args)]

        compare_np_args = [np.nansum(i - j) for i, j in zip(result1, result4)]
        compare_tf_args = [np.nansum(i - j) for i, j in zip(result2, result3)]
        is_np_equal = np.all(np.less(compare_np_args, Epsilon))
        is_tf_equal = np.all(np.less(compare_tf_args, Epsilon))
        self.assertTrue(is_np_equal)
        self.assertTrue(is_tf_equal)
        
    def test_anomaly_scores(self):
        tf_diff = tf.subtract(get_anomaly_scores(self.tf_X),
                              scoring.get_anomaly_scores(self.tf_X))
        np_diff = np.subtract(get_anomaly_scores(self.np_X),
                               scoring.get_anomaly_scores(self.np_X))
        self.assertLess(np.nansum(tf_diff), Epsilon)
        self.assertLess(np.nansum(np_diff), Epsilon)