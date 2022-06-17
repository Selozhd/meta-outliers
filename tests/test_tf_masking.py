"""Unittests for the Mask2d and ApplyMask2d tf.keras layers."""

import unittest
import pytest

import tensorflow as tf
import numpy as np

from outliers.autoencoder import Mask2d, ApplyMask2d


def relu(x):
    return x*(x > 0)


class MaskTestLayer(tf.keras.layers.Layer):
    """Testing layer that asserts input mask in `.call()`."""

    def __init__(self, mask_value=0., expected_mask=None, **kwargs):
        super(MaskTestLayer, self).__init__(**kwargs)
        self.supports_masking = True
        self.mask_value = mask_value
        self.expected_mask = expected_mask

    def call(self, inputs, mask=None):
        if mask is None:
            assert self.expected_mask is None
        else:
            assert np.all(mask == self.expected_mask)
        return inputs


class TestMasking(unittest.TestCase):

    def setUp(self):
        self.data = np.random.normal(0, 1, (100, 3))

    def _mock_regression(self, ndim, expected_mask):
        model = tf.keras.Sequential([
            Mask2d(mask_value=0.),
            MaskTestLayer(expected_mask=expected_mask),
            tf.keras.layers.Dense(ndim),
            ApplyMask2d(),
        ])
        return model

    def test_mask_shape(self):
        data = relu(self.data)
        expected_mask = data > 0
        model = self._mock_regression(3, expected_mask)
        model(data)
    
    def test_mask_failure(self):
        data = relu(self.data)
        model = self._mock_regression(3, expected_mask=None)
        with pytest.raises(AssertionError):
            model(data)
    
    def test_masked_output(self):
        data = relu(self.data)
        expected_mask = data > 0
        model = self._mock_regression(3, expected_mask)
        dense_layer = model.layers[-2]
        result = model(data)
        expected_result = dense_layer(data) * expected_mask
        assert np.array_equal(result, expected_result)
    
    def test_unmasked_output(self):
        data = self.data
        expected_mask = data != 0
        model = self._mock_regression(3, expected_mask)
        dense_layer = model.layers[-2]
        result = model(data)
        expected_result = dense_layer(data)
        assert np.array_equal(result, expected_result)