"""Unittests for AutoencoderAnomaly and autoencoder utils."""

import unittest
import pytest

import numpy as np
import tensorflow as tf
from outliers.autoencoder import (AutoencoderAnomaly, NotFittedModelError,
                                  get_autoencoder, get_autoencoder_shapes,
                                  get_lazy_dimensions, get_matrix_dimensions)


class Test_Utils(unittest.TestCase):

    def setUp(self):
        self.dims = [784, 256, 128, 32]
        self.model = get_autoencoder(self.dims)

    def test_autodims(self):
        lazy_dims = get_lazy_dimensions(self.dims)
        encoder, decoder = get_matrix_dimensions(self.dims)
        matrix_dims = [*encoder, *decoder]
        end_dims = [dim[1] for dim in matrix_dims]
        self.assertEqual(lazy_dims, end_dims)

    def test_weights(self):
        X = np.random.normal(0, 1, (50, 784))
        output = self.model(X)

        encoder, decoder = get_matrix_dimensions(self.dims)
        matrix_dims = [*encoder, *decoder]
        shapes = get_autoencoder_shapes(self.model)

        output_shape = tuple(output.shape)
        self.assertEqual(output_shape, (50, 784))
        self.assertEqual(matrix_dims, shapes)


class TestAutoencoderAnomaly(unittest.TestCase):

    def setUp(self):
        self.X = np.random.normal(0, 1, (128, 8))
        self.y = np.random.randint(0, 2, 128)
        self.autoencoder = get_autoencoder([8, 4, 1])

    def test_shape(self):
        model = AutoencoderAnomaly(self.autoencoder, 2)
        result = model(self.X)
        assert result.shape == self.X.shape

    def test_check_results_before_fit(self):
        model = AutoencoderAnomaly(self.autoencoder, 2)
        with pytest.raises(NotFittedModelError):
            model.training_results

    def test_anomaly_percentage(self):
        model = AutoencoderAnomaly(self.autoencoder, 2)
        model.compile(loss=tf.losses.mse, optimizer=tf.optimizers.Adam())
        model.custom_fit(X=self.X, y=self.y, epochs=1, batch_size=32, verbose=0)
        self.assertAlmostEqual(np.mean(model.training_results), 0.02, places=2)

    def test_predictions_equal_scores(self):
        model = AutoencoderAnomaly(self.autoencoder, 2)
        model.compile(loss=tf.losses.mse, optimizer=tf.optimizers.Adam())
        model.custom_fit(X=self.X, y=self.y, epochs=1, batch_size=32, verbose=0)
        preds = model.predict_scores(self.X)
        assert np.array_equal(preds, model.training_scores)