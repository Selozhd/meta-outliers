import unittest
import pytest

import numpy as np

from outliers.preprocesor import sample_with_anomalies


def find_row_index(x, X):
    """Finds the row index of 1d array `x` in 2d array `X`."""
    return np.where([np.all(np.equal(x, row)) for row in X])[0][0]


class TestSampleWithAnomalies(unittest.TestCase):

    def setUp(self):
        self.X = np.random.normal(0, 1, (30, 4))
        self.y = np.concatenate([np.repeat(i, 10) for i in range(3)])

    def test_sample_with_percentage(self):
        sampler = sample_with_anomalies(0, 2, percentage=0.1)
        sample, labels = sampler(self.X, self.y)

        row_class = [self.y[find_row_index(row, self.X)] for row in sample]

        expected_labels = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
        expected_class = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2]
        self.assertEqual(sample.shape, (11, 4))
        assert np.all(np.equal(labels, expected_labels))
        assert np.all(np.equal(row_class, expected_class))

    def test_sample_with_ratio(self):
        sampler = sample_with_anomalies(1, 0, ratio=1.)
        sample, labels = sampler(self.X, self.y)

        row_class = [self.y[find_row_index(row, self.X)] for row in sample]

        expected_labels = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                           1, 1, 1, 1, 1, 1, 1, 1, 1, 1] # yapf: disable
        expected_class = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                          0, 0, 0, 0, 0, 0, 0, 0, 0, 0] # yapf: disable
        self.assertEqual(sample.shape, (20, 4))
        assert np.all(np.equal(labels, expected_labels))
        assert np.all(np.equal(row_class, expected_class))

    def test_errors(self):
        with pytest.raises(ValueError):
            sampler = sample_with_anomalies(1, 0, percentage=0.2, ratio=1.)
        with pytest.raises(ValueError):
            sampler = sample_with_anomalies(1, 0, percentage=None, ratio=None)
        with pytest.raises(ValueError):
            sampler = sample_with_anomalies(1, 0, percentage=-.1, ratio=None)
        with pytest.raises(ValueError):
            sampler = sample_with_anomalies(1, 0, percentage=1.1, ratio=None)
        with pytest.raises(ZeroDivisionError):
            sampler = sample_with_anomalies(1, 0, percentage=1, ratio=None)
        with pytest.raises(ValueError):
            sampler = sample_with_anomalies(1, 0, percentage=None, ratio=-1)
        with pytest.raises(ValueError):
            sampler = sample_with_anomalies(1, [0, 2], ratio=0.1, p=[3, 3, 1])
        with pytest.raises(ValueError):
            sampler = sample_with_anomalies(1, [0, 2], ratio=0.1, p=[1])
        # Should NOT raise an Error
        sampler = sample_with_anomalies(1, 'all', percentage=None, ratio=3)

    def test_sample_with_no_anomalies(self):
        sampler = sample_with_anomalies(0, 1, percentage=0)
        sample, labels = sampler(self.X, self.y)

        row_class = [self.y[find_row_index(row, self.X)] for row in sample]

        expected_labels = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        expected_class = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.assertEqual(sampler.ratio, 0)
        self.assertEqual(sample.shape, (10, 4))
        assert np.all(np.equal(labels, expected_labels))
        assert np.all(np.equal(row_class, expected_class))

    def test_sample_from_all(self):
        sampler = sample_with_anomalies(1, 'all', ratio=0.6)
        sample, labels = sampler(self.X, self.y)

        row_class = [self.y[find_row_index(row, self.X)] for row in sample]
        n_normals = np.sum(np.equal(row_class, 1))
        n_anomalies = np.sum(np.not_equal(row_class, 1))

        expected_labels = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
        self.assertEqual(sample.shape, (16, 4))
        self.assertEqual(n_normals, 10)
        self.assertEqual(n_anomalies, 6)
        self.assertIn(0, row_class)
        self.assertIn(2, row_class)
        assert np.all(np.equal(labels, expected_labels))

    def test_sample_with_multiple_labels(self):
        sampler = sample_with_anomalies(1, [0, 2], ratio=0.5, p=[0, 1])
        sample, labels = sampler(self.X, self.y, size=8)

        row_class = [self.y[find_row_index(row, self.X)] for row in sample]
        n_normals = np.sum(np.equal(row_class, 1))
        n_anomalies = np.sum(np.not_equal(row_class, 1))

        expected_labels = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]
        expected_class = [1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2]
        self.assertEqual(sample.shape, (12, 4))
        self.assertEqual(n_normals, 8)
        self.assertEqual(n_anomalies, 4)
        assert np.all(np.equal(labels, expected_labels))
        assert np.all(np.equal(row_class, expected_class))

    def test_sample_with_seed(self):
        sampler1 = sample_with_anomalies(0, 1, percentage=0.2)
        sampler2 = sample_with_anomalies(0, 1, percentage=0.2)
        x1, y1 = sampler1(self.X, self.y, seed=54)
        x2, y2 = sampler2(self.X, self.y, seed=54)
        assert np.all(np.equal(x1, x2))
        assert np.all(np.equal(y1, y2))

    def test_different_seeds(self):
        sampler1 = sample_with_anomalies(0, 1, percentage=0.2)
        sampler2 = sample_with_anomalies(0, 1, percentage=0.2)
        x1, y1 = sampler1(self.X, self.y, seed=54)
        x2, y2 = sampler2(self.X, self.y, seed=24)
        # We still expect y's to be the same
        assert np.any(np.not_equal(x1, x2))
        assert np.all(np.equal(y1, y2))


if __name__ == '__main__':
    unittest.main()