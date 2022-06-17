import unittest
import joblib

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn import decomposition, preprocessing

from constants import DATA_DIR
from outliers.utils import read_json_file, pickle_dump
from outliers.preprocesor import Centralizor, Pipeline


def _downsample(dataframe, colname, value, rate):
    """Down-sample from dataframe based on colname and rate."""
    data_normal = dataframe[dataframe[colname] != value]
    data_sample = dataframe[dataframe[colname] == value]
    N_anom = len(data_sample)
    data_sample = data_sample.sample(int(N_anom*rate + 0.5),
                                     replace=False,
                                     random_state=42)
    data = pd.concat([data_normal, data_sample], ignore_index=True)
    return data


class _Preprocessor:
    """Custom preprocessor for unit testing."""

    def __init__(self, continuous_cols, discrete_cols):
        self.continuous_cols = continuous_cols
        self.discrete_cols = discrete_cols

    def fit_transform(self, df):
        self.whitener = decomposition.PCA(whiten=True)
        self.onehot_encoder = preprocessing.OneHotEncoder(sparse=False)
        self.minmax_scaler = preprocessing.MinMaxScaler()
        cts_data = df[self.continuous_cols]
        cat_data = df[self.discrete_cols]
        self.means = cts_data.mean()
        cts_data = cts_data - self.means
        cts_data = self.whitener.fit_transform(cts_data)
        cts_data = self.minmax_scaler.fit_transform(cts_data)
        cat_data = self.onehot_encoder.fit_transform(cat_data)
        return np.concatenate([cts_data, cat_data], axis=-1)

    def transform(self, df):
        cts_data = df[self.continuous_cols]
        cat_data = df[self.discrete_cols]
        cts_data = cts_data - self.means
        cts_data = self.whitener.transform(cts_data)
        cts_data = self.minmax_scaler.transform(cts_data)
        cat_data = self.onehot_encoder.transform(cat_data)
        return np.concatenate([cts_data, cat_data], axis=-1)


class Test_Preprocess(unittest.TestCase):
    """Tests for Pipeline."""

    def setUp(self):
        colnames = read_json_file(DATA_DIR + "colnames.json")
        data = pd.read_csv(DATA_DIR + "kddcup.data_10_percent_corrected",
                           header=0,
                           names=colnames.keys(),
                           nrows=100000)
        data["is_normal"] = (data["is_normal"] == "normal.").astype(int)
        data = _downsample(data, "is_normal", 1, 0.1)
        self.cts_cols = [k for k, v in colnames.items() if v == "continuous"]
        self.cat_cols = [k for k, v in colnames.items() if v == "symbolic"]
        self.data_train, self.data_test = train_test_split(data,
                                                           random_state=42)

    def _get_transforms(self):
        centre = Centralizor()
        whitener = decomposition.PCA(whiten=True)
        onehot_encoder = preprocessing.OneHotEncoder(sparse=False)
        minmax_scaler = preprocessing.MinMaxScaler()
        cat_trans = [onehot_encoder]
        cts_trans = [centre, whitener, minmax_scaler]
        return cts_trans, cat_trans

    def test_fit_transform(self):
        preprocessor = _Preprocessor(self.cts_cols, self.cat_cols)
        pipeline = Pipeline(self.cts_cols, self.cat_cols)
        pipeline = pipeline.build(*self._get_transforms())

        train1 = preprocessor.fit_transform(self.data_train)
        train2 = pipeline.fit_transform(self.data_train)
        test1 = preprocessor.transform(self.data_test)
        test2 = pipeline.transform(self.data_test)
        self.assertEqual(np.sum(train1 - train2), 0)
        self.assertEqual(np.sum(test1 - test2), 0)

    def test_saving(self):
        pipeline = Pipeline(self.cts_cols, self.cat_cols)
        pipeline = pipeline.build(*self._get_transforms())
        train1 = pipeline.fit_transform(self.data_train)
        joblib.dump(pipeline, 'tests/pipeline.bin')
        pipeline2 = joblib.load('tests/pipeline.bin')
        train2 = pipeline2.fit_transform(self.data_train)
        self.assertEqual(np.sum(train1 - train2), 0)


class test_DataHandling(unittest.TestCase):
    """Tests the behaviour of `tf.data.Dataset`."""

    def setUp(self):
        self.N = 150
        self.batch_size = 32
        X = np.random.normal(0, 1, (self.N, 99))
        idx = np.arange(self.N).reshape(-1, 1)
        self.X = np.hstack([idx, X])
        self.data = tf.data.Dataset.from_tensor_slices(self.X)

    def test_shuffle(self):
        data_shuffle = self.data.shuffle(self.N)
        idx = [int(row[0]) for row in list(data_shuffle)]
        self.assertLess(np.mean(idx == np.arange(self.N)), 0.1)
        self.assertTrue(np.all(np.sort(idx) == np.arange(self.N)))

    def test_batch(self):
        data_batch = self.data.batch(self.batch_size)
        batches = list(data_batch)
        idx = batches[0][:, 0]
        self.assertEqual(len(batches), np.ceil(self.N/self.batch_size))
        self.assertTrue(np.all(idx == np.arange(self.batch_size)))

    def test_reshuffle(self):
        data = self.data.shuffle(self.N, reshuffle_each_iteration=True)
        data = data.batch(self.batch_size)
        batches1 = list(data)
        batch_idx1 = batches1[0][:, 0]
        batches2 = list(data)
        batch_idx2 = batches2[0][:, 0]
        row_id_match = np.sort(batch_idx1) == np.sort(batch_idx2)
        self.assertLess(np.mean(row_id_match), 0.1)