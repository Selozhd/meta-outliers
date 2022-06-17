"""Unittests for training and evaluation scripts."""

import unittest

import numpy as np
import tensorflow as tf

from outliers.autoencoder import AutoencoderAnomaly, get_autoencoder
from outliers.preprocesor import sample_with_anomalies
from outliers.training import TrainConfig, Trainer


def get_random_states(n_state, seed=42):
    """Repeatable random states from a single seed."""
    rng = np.random.default_rng(seed)
    seeds = rng.integers(0, 1000000, n_state)
    return [np.random.default_rng(i) for i in seeds]


class MockDataReader:
    """Mock of `outliers.data_reader.get()` result."""

    names = ['mock', 'mock_data']

    def __init__(self, name=None, seed=None):
        self.name = "MockedData"
        self.seed = seed if seed else np.random.randint(0, 100)
        self.rng1, self.rng2 = get_random_states(2, seed=self.seed)

    @property
    def coltypes(self):
        cts_cols = range(8)
        cat_cols = None
        return cts_cols, cat_cols

    def load_train(self, test_size=None):
        x = self.rng1.normal(0, 1, (100, 8))
        y = self.rng2.integers(0, 2, 100)
        return x, y


def _trainer_report_summary(trainer):
    return np.array([trainer.report.f1_score, trainer.report.accuracy,
            trainer.report.precision, trainer.report.auc])


class TestTrainer(unittest.TestCase):

    def setUp(self) -> None:
        self.train_config = TrainConfig(
            model=AutoencoderAnomaly,
            model_kwargs={'alpha': 2},
            autoencoder=get_autoencoder([8, 4, 1]),
            sampler=sample_with_anomalies(0, 1, percentage=0.2),
            seed=123,  # This seed is not used
            cts_transforms=[],
            cat_transforms=[],
            loss=tf.losses.MSE,
            optimizer=tf.optimizers.SGD(),
            metrics=[],
            callbacks=[],
            batch_size=100,
            epochs=1,
        )

    def _run_trainer(self, trainer, train_cfg):
        trainer.create_training_data(MockDataReader(seed=trainer.seed),
                                     sampler=train_cfg.sampler)
        trainer.preprocess(cts_transforms=train_cfg.cts_transforms,
                           cat_transforms=train_cfg.cat_transforms)
        trainer.build_model(autoencoder=train_cfg.autoencoder,
                            model=train_cfg.model,
                            model_kwargs=train_cfg.model_kwargs,
                            metrics=train_cfg.metrics,
                            optimizer=train_cfg.optimizer,
                            loss=train_cfg.loss)
        trainer.train(batch_size=train_cfg.batch_size,
                      epochs=train_cfg.epochs,
                      callbacks=train_cfg.callbacks,
                      validation_data=(trainer.X, trainer.y))
        trainer.evaluate()
        return trainer

    def test_reproducibility(self):
        trainer1 = Trainer(name='MockTrainer1', seed=123)
        trainer2 = Trainer(name='MockTrainer2', seed=123)
        trainer1 = self._run_trainer(trainer1, self.train_config)
        trainer2 = self._run_trainer(trainer2, self.train_config)
        report1 = _trainer_report_summary(trainer1)
        report2 = _trainer_report_summary(trainer2)
        self.assertAlmostEqual(np.sum(report1 - report2), 0, places=3)
        trainer1.delete_module()
        trainer2.delete_module()