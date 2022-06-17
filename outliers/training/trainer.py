"""Comprehensive training and evaluation scripts."""

import logging
import pprint
import re
from pathlib import Path

import joblib
import numpy as np

from constants import EXPERIMENTS_DIR
from order_statistics import scoring
from outliers import data_reader
from outliers.classification_utils import get_classification_report
from outliers.plotting import plot_anomalies, plot_metric, plot_train_metric
from outliers.preprocesor import Pipeline, pd_wrapper, sample_with_anomalies
from outliers.training.config import AlgorithmType
from outliers.training.summary import (get_md_summary, get_tf_config,
                                       preprocess_pipeline_summary,
                                       summarize_train_config)
from outliers.utils import (delete_directory_tree, get_logger,
                            join_list_of_dict, json_to_string, name_of,
                            write_json_file, write_text_file)

logger = logging.getLogger(__name__)
begins_with_val = re.compile('^(val_)')


def is_validation_metric(metric_name):
    match = begins_with_val.match(metric_name)
    return match is not None


def get_path(path):
    if not path.exists():
        path.mkdir()
    return path


class ModulePath:
    """Stores the paths for all directories within an experiment."""

    def __init__(self, name) -> None:
        self.name = name
        self._experiment_dir = get_path(Path(EXPERIMENTS_DIR)/self.name)

    @property
    def experiment_dir(self):
        return get_path(self._experiment_dir)

    @property
    def models_dir(self):
        return get_path(self._experiment_dir/'models')

    @property
    def plotting_dir(self):
        return get_path(self._experiment_dir/'plots')

    @property
    def evals_dir(self):
        return get_path(self._experiment_dir/'evals')

    @property
    def trained_models(self):
        return [m.name for m in self.models_dir.iterdir()]

    @property
    def algorithm(self):
        cands = [AlgorithmType.from_name(name) for name in self.trained_models]
        return [cand for cand in cands if isinstance(cand, AlgorithmType)][0]

    def delete_module(self):
        delete_directory_tree(self._experiment_dir)


class Trainer(ModulePath):
    """Configurable script for training.
    
    Attributes:
        name: Name of the directory under `EXPERIMENTS_DIR` to save the results.
        config: Optional dictionary containing all model parameters.
            Possible to use `outliers.training.TrainConfig.to_dict()`.
        seed: Integer seed for reproducibility.
    """

    def __init__(self, name, config=None, seed=None):
        super(Trainer, self).__init__(name)
        self.logger_file = self.experiment_dir/'train_logs'
        self.logger = get_logger(self.logger_file)
        self.train_config = config
        self.seed = seed

    def create_training_data(self, dataset_name, sampler=None, **kwargs):
        self.logger.info(f'Dataset Name: {dataset_name}')
        self.data_reader = data_reader.get(dataset_name)
        images, labels = self.data_reader.load_train(**kwargs)
        self.logger.info(f'Dataset shape: {images.shape}')
        if sampler is not None:
            self.logger.info(
                'Sampling Started with parameters: \n\t {params}'.format(
                    params=sampler.get_params()))
            self.X, self.y = pd_wrapper(sampler)(images, labels, seed=self.seed)
            self.sampler = sampler
            self.logger.info('Sampling Finished')
        else:
            self.logger.info('No sampler is given. skipping ...')
            self.X, self.y = images, labels
            self.sampler = None
        return self

    def preprocess(self, cts_transforms, cat_transforms):
        self.logger.info('Preprocess Started')
        cts_cols, cat_cols = self.data_reader.coltypes
        self.pipeline = Pipeline(cts_cols,
                                 cat_cols).build(cts_transforms, cat_transforms)
        self.X = self.pipeline.fit_transform(self.X)
        cts_summary = preprocess_pipeline_summary(self.pipeline.cts_transforms)
        cat_summary = preprocess_pipeline_summary(self.pipeline.cat_transforms)
        self.logger.info('Preprocess Pipeline:\n'
                         '- Continous Columns:\n'
                         f'{pprint.pformat(cts_summary)}\n'
                         '- Categorical Columns:\n'
                         f'{pprint.pformat(cat_summary)}\n')
        self.logger.info(f'Dataset shape after preprocess: {self.X.shape}')
        self.logger.info('Preprocess Finished')
        return self

    def build_model(self, autoencoder, model, model_kwargs, metrics, optimizer,
                    loss):
        autoencoder(self.X)  ## Initialize weights
        autoencoder.summary(print_fn=self.logger.info)
        model_config = json_to_string(model_kwargs)
        self.logger.info(f'{name_of(model)} Config: {model_config}')
        self.model = model(autoencoder, **model_kwargs)
        self.model_kwargs = model_kwargs
        optimizer_config = pprint.pformat(optimizer.get_config())
        self.logger.info(f'Optimizer Config: {optimizer_config}')
        self.model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
        return self

    def train(self, batch_size, epochs, callbacks=None, validation_data=None):
        self.callbacks = callbacks if callbacks is not None else []
        callback_config = join_list_of_dict(
            [get_tf_config(cb) for cb in self.callbacks])
        training_params = json_to_string({
            'batch_size': batch_size,
            'epochs': epochs,
            'callbacks': callback_config
        })
        self.logger.info(f'Training Params:\n {training_params}')
        self.logger.info('Training Started')
        self.history = self.model.custom_fit(self.X,
                                             self.y,
                                             batch_size=batch_size,
                                             epochs=epochs,
                                             callbacks=self.callbacks,
                                             validation_data=validation_data)
        self.logger.info('Training Finished')
        return self

    def _get_metrics(self):
        metrics = list(self.history.history.keys())
        metrics = [metric for metric in metrics if is_validation_metric(metric)]
        metrics = [begins_with_val.sub('', metric) for metric in metrics]
        return metrics

    def evaluate(self):
        report = get_classification_report(y_true=self.y,
                                           y_scores=self.model.training_scores,
                                           threshold=self.model.threshold,
                                           name=name_of(self.model))
        report.plot_roc_curve(self.plotting_dir/'roc_curve')
        self.report = report
        self.logger.info(report.report())
        self.logger.info('Saving Plots..')
        metrics = self._get_metrics()
        for metric in metrics:
            plot_metric(self.history, metric,
                        self.plotting_dir/f'{metric}_plot')
            if 'kappa_threshold' in self.history.history.keys():
                plot_train_metric(self.history, 'kappa_threshold',
                                  self.plotting_dir/'kappa_threshold_plot')
        kwargs = {
            'threshold':
                self.model.threshold,
            'cut_off':
                scoring.get_cut_off_index(self.model.training_scores,
                                          self.model.threshold)
        }
        plot_anomalies(
            df=self.model.training_scores,
            predictions=self.model.training_scores > self.model.threshold,
            save=self.plotting_dir/'results_plot',
            **kwargs)
        return self

    def save(self):
        self.logger.info('Saving Models..')
        joblib.dump(self.pipeline, self.models_dir/'preprocessor')
        if self.sampler is not None:
            joblib.dump(self.sampler.get_params(), self.models_dir/'sampler')
        self.model.save(self.models_dir)
        summary = get_md_summary(self)
        write_text_file(summary, self.experiment_dir/'summary.md')
        if self.train_config is not None:
            cfg_summary = summarize_train_config(self.train_config)
            cfg_summary = write_json_file(cfg_summary,
                                          self.experiment_dir/'config.json')
        return self


class Evaluator(ModulePath):

    def __init__(self, name):
        super(Evaluator, self).__init__(name)
        self.logger_file = self.evals_dir/'eval_logs'
        self.logger = get_logger(self.logger_file)
        self.load_models()

    def load_models(self):
        self.logger.info("Loading models...")
        model = self.algorithm.get()
        self.model = model.load_from_path(self.models_dir)
        self.pipeline = joblib.load(self.models_dir/"preprocessor")
        sampler_path = self.models_dir/"sampler"
        if sampler_path.exists():
            sampler_config = joblib.load(sampler_path)
            self.sampler = sample_with_anomalies.from_config(sampler_config)
        else:
            self.sampler = None

    def eval(self, x_test, y_test, save_data=False):
        x_test = self.pipeline.transform(x_test)
        y_scores = self.model.predict_scores(x_test)
        outlier_label = self.sampler.label if self.sampler else 1
        y_test = np.asarray(y_test == outlier_label, dtype=np.int32)
        report = get_classification_report(y_test,
                                           y_scores,
                                           self.model.threshold,
                                           name=name_of(self.model))
        self.logger.info(report.report())
        report.plot_roc_curve(self.evals_dir/'roc_curve')
        if save_data:
            self.logger.info("Saving test data...")
            np.save(self.evals_dir/"x_test", np.asarray(x_test))
            np.save(self.evals_dir/"y_test", np.asarray(y_test))
            self.logger.info(report.report())
        return report
