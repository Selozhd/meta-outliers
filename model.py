"""Experiments and examples from the library."""

import datetime
import functools
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

from order_statistics import distributions
from order_statistics import utils as stat_utils
from outliers.autoencoder import *
from outliers.autoencoder_builder import AutoencoderParams, build_autoencoder
from outliers.classification_utils import get_classification_report
from outliers.cumulative_anomaly import *
from outliers.data_reader import *
from outliers.multigan_anomaly import *
from outliers.orderstats_anomaly import *
from outliers.plotting import *
from outliers.preprocesor import *
from outliers.training import AlgorithmType, TrainConfig, Trainer
from outliers.utils import write_json_file, write_text_file, read_file
from outliers.vae import VAEParams, build_vae


def get_experiment_name(model, data):
    time = datetime.datetime.now().strftime("%b%d-%H%M%S")
    return '_'.join((model, data, time))


def _get_path(path):
    if not path.exists():
        path.mkdir()
    return path


def normalize_image(x):
    return np.asarray(x, np.float32)/256.0


def partial_autoencoder(params):
    return functools.partial(build_autoencoder, **{"params": params})


def partial_vae(params):
    return functools.partial(build_vae, **{"params": params})


def read_results(filepath):
    filepath = Path(filepath)
    pyod_result = read_file(filepath / "pyod/results_test")
    files = [f for f in filepath.iterdir() if f.stem != 'pyod']
    os_result1, os_result2 = [read_file(f / "evals/test_roc") for f in files]
    print(pyod_result)
    print(os_result1)
    print(os_result2)


def get_mnist_model():
    model = tf.keras.Sequential([
        layers.Dense(256),
        layers.Dense(32),
        layers.Dense(256),
        layers.Dense(784),
    ])
    return model


def get_abu_model(dim):
    model = tf.keras.Sequential([
        layers.Dense(100),
        layers.Dense(50),
        layers.Dense(25),
        layers.Dense(50),
        layers.Dense(100),
        layers.Dense(dim),
    ])
    return model


def train_mnist(mnist_data, train_cfg):
    experiment_name = get_experiment_name(model=train_cfg.model.name,
                                          data="MNIST")
    trainer = Trainer(experiment_name, train_cfg.to_dict(), train_cfg.seed)
    trainer.create_training_data(mnist_data, sampler=train_cfg.sampler)
    trainer.preprocess(cts_transforms=train_cfg.cts_transforms,
                       cat_transforms=train_cfg.cat_transforms)
    trainer.build_model(autoencoder=train_cfg.autoencoder,
                        model=train_cfg.model.get(),
                        model_kwargs=train_cfg.model_kwargs,
                        metrics=train_cfg.metrics,
                        optimizer=train_cfg.optimizer,
                        loss=train_cfg.loss)
    trainer.train(batch_size=train_cfg.batch_size,
                  epochs=train_cfg.epochs,
                  callbacks=train_cfg.callbacks,
                  validation_data=(trainer.X, trainer.y))
    trainer.evaluate()
    trainer.save()
    return trainer


def train_builder(dataset_name, train_cfg):
    """Training autoencoder with lazy first layer."""
    experiment_name = get_experiment_name(model=train_cfg.model.name,
                                          data=dataset_name.upper())
    trainer = Trainer(experiment_name, train_cfg.to_dict(), train_cfg.seed)
    trainer.create_training_data(dataset_name, sampler=train_cfg.sampler)
    trainer.preprocess(cts_transforms=train_cfg.cts_transforms,
                       cat_transforms=train_cfg.cat_transforms)
    trainer.build_model(autoencoder=train_cfg.autoencoder(x=trainer.X),
                        model=train_cfg.model.get(),
                        model_kwargs=train_cfg.model_kwargs,
                        metrics=train_cfg.metrics,
                        optimizer=train_cfg.optimizer,
                        loss=train_cfg.loss)
    trainer.train(batch_size=train_cfg.batch_size,
                  epochs=train_cfg.epochs,
                  callbacks=train_cfg.callbacks,
                  validation_data=(trainer.X, trainer.y))
    trainer.evaluate()
    trainer.save()
    return trainer


@plot_utils.plotting_style(PLOTTING_STYLE)
def plot_values(values, ylabel="$\kappa$", title=''):
    """Plots a training metric of a model history."""
    plt.plot(values)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel('epoch')
    plt.show()


if False and __name__ == '__main__':
    for N in range(10):
        ##  MNIST Pipeline
        from pyod.models.auto_encoder import AutoEncoder
        sampler = sample_with_anomalies(N, 'all', percentage=0.01)
        pyod_path = _get_path(Path("experiments/MNIST"+ str(N)))
        func = ApplyFunction(normalize_image)
        kappa_stop = KappaThresholdStopping()
        cts = [func]
        cat = []

        trainer_config = TrainConfig(
            model=AlgorithmType.OrderStats,
            model_kwargs={'loss_weights': False},
            autoencoder=get_mnist_model(),
            sampler=sampler,
            seed=42,
            cts_transforms=cts,
            cat_transforms=cat,
            loss=tf.losses.MSE,
            optimizer=tf.optimizers.Adam(learning_rate=1e-3),
            metrics=[tf.metrics.AUC(), tf.metrics.Precision()],
            callbacks=[],
            batch_size=2991,
            epochs=100,
        )
        trainer = train_mnist('digits', trainer_config)
        config2 = trainer_config.to_dict()
        config2['model_kwargs'] = {'loss_weights': True}
        config2['callbacks'] = [kappa_stop]
        trainer_config2 = TrainConfig.from_dict(config2)
        trainer2 = train_mnist('digits', trainer_config2)

        params = {
            "hidden_neurons": [256, 32, 256],
            "hidden_activation": None,
            "output_activation": None,
            "dropout_rate": 0.,
            "l2_regularizer": 0.,
            "batch_size": trainer_config.batch_size,
            "validation_size": 0.,
            "contamination": 0.01
            }
        ae_pyod = AutoEncoder(preprocessing=False, **params)
        ae_pyod.fit(trainer.X)
        y_pyod = ae_pyod.predict_proba(trainer.X)[:, 1]
        report_pyod = get_classification_report(trainer.y, y_pyod, 0.5)

        ## Test Pipeline
        x_test, y_test = trainer.data_reader.load_test()
        x_test, y_test = trainer.sampler(x_test, y_test)
        x_test = normalize_image(x_test)
        y_scores = trainer.model.predict_scores(x_test)
        y_scores2 = trainer2.model.predict_scores(x_test)
        y_pyod = ae_pyod.predict_proba(x_test)[:, 1]
        report_trainer_test = get_classification_report(y_test, y_scores, trainer.model.threshold)
        report_trainer2_test = get_classification_report(y_test, y_scores2, trainer2.model.threshold)
        plot_anomalies(np.sort(y_scores), np.sort(y_scores) > trainer.model.threshold,
                    save=trainer.evals_dir / "outliers_in_test",
                    **{"threshold": trainer.model.threshold,
                        "cut_off": scoring.get_cut_off_index(np.sort(y_scores) ,trainer.model.threshold)})
        plot_anomalies(np.sort(y_scores2), np.sort(y_scores2) > trainer2.model.threshold,
                    save=trainer2.evals_dir / "outliers_in_test",
                    **{"threshold": trainer2.model.threshold,
                        "cut_off": scoring.get_cut_off_index(np.sort(y_scores), trainer.model.threshold)})
        report_pyod_test = get_classification_report(y_test, y_pyod, 0.5)

        ex_path = _get_path(pyod_path / "pyod")
        ae_pyod.model_.save(ex_path / "autoencoder")
        train_rep = report_pyod.report()
        test_rep = report_pyod_test.report()
        report_pyod.plot_roc_curve( ex_path / "roc_train")
        report_pyod_test.plot_roc_curve(ex_path / "roc_test")
        write_text_file(train_rep, ex_path / "results_train")
        write_text_file(test_rep, ex_path / "results_test")
        write_json_file(params, ex_path / "config.json")
        report_trainer_test.plot_roc_curve(trainer.evals_dir / "test_result")
        report_trainer2_test.plot_roc_curve(trainer2.evals_dir / "test_result")
        write_text_file(report_trainer_test.report(), trainer.evals_dir / "test_roc")
        write_text_file(report_trainer2_test.report(), trainer2.evals_dir / "test_roc")

        print(report_trainer_test)
        print(report_trainer2_test)
        print(report_pyod_test)

if False and __name__ == "__main__":
    ##  MUSK Pipeline
    from pyod.models.auto_encoder import AutoEncoder
    from sklearn.preprocessing import StandardScaler
    musk_path = _get_path(Path("experiments/Musk"))
    kappa_stop = KappaThresholdStopping()
    sampler = sample_with_anomalies(0, 1, percentage=0.01)
    scaler = StandardScaler()
    cts = [scaler]
    cat = []

    params = AutoencoderParams.from_dict({
        "hidden_neurons": [128, 64, 32, 64, 128],
        "hidden_activation": 'relu',
        "output_activation": None,
        "dropout_rate": 0.2,
        "l2_regularizer": 0.1
    })

    trainer_config = TrainConfig(
        model=AlgorithmType.OrderStats,
        model_kwargs={'loss_weights': False},
        autoencoder=partial_autoencoder(params),
        sampler=sampler,
        seed=42,
        cts_transforms=cts,
        cat_transforms=cat,
        loss=tf.losses.mse,
        optimizer=tf.optimizers.Adam(),
        metrics=[tf.metrics.AUC(), tf.metrics.Precision()],
        callbacks=[],
        batch_size=32,
        epochs=100)
    trainer = train_builder('musk', trainer_config)
    config2 = trainer_config.to_dict()
    config2['model_kwargs'] = {'loss_weights': True}
    config2['callbacks'] = [kappa_stop]
    trainer_config2 = TrainConfig.from_dict(config2)
    trainer2 = train_builder('musk', trainer_config2)

    ae_pyod = AutoEncoder(preprocessing=False, **params.to_dict())
    ae_pyod.fit(trainer.X)
    y_pyod = ae_pyod.predict_proba(trainer.X)[:, 1]
    report_pyod = get_classification_report(trainer.y, y_pyod, 0.5)

    ## Test Pipeline
    x_test, y_test = trainer.data_reader.load_test()
    x_test, y_test = sampler(x_test, y_test)
    y_scores = trainer.model.predict_scores(x_test)
    y_scores2 = trainer2.model.predict_scores(x_test)
    y_pyod = ae_pyod.predict_proba(x_test)[:, 1]
    report_trainer_test = get_classification_report(y_test, y_scores, trainer.model.threshold)
    report_trainer2_test = get_classification_report(y_test, y_scores2, trainer2.model.threshold)
    plot_anomalies(np.sort(y_scores), np.sort(y_scores) > trainer.model.threshold,
                save=trainer.evals_dir / "outliers_in_test",
                **{"threshold": trainer.model.threshold,
                    "cut_off": scoring.get_cut_off_index(np.sort(y_scores), trainer.model.threshold)})
    plot_anomalies(np.sort(y_scores2), np.sort(y_scores2) > trainer2.model.threshold,
                save=trainer2.evals_dir / "outliers_in_test",
                **{"threshold": trainer2.model.threshold,
                    "cut_off": scoring.get_cut_off_index(np.sort(y_scores), trainer.model.threshold)})
    report_pyod_test = get_classification_report(y_test, y_pyod, 0.5)

    ex_path = _get_path(musk_path / "pyod")
    ae_pyod.model_.save(ex_path / "autoencoder")
    train_rep = report_pyod.report()
    test_rep = report_pyod_test.report()
    report_pyod.plot_roc_curve( ex_path / "roc_train")
    report_pyod_test.plot_roc_curve(ex_path / "roc_test")
    write_text_file(train_rep, ex_path / "results_train")
    write_text_file(test_rep, ex_path / "results_test")
    write_json_file(params.to_dict(), ex_path / "config.json")
    report_trainer_test.plot_roc_curve(trainer.evals_dir / "test_result")
    report_trainer2_test.plot_roc_curve(trainer2.evals_dir / "test_result")
    write_text_file(report_trainer_test.report(), trainer.evals_dir / "test_roc")
    write_text_file(report_trainer2_test.report(), trainer2.evals_dir / "test_roc")

    print(report_trainer_test)
    print(report_trainer2_test)
    print(report_pyod_test)