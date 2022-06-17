"""Utility functions for summarizing the details of a training session."""

import datetime

from outliers.utils import name_of

_JSONABLE = (list, dict, int, str, float, tuple, set)

_SUMMARY_FORMAT = '''
# Model Summary
    Date: {date}
    Data: {data}
    Sampling: {sampling}
## Configs
    Outlier Model: {model}
    Model Config: {model_config}
    Early Stopping: {early_stopping}
    Loss Function: {loss}
    Optimizer: {optimizer}
## Sucesss
    F1 score: {f1}
    AUC: {auc}
'''.strip('\n')


def _jsonnable_dict(dic):
    return {
        k: v if isinstance(v, _JSONABLE) else str(v) for k, v in dic.items()
    }


def preprocess_pipeline_summary(pipeline):
    return {name_of(obj): _jsonnable_dict(obj.get_params()) for obj in pipeline}


def get_tf_config(tf_obj):
    try:
        config = tf_obj.get_config()
    except:
        config = None
    return {name_of(tf_obj): config}


def _names_of(ls):
    return [name_of(i) for i in ls]


def get_md_summary(trainer):
    """Creates a concise readible summary of an experiment."""
    return _SUMMARY_FORMAT.format(
        date=datetime.datetime.now().strftime('%d %B %Y'),
        data=trainer.data_reader.name,
        sampling=trainer.sampler.get_params() if trainer.sampler else None,
        model=name_of(trainer.model),
        model_config=trainer.model_kwargs,
        early_stopping=_names_of(trainer.callbacks),
        loss=name_of(trainer.model.loss),
        optimizer=name_of(trainer.model.optimizer),
        f1=trainer.report.f1_score,
        auc=trainer.report.auc,
    )


def summarize_train_config(config_dict):
    """Comprehensive summary of train config in JSON format."""
    _float_or_str = lambda v: v if isinstance(v, str) else float(v)
    optimizer_config = config_dict.get('optimizer').get_config()
    sampler = config_dict.get("sampler")
    summary = {
        "model": {
            config_dict.get('model').name: config_dict.get('model_kwargs')
        },
        "sampler":
            sampler.get_params() if sampler else None,
        "cts_transforms":
            preprocess_pipeline_summary(config_dict['cts_transforms']),
        "cat_transforms":
            preprocess_pipeline_summary(config_dict['cat_transforms']),
        "loss":
            _jsonnable_dict(get_tf_config(config_dict['loss'])),
        "optimizer": {k: _float_or_str(v) for k, v in optimizer_config.items()},
        "metrics": [
            metric.get_config() for metric in config_dict.get('metrics')
        ],
        "callbacks": [get_tf_config(cb) for cb in config_dict.get('callbacks')],
        "batch_size":
            config_dict.get("batch_size"),
        "epochs":
            config_dict.get("epochs"),
    }
    return summary