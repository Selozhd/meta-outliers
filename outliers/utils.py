import inspect
import json
import logging
import os
import pickle
import shutil
from collections import ChainMap


def read_file(filename, encoding="utf-8"):
    """Read text from a file."""
    try:
        with open(filename, encoding=encoding) as f:
            return f.read()
    except FileNotFoundError:
        raise ValueError("File '{}' does not exist.".format(filename))


def read_json_file(filename):
    """Read json from a file."""
    try:
        with open(filename, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    except ValueError as e:
        raise ValueError("Failed to read json from '{}'. Error: "
                         "{}".format(os.path.abspath(filename), e))


def json_to_string(js_dic, **kwargs):
    """Converts a dictionary to a string."""
    indent = kwargs.pop("indent", 2)
    ensure_ascii = kwargs.pop("ensure_ascii", False)
    return json.dumps(js_dic,
                      indent=indent,
                      ensure_ascii=ensure_ascii,
                      **kwargs)


def write_text_file(content, filepath, encoding="utf-8", append=False):
    """Writes text to a file.
    Args:
        content: The content to write.
        filepath: The path to which the content should be written.
        encoding: The encoding which should be used.
        append: Whether to append to the file or to truncate the file.
    """
    mode = "a" if append else "w"
    with open(filepath, mode, encoding=encoding) as file:
        file.write(content)


def write_json_file(js_obj, filepath, **kwargs):
    """Writes object to a json_file."""
    json_string = json_to_string(js_obj, **kwargs)
    write_text_file(json_string, filepath)


def pickle_load(filepath):
    """Loads an object from pickle."""
    with open(filepath, "rb") as f:
        return pickle.load(f)


def pickle_dump(filepath, obj):
    """Saves object to a pickle."""
    with open(filepath, "wb") as f:
        pickle.dump(obj, f)


def delete_directory_tree(filepath):
    shutil.rmtree(filepath)


def get_logger(out_file=None):
    logger = logging.getLogger()
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S")
    logger.handlers = []
    logger.setLevel(logging.INFO)

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)

    if out_file is not None:
        fh = logging.FileHandler(out_file)
        fh.setFormatter(formatter)
        fh.setLevel(logging.INFO)
        logger.addHandler(fh)
    return logger


def ordered(obj):
    """Orders the items of lists and dictionaries."""
    if isinstance(obj, dict):
        return sorted((k, ordered(v)) for k, v in obj.items())
    if isinstance(obj, list):
        return sorted(ordered(x) for x in obj)
    else:
        return obj


def lazy_property(function):
    """Allows to avoid recomputing a property over and over.

    The result gets stored in a local var. Computation of the property
    will happen once, on the first call of the property. All
    succeeding calls will use the value stored in the private property."""

    attr_name = "_lazy_" + function.__name__

    @property
    def _lazyprop(self):
        if not hasattr(self, attr_name):
            setattr(self, attr_name, function(self))
        return getattr(self, attr_name)

    return _lazyprop


def name_of(obj):
    """Returns the name of function or class."""
    class_name = type(obj).__name__
    if class_name in ["function", "type"]:
        return obj.__name__
    else:
        return class_name


def arguments_of(function):
    """Returns the parameters of the function `function` t."""
    return list(inspect.signature(function).parameters.keys())


def clean_null_values(dic):
    """Cleans the entries with value None in a given dictionary."""
    return {key: value for key, value in dic.items() if value is not None}


def clean_null_from_list(ls):
    """Cleans elements with value None in a given list."""
    return [i for i in ls if i is not None]


def join_list_of_dict(ls):
    """Joins a list of dictionaries in a single dictionary."""
    return dict(ChainMap(*ls))


def inverse_dic_lookup(dic, item):
    """Looks up dictionary key using value."""
    return next(key for key, value in dic.items() if value == item)
