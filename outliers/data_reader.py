"""Data sets for anomaly experiments.

References for data sets:
    [1] http://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html
    [2] H. Xiao, K. Rasul, and R. Vollgraf, ‘‘Fashion-MNIST: A novel image dataset for
        benchmarking machine learning algorithms, 2017, arXiv:1708.07747. [Online].
        Available: http://arxiv.org/abs/1708.07747
    [3] http://xudongkang.weebly.com/data-sets.html
"""

import gzip
import os

import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing import image as keras_image

from constants import DATA_DIR
from outliers.utils import read_json_file


def array_to_image(X):
    if X.ndim == 2:
        X = np.expand_dims(X, -1)
    return keras_image.array_to_img(X)


def load_mnist(dirname, kind='train'):
    """Loads MNIST data set.
    Args:
        dirname: Name of the directory which contains the .gz files.
        kind: either 'train' or 'test'.
    Returns:
        images: 28px x 28px of images, in np.array of shape (# IMAGES, 784).
        labels: Digit drawn in each image, in np.array of shape (# IMAGES,).
    """
    path = DATA_DIR + dirname
    labels_path = os.path.join(path, f"{kind}-labels-idx1-ubyte.gz")
    images_path = os.path.join(path, f"{kind}-images-idx3-ubyte.gz")

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784)
    return images, labels


def load_kddcup(nrows):
    colnames = read_json_file(DATA_DIR + "colnames.json")
    data = pd.read_csv(DATA_DIR + "kddcup.data_10_percent_corrected",
                       header=0,
                       names=colnames.keys(),
                       nrows=nrows)
    data["is_normal"] = (data["is_normal"] == "normal.").astype(int)
    cts_cols = [k for k, v in colnames.items() if v == "continuous"]
    cat_cols = [k for k, v in colnames.items() if v == "symbolic"]
    return data, cts_cols, cat_cols


def load_abu(kind='airport', no=1):
    path = DATA_DIR + 'ABU'
    data = loadmat(os.path.join(path, f'abu-{kind}-{no}.mat'))
    X = data['data']
    y = data['map']
    return X, y


def abu_to_PIL(X):
    return [array_to_image(X[:, :, i]) for i in range(X.shape[2])]


def load_gan_dataset(filename, seed=None):
    filepath = DATA_DIR + 'gan_datasets/' + filename
    data = pd.read_table(filepath, sep=',', header=None)
    data = data.sample(frac=1, random_state=seed).reset_index(drop=True)
    id = data.pop(0)
    y = data.pop(1)
    data_x = data.to_numpy()
    data_id = id.values
    data_y = y.values
    return data_x, data_y, data_id


class MNISTReader:

    names = ['digits', 'mnist_digits', 'fashion', 'mnist_fashion']

    def __init__(self, name, seed=None):
        if name.lower() in ['digits', 'mnist_digits']:
            self.name = 'MNIST_digits'
        elif name.lower() in ['fashion', 'mnist_fashion']:
            self.name = 'MNIST_fashion'
        else:
            raise ValueError("MNIST data name not recognized.")
        self.seed = seed

    @property
    def coltypes(self):
        return range(784), None

    def load_train(self, test_size=None):
        return load_mnist(self.name, kind="train")

    def load_test(self, test_size=None):
        return load_mnist(self.name, kind="t10k")


class KDDCupReader:

    names = ['kdd', 'kddcup']

    def __init__(self, name=None, seed=None):
        self.name = "KDDCup"
        self.seed = seed
        self.colnames = read_json_file(DATA_DIR + "colnames.json")

    @property
    def coltypes(self):
        cts_cols = [k for k, v in self.colnames.items() if v == "continuous"]
        cat_cols = [k for k, v in self.colnames.items() if v == "symbolic"]
        return cts_cols, cat_cols

    def load_train(self, nrows, test_size=0.2):
        x = pd.read_csv(DATA_DIR + "kddcup.data_10_percent_corrected",
                        header=0,
                        names=self.colnames.keys(),
                        nrows=nrows)
        y = np.asarray(x["is_normal"] != "normal.", dtype=np.int32)
        x = x.drop(columns="is_normal")
        x_train, _, y_train, _ = train_test_split(x,
                                                  y,
                                                  test_size=test_size,
                                                  random_state=self.seed)
        return x_train, y_train

    def load_test(self, nrows, test_size=0.2):
        x = pd.read_csv(DATA_DIR + "kddcup.data_10_percent_corrected",
                        header=0,
                        names=self.colnames.keys(),
                        nrows=nrows)
        y = np.asarray(x["is_normal"] != "normal.", dtype=np.int32)
        x = x.drop(columns="is_normal")
        _, x_test, _, y_test = train_test_split(x,
                                                y,
                                                test_size=test_size,
                                                random_state=self.seed)
        return x_test, y_test


class UCIReader:

    names = ['annthyroid', 'onecluster', 'spambase', 'waveform', 'wdbc']

    def __init__(self, name, seed=None):
        if name.lower() in ['annthyroid']:
            self.name = 'Annthyroid'
        elif name.lower() in ['onecluster']:
            self.name = 'onecluster'
        elif name.lower() in ['spambase']:
            self.name = 'SpamBase'
        elif name.lower() in ['waveform']:
            self.name = 'Waveform'
        elif name.lower() in ['wdbc']:
            self.name = 'WDBC'
        else:
            raise ValueError("Data name not recognized.")
        self.seed = seed

    @property
    def coltypes(self):
        return range(self.shape[-1]), None

    def load_train(self, test_size=0.2):
        x, y, self.id = load_gan_dataset(self.name, self.seed)
        self.shape = x.shape
        y = np.asarray(y != 'nor', np.int32)
        x_train, _, y_train, _ = train_test_split(x,
                                                  y,
                                                  test_size=test_size,
                                                  random_state=self.seed)
        return x_train, y_train

    def load_test(self, test_size=0.2):
        x, y, self.id = load_gan_dataset(self.name, self.seed)
        self.shape = x.shape
        y = np.asarray(y != 'nor', np.int32)
        _, x_test, _, y_test = train_test_split(x,
                                                y,
                                                test_size=test_size,
                                                random_state=self.seed)
        return x_test, y_test


class MATReader:

    names = ['musk', 'speech']

    def __init__(self, name=None, seed=42):
        if name.lower() in self.names:
            self.name = name.capitalize()
        else:
            raise ValueError("Data name not recognized.")
        self.seed = seed

    @property
    def coltypes(self):
        return range(self.shape[-1]), None

    def load_train(self, test_size=None):
        data = loadmat(DATA_DIR + self.name + '.mat')
        x = data['X']
        y = data['y'].flatten()
        self.shape = x.shape
        x_train, _, y_train, _ = train_test_split(x,
                                                  y,
                                                  test_size=test_size,
                                                  random_state=self.seed)
        return x_train, y_train

    def load_test(self, test_size=None):
        data = loadmat(DATA_DIR + self.name + '.mat')
        x = data['X']
        y = data['y'].flatten()
        _, x_test, _, y_test = train_test_split(x,
                                                y,
                                                test_size=test_size,
                                                random_state=self.seed)
        return x_test, y_test


def get(name):
    """Returns the DataReader matching the given name."""
    if not isinstance(name, str):
        # Could pass a DataReader as a name
        # only for unittesting
        return name
    if name.lower() in MNISTReader.names:
        return MNISTReader(name)
    elif name.lower() in KDDCupReader.names:
        return KDDCupReader()
    elif name.lower() in UCIReader.names:
        return UCIReader(name)
    elif name.lower() in MATReader.names:
        return MATReader(name)
    else:
        raise ValueError("Dataset name not recognized.")
