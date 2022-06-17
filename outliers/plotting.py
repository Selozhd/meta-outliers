import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import patches
from sklearn.metrics import RocCurveDisplay

from order_statistics import plot_utils
from constants import PLOTTING_STYLE


def _save_or_show(filepath):
    if filepath is not None:
        plt.savefig(filepath)
        plt.clf()
    else:
        plt.show()


@plot_utils.plotting_style(PLOTTING_STYLE)
def plot_roc_curve(fpr, tpr, roc_auc, save=None, **kwargs):
    name = kwargs.get("name")
    f, ax = plt.subplots(1)
    viz = RocCurveDisplay(fpr=fpr,
                          tpr=tpr,
                          roc_auc=roc_auc,
                          estimator_name=name,
                          pos_label=None)
    viz.plot(ax=ax, name=name)
    plt.legend(loc='lower right')
    ax.plot([0, 1], [0, 1], 'r--')
    plt.title('Receiver Operating Characteristic')
    _save_or_show(save)


@plot_utils.plotting_style(PLOTTING_STYLE)
def plot_anomalies(df, predictions, save=None, **kwargs):
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)
    predictions = np.asarray(predictions)
    df_subset = df[(predictions == 1)]
    fig, ax = plt.subplots()
    df.plot(legend=False, ax=ax, color="deepskyblue")
    df_subset.plot(legend=False, ax=ax, color="#E3242B")
    labels = ['Normal', 'Anomalous']
    plt.legend(labels, ncol=1)
    plt.xlabel('i-th order statistics of the sample')
    plt.ylabel('Value of the statistic')
    plt.title(f"Outlier scores across sample")
    if kwargs:
        empty_patch = patches.Patch(color='none')
        handles, _ = plt.gca().get_legend_handles_labels()
        handles.append(empty_patch)
        handles.append(empty_patch)
        labels.append(f'Threshold value is: {kwargs["threshold"]}')
        labels.append(f'Cut-off point: {kwargs["cut_off"]}')
        plt.legend(handles, labels)
    _save_or_show(save)


@plot_utils.plotting_style(PLOTTING_STYLE)
def plot_metric(history, metric="loss", save=None):
    """Plots a training metric of a model history."""
    plt.plot(history.history[metric])
    plt.plot(history.history[f'val_{metric}'])
    plt.title(f'model {metric}')
    plt.ylabel(f'{metric}')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    _save_or_show(save)


@plot_utils.plotting_style(PLOTTING_STYLE)
def plot_train_metric(history, metric="loss", save=None):
    """Plots a training metric of a model history."""
    plt.plot(history.history[metric])
    plt.title(f'model {metric}')
    plt.ylabel(f'{metric}')
    plt.xlabel('epoch')
    _save_or_show(save)


def plot_gan_history(train_history, name, save=None):
    dy = train_history['discriminator_loss']
    gy = train_history['generator_loss']
    #aucy = train_history['auc']
    x = np.linspace(1, len(dy), len(dy))
    fig, ax = plt.subplots()
    ax.plot(x, dy, color='green')
    ax.plot(x, gy, color='red')
    #ax.plot(x, aucy, color='yellow', linewidth = '3')
    _save_or_show(save)


def plot_multigan_history(train_history, name, save=None):
    dy = train_history['discriminator_loss']
    gy = train_history['generator_loss']
    #auc_y = train_history['auc']
    for i in range(k):
        names['gy_' + str(i)] = train_history['sub_generator{}_loss'.format(i)]
    x = np.linspace(1, len(dy), len(dy))
    fig, ax = plt.subplots()
    ax.plot(x, dy, color='blue')
    ax.plot(x, gy, color='red')
    #ax.plot(x, auc_y, color='yellow', linewidth = '3')
    for i in range(k):
        ax.plot(x, names['gy_' + str(i)], color='green', linewidth='0.5')
    _save_or_show(save)


def plot_error_histogram(errors, n_bins, save=None):
    n, bins, patches = plt.hist(np.asarray(errors),
                                n_bins,
                                density=True,
                                facecolor='g',
                                alpha=0.75)
    plt.xlabel('Error Values')
    plt.ylabel('Count')
    plt.title('MSE Plot')
    plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
    #plt.xlim(40, 160)
    #plt.ylim(0, 0.03)
    plt.grid(True)
    _save_or_show(save)


def plot_tail_density(sigma):
    t = np.linspace(0, 10, 100)
    pdf = t / sigma * np.exp(-t / sigma)
    cdf = 1 - np.exp(-t / sigma)
    # P(X > t) < 2 exp(-t/K1)
    # pdf = d/dt 1 - 2 exp(-t/K1)
    # = 2t/K1 exp(-t/K1)
    plt.axvline(color="grey")
    #plt.axline((0, 0.5), slope=0.25, color="black", linestyle=(0, (5, 5)))
    plt.plot(t,
             pdf,
             linewidth=2,
             label=r"$pdf(x;\sigma) = \frac{x}{\sigma} e^{-t/\sigma} $")
    plt.plot(t, cdf, linewidth=2, label=r"$cdf(x;\sigma) = 1 - e^{-t/\sigma} $")
    plt.xlim(-10, 10)
    plt.xlabel("t")
    plt.legend(fontsize=14)
    plt.show()