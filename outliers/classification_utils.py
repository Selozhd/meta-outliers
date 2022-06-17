from typing import NamedTuple

import numpy as np
import sklearn.metrics as sk_metrics

from outliers.plotting import plot_roc_curve


def get_classification_report(y_true, y_scores, threshold, name=None):
    """Calculates metrics and returns a ClassificationReport."""
    y_pred = y_scores > threshold
    f1 = sk_metrics.f1_score(y_true=y_true, y_pred=y_pred)
    precision = sk_metrics.precision_score(y_true, y_pred)
    accuracy = sk_metrics.accuracy_score(y_true, y_pred)
    recall = sk_metrics.recall_score(y_true, y_pred)
    conf_matrix = sk_metrics.confusion_matrix(y_true, y_pred)
    fpr, tpr, _ = sk_metrics.roc_curve(y_true, y_scores)
    auc = sk_metrics.roc_auc_score(y_true, y_scores, average="micro")
    return ClassificationReport(f1_score=f1,
                                precision=precision,
                                accuracy=accuracy,
                                recall=recall,
                                conf_matrix=conf_matrix,
                                auc=auc,
                                roc_curve=(fpr, tpr),
                                name=name)


class ClassificationReport(NamedTuple):
    """Binary classification results."""

    f1_score: float
    precision: float
    accuracy: float
    recall: float
    conf_matrix: np.ndarray
    auc: float
    roc_curve: tuple
    name: str

    def _repr_conf_matrix(self):
        return np.str(self.conf_matrix).replace('\n', '\n\t\t\t' + 3*' ')

    def __str__(self):
        conf_matrix = self._repr_conf_matrix()
        report = [
            'Classification Report:',
            '\t F1 score: {x}'.format(x=self.f1_score),
            '\t AUC: {x}'.format(x=self.auc),
            '\t Confusion Matrix: {x}'.format(x=conf_matrix),
        ]
        return '\n'.join(report)

    __repr__ = __str__

    def plot_roc_curve(self, save=None):
        fpr, tpr = self.roc_curve
        return plot_roc_curve(fpr,
                              tpr,
                              self.auc,
                              save=save,
                              **{"name": self.name})

    def report(self):
        conf_matrix = self._repr_conf_matrix()
        report = [
            'Classification Report:',
            '\t Accuracy: {x}'.format(x=self.f1_score),
            '\t Precision: {x}'.format(x=self.f1_score),
            '\t Recall: {x}'.format(x=self.f1_score),
            '\t F1 score: {x}'.format(x=self.f1_score),
            '\t AUC: {x}'.format(x=self.auc),
            '\t Confusion Matrix: {x}'.format(x=conf_matrix),
        ]
        return '\n'.join(report)