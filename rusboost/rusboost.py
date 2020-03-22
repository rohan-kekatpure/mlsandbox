import numpy as np
from collections import Counter
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as pl


def trace():
    import ipdb
    ipdb.set_trace()


class RusBoost(AdaBoostClassifier):
    """
    Random understampling boosting classifier for binary classification
    """
    @staticmethod
    def _get_most_common(a):
        return Counter(a).most_common(1)[0]

    @staticmethod
    def _undersample(X, y, majority_label, minority_count):
        all_idx = np.arange(X.shape[0], dtype=np.int)
        midx = y == majority_label
        majority_idx = all_idx[midx]
        minority_idx = all_idx[~midx]
        sample_idx = np.random.permutation(majority_idx)[:minority_count]
        sample_idx = np.append(sample_idx, minority_idx)
        return sample_idx

    def _boost(self, iboost, X, y, sample_weight, random_state):
        majority_label, maj_count = self._get_most_common(y)
        minority_count = X.shape[0] - maj_count
        idx = self._undersample(X, y, majority_label, minority_count)
        X_sampled = X[idx]
        y_sampled = y[idx]
        sample_weights_sampled = sample_weight[idx]
        boost_args = (iboost, X_sampled, y_sampled, sample_weights_sampled, random_state)
        new_weights_sampled, estimator_weight, estimator_error = super(RusBoost, self)._boost(*boost_args)

        # Modify only the weights of sampled examples
        sample_weight[idx] = new_weights_sampled

        return sample_weight, estimator_weight, estimator_error