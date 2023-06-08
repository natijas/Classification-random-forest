from typing import Union, Callable, List

import numpy as np
import pandas as pd
import scipy.stats

from algorithms.C45 import C45
from algorithms.ID3 import ID3


class RandomForest:
    def __init__(self, n_estimators: int, max_features: Union[int, str],
                 tree_constructor: Callable[[], Union[C45, ID3]], bootstrap_fraction: float,
                 sampling_temperature: float = 1, random_seed=None):
        self._n_estimators = n_estimators
        self._max_features = max_features
        self._trees = [tree_constructor() for _ in range(n_estimators)]
        self._tree_features = [[] for _ in range(n_estimators)]
        self._bootstrap_fraction = bootstrap_fraction
        self._sampling_temperature = sampling_temperature
        self._random_seed = random_seed

    def _generate_bootstrap_samples(self, X: pd.DataFrame, Y: pd.Series, probabilities: np.ndarray, rng: np.random.RandomState) -> tuple:
        if len(X) != len(Y):
            raise ValueError("Inputs X and y must have the same length")

        bootstrap_indices = rng.choice(range(X.shape[0]), size=round(X.shape[0] * self._bootstrap_fraction),
                                       replace=True, p=probabilities)
        oob_indices = [i for i in range(len(X)) if i not in bootstrap_indices]

        X_bootstrap, Y_bootstrap = X.iloc[bootstrap_indices], Y.iloc[bootstrap_indices]

        return X_bootstrap, Y_bootstrap, oob_indices

    def _get_max_features(self, n_features: int) -> int:
        if isinstance(self._max_features, str):
            if self._max_features == 'sqrt':
                return int(np.sqrt(n_features))
            elif self._max_features == 'log2':
                return int(np.log2(n_features))
            else:
                raise ValueError("max_features should be 'sqrt', 'log2', or an integer")
        elif isinstance(self._max_features, int):
            if self._max_features > n_features:
                raise ValueError("max_features should not be greater than number of features")
            return self._max_features
        else:
            raise ValueError("max_features should be 'sqrt', 'log2', or an integer")

    def _get_sample_probabilities(self, correct_predictions: List[List[bool]]) -> np.ndarray:
        if not any(map(len, correct_predictions)):
            # no oob predictions at all
            return np.ones(len(correct_predictions)) / len(correct_predictions)  # every sample with equal probability

        mean_accuracy = np.mean([np.mean(correct_list) for correct_list in correct_predictions if correct_list])
        accuracy = np.array(
            [np.mean(correct_list) if correct_list else mean_accuracy for correct_list in correct_predictions])
        error = 1 - accuracy
        sharpened_values = error ** self._sampling_temperature
        return sharpened_values / sharpened_values.sum()

    def fit(self, X: pd.DataFrame, Y: pd.Series):
        max_features = self._get_max_features(X.shape[1])

        # correct_prediction[i] returns a list of bools indicating if a prediction of a tree was correct
        # for a list of trees in which sample i was in OOB set
        correct_prediction: List[List[bool]] = [[] for _ in range(len(X))]
        
        rng = np.random.RandomState(self._random_seed)

        for i, tree in enumerate(self._trees):
            self._tree_features[i] = rng.choice(X.columns, size=max_features, replace=False)
            X_bootstrap, Y_bootstrap, oob_indices = self._generate_bootstrap_samples(
                X[self._tree_features[i]], Y, self._get_sample_probabilities(correct_prediction), rng)
            tree.fit(X_bootstrap, Y_bootstrap)

            correct = tree.predict(X.iloc[oob_indices][self._tree_features[i]]) == Y.iloc[oob_indices]
            for j, sample_correct in zip(oob_indices, correct):
                correct_prediction[j].append(sample_correct)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if len(X.shape) != 2:
            raise ValueError("Input X must be a 2D array")

        if not self._trees:
            raise ValueError("No trees have been created")

        ensemble_preds = np.stack(
            [tree.predict(X[tree_features]) for tree, tree_features in zip(self._trees, self._tree_features)])
        predictions = scipy.stats.mode(ensemble_preds, axis=0).mode[0]
        return predictions

    # def _oob_score(self, X: np.ndarray, Y: np.ndarray) -> float:
    #     if len(X) != len(Y):
    #         raise ValueError("Inputs X and Y must have the same length")
    #
    #     if not self._trees:
    #         raise ValueError("No trees have been created")
    #
    #     oob_scores = []
    #
    #     for idx, tree in enumerate(self._trees):
    #         _, _, X_oob, Y_oob = self._generate_bootstrap_samples(X, Y)
    #         Y_pred = tree.predict(X_oob)
    #         correctly_classified = np.sum(Y_oob == Y_pred)
    #         total_samples = len(Y_oob)
    #         oob_scores.append(correctly_classified / total_samples)
    #
    #     return np.mean(oob_scores)

    # @property
    # def feature_importances_(self):
    #     all_importances = [tree.feature_importances_ for tree in self._trees]
    #     avg_importances = np.mean(all_importances, axis=0)
    #     return avg_importances
