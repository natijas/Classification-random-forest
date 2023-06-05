from typing import Union, Type, List, Dict

import numpy as np

from algorithms.C45 import C45
from algorithms.ID3 import ID3


class RandomForest:
    def __init__(self, n_estimators: int, max_features: Union[int, str], max_depth: int,
                 TreeType: Type[Union[C45, ID3]], tree_params: Dict[str, Union[int, List[str]]]):
        self._n_estimators = n_estimators
        self._max_features = max_features
        self._max_depth = max_depth
        self._trees = [TreeType(**tree_params) for _ in range(n_estimators)]
        self._sample_indices = []

    def _generate_bootstrap_samples(self, X: np.ndarray, Y: np.ndarray) -> tuple:
        if len(X) != len(Y):
            raise ValueError("Inputs X and y must have the same length")

        bootstrap_indices = np.random.choice(range(X.shape[0]), size=X.shape[0], replace=True)
        oob_indices = [i for i in range(len(X)) if i not in bootstrap_indices]

        X_bootstrap, Y_bootstrap = X[bootstrap_indices], Y[bootstrap_indices]
        X_oob, Y_oob = X[oob_indices], Y[oob_indices]
        self._sample_indices.append(bootstrap_indices)

        return X_bootstrap, Y_bootstrap, X_oob, Y_oob

    def fit(self, X: np.ndarray, Y: np.ndarray):
        for tree in self._trees:
            X_bootstrap, Y_bootstrap, _, _ = self._generate_bootstrap_samples(X, Y)
            tree.fit(X_bootstrap, Y_bootstrap)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if len(X.shape) != 2:
            raise ValueError("Input X must be a 2D array")

        if not self._trees:
            raise ValueError("No trees have been created")

        predictions = []
        for sample in X:
            ensemble_preds = [tree.predict(np.array([sample])) for tree in self._trees]
            final_pred = max(ensemble_preds, key=ensemble_preds.count)
            predictions.append(final_pred)
        return np.array(predictions)

    def _oob_score(self, X: np.ndarray, Y: np.ndarray) -> float:
        if len(X) != len(Y):
            raise ValueError("Inputs X and Y must have the same length")

        if not self._trees:
            raise ValueError("No trees have been created")

        oob_scores = []

        for idx, tree in enumerate(self._trees):
            _, _, X_oob, Y_oob = self._generate_bootstrap_samples(X, Y)
            Y_pred = tree.predict(X_oob)
            correctly_classified = np.sum(Y_oob == Y_pred)
            total_samples = len(Y_oob)
            oob_scores.append(correctly_classified / total_samples)

        return np.mean(oob_scores)
