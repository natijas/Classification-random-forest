from multiprocessing import Pool
from typing import Dict, Union, Any, Iterable
from collections import Counter

import numpy as np
import pandas as pd


class Node:
    def __init__(self, feature, children):
        self.feature: str = feature
        self.children: Dict[str, Union[Node, Leaf]] = children

    def __repr__(self):
        return f'Node({repr(self.feature)}, {repr(self.children)})'


class Leaf:
    def __init__(self, value):
        self.value = value

    def __repr__(self):
        return f'Leaf({repr(self.value)})'


class ID3:
    def __init__(self, max_depth, threads=None, random_state=None):
        self._max_depth = max_depth
        self._threads = threads
        self._root = None
        self._most_frequent_class = None
        self.random_state = random_state

    def __repr__(self):
        return f'ID3(root={self._root})'

    def _information_gain(self, X: pd.DataFrame, Y: pd.Series, x: str) -> float:
        '''
        Calculates information gain for a column `x` and returns it
        '''
        entropy = lambda Y: -sum(
            [counts / len(Y) * np.log(counts / len(Y)) for counts in np.unique(Y, return_counts=True)[1]])
        divided_entropy = sum([(X[x] == j).sum() / len(X) * entropy(Y[X[x] == j]) for j in X[x].unique()])
        information_gain = entropy(Y) - divided_entropy
        return information_gain

    def _max_information_gain(self, X: pd.DataFrame, Y: pd.Series) -> str:
        '''
        Calculates for each features information gain and returns a feature column with the largest information gain
        '''
        features_entropy = [self._information_gain(X, Y, col) for col in X.columns]
        return X.columns[features_entropy.index(max(features_entropy))]

    def _fit_algorithm(self, X: pd.DataFrame, Y: pd.Series, depth: int) -> Union[Node, Leaf]:
        '''
        Main ID3 algorithm function, returns a Node or a Leaf
        '''
        if len(X) == 0 or len(X.columns) == 0:
            return Leaf(self._most_frequent_class)
        if depth == self._max_depth or Y.nunique() == 1:
            return Leaf(Counter(Y).most_common(1)[0][0])

        best_column = self._max_information_gain(X, Y)
        children = {}
        for value in X[best_column].unique():
            mask = X[best_column] == value
            children[value] = self._fit_algorithm(X[mask].drop(columns=best_column), Y[mask], depth + 1)
        return Node(best_column, children)

    def fit(self, X: pd.DataFrame, Y: pd.Series) -> None:
        '''
        fit function, that calculates a root node and most frequent class
        '''
        self._most_frequent_class = Counter(Y).most_common(1)[0][0]
        self._root = self._fit_algorithm(X, Y, 0)

    def _predict_single(self, sample: Dict[str, Any]) -> str:
        '''
        Predict a label for a single sample
        '''
        current_node: Union[Node, Leaf] = self._root
        while not isinstance(current_node, Leaf):
            value = sample[current_node.feature]
            current_node = current_node.children.get(value, None)
            if current_node is None:
                return self._most_frequent_class

        return current_node.value

    def predict(self, X: pd.DataFrame) -> Iterable[Any]:
        '''
        Returns predicted value of terminal Node on X
        '''
        if self._threads is not None:
            with Pool(self._threads) as pool:
                return pool.map(self._predict_single, [row for _, row in X.iterrows()])
        return list(map(self._predict_single, (row for _, row in X.iterrows())))
