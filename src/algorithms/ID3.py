from collections import Counter
from multiprocessing import Pool
from typing import Dict, Union, Any, Iterable, List, Optional

import numpy as np
import pandas as pd

class Node:
    """
    Node class that represents a node in a tree structure.
    """
    def __init__(self, feature, children):
        """
        Node constructor.

        :param feature: str
            The feature that this node represents.
        :param children: Dict[str, Union[Node, Leaf]]
            The children of this node, represented as a dictionary mapping
            from string to either Node or Leaf instances.
        """
        self.feature: str = feature
        self.children: Dict[str, Union[Node, Leaf]] = children

    def __repr__(self):
        """
        String representation for the Node object.

        :return: str
            String representation of Node object.
        """
        return f'Node({repr(self.feature)}, {repr(self.children)})'


class Leaf:
    """
    Leaf class that represents a leaf in a tree structure.
    """
    def __init__(self, value):
        """
        Leaf constructor.

        :param value: Any
            The value that this leaf holds.
        """
        self.value = value

    def __repr__(self):
        """
        String representation for the Leaf object.

        :return: str
            String representation of Leaf object.
        """
        return f'Leaf({repr(self.value)})'


class ID3:
    """
    ID3 class implements the ID3 (Iterative Dichotomiser 3) algorithm
    for creating a decision tree from a dataset.
    """
    def __init__(self, max_depth, features_to_use: Optional[List[str]] = None, threads=None):
        """
        Constructor for ID3 class.

        :param max_depth: int
            Maximum depth allowed for the decision tree.
        :param features_to_use: Optional[List[str]]
            List of feature names to consider while building the tree.
            If None, all features will be considered.
        :param threads: Optional[int]
            Number of threads to use for parallel processing during prediction.
            If None, multithreading will not be used.
        """
        self._max_depth = max_depth
        self._threads = threads
        self._root = None
        self._most_frequent_class = None
        self._features_to_use = features_to_use

    def __repr__(self):
        """
        String representation for the ID3 object.

        :return: str
            String representation of ID3 object.
        """
        return f'ID3(root={self._root})'

    def _information_gain(self, X: pd.DataFrame, Y: pd.Series, x: str) -> float:
        """
        Calculate information gain for a column `x` in the dataset.

        :param X: pd.DataFrame
            The feature vectors of the dataset.
        :param Y: pd.Series
            The target values of the dataset.
        :param x: str
            The feature column to calculate the information gain for.
        :return: float
            The information gain.
        """
        entropy = lambda Y: -sum(
            [counts / len(Y) * np.log(counts / len(Y)) for counts in np.unique(Y, return_counts=True)[1]])
        divided_entropy = sum([(X[x] == j).sum() / len(X) * entropy(Y[X[x] == j]) for j in X[x].unique()])
        information_gain = entropy(Y) - divided_entropy
        return information_gain

    def _max_information_gain(self, X: pd.DataFrame, Y: pd.Series) -> str:
        """
        Calculate the feature with maximum information gain.

        :param X: pd.DataFrame
            The feature vectors of the dataset.
        :param Y: pd.Series
            The target values of the dataset.
        :return: str
            The name of the feature with the highest information gain.
        """
        features_entropy = [self._information_gain(X, Y, col) for col in X.columns]
        return X.columns[np.argmax(features_entropy)]

    def _fit_algorithm(self, X: pd.DataFrame, Y: pd.Series, depth: int) -> Union[Node, Leaf]:
        """
        The main function for the ID3 algorithm. Recursively creates nodes or leaves based on the conditions.

        :param X: pd.DataFrame
            The feature vectors of the dataset.
        :param Y: pd.Series
            The target values of the dataset.
        :param depth: int
            Current depth of the tree.
        :return: Union[Node, Leaf]
            Returns a Node or a Leaf depending on the conditions.
        """
        if len(X) == 0:
            return Leaf(self._most_frequent_class)
        if depth == self._max_depth or Y.nunique() == 1 or len(X.columns) == 0:
            return Leaf(Counter(Y).most_common(1)[0][0])

        best_column = self._max_information_gain(X, Y)
        children = {}
        for value in X[best_column].unique():
            mask = X[best_column] == value
            children[value] = self._fit_algorithm(X[mask].drop(columns=best_column), Y[mask], depth + 1)
        return Node(best_column, children)

    def fit(self, X: pd.DataFrame, Y: pd.Series) -> None:
        """
        Fit function that calculates a root node and the most frequent class.

        :param X: pd.DataFrame
            The feature vectors of the dataset.
        :param Y: pd.Series
            The target values of the dataset.
        """
        self._most_frequent_class = Counter(Y).most_common(1)[0][0]
        if self._features_to_use is not None:
            X = X[sorted(set(self._features_to_use).intersection(set(X.columns)))]
        self._root = self._fit_algorithm(X, Y, 0)

    def _predict_single(self, sample: Dict[str, Any]) -> str:
        """
        Predict a label for a single sample.

        :param sample: Dict[str, Any]
            A single sample in the form of a dictionary where keys are feature names and values are feature values.
        :return: str
            Predicted class label for the input sample.
        """
        current_node: Union[Node, Leaf] = self._root
        while not isinstance(current_node, Leaf):
            value = sample[current_node.feature]
            current_node = current_node.children.get(value, None)
            if current_node is None:
                return self._most_frequent_class

        return current_node.value

    def predict(self, X: pd.DataFrame) -> Iterable[Any]:
        """
        Return predicted class labels for the input dataset.

        :param X: pd.DataFrame
            The feature vectors of the dataset.
        :return: Iterable[Any]
            Predicted class labels for all instances in the input dataset.
        """
        if self._features_to_use is not None:
            X = X[sorted(set(self._features_to_use).intersection(set(X.columns)))]
        if self._threads is not None:
            with Pool(self._threads) as pool:
                return pool.map(self._predict_single, [row for _, row in X.iterrows()])
        return list(map(self._predict_single, (row for _, row in X.iterrows())))
