from typing import Dict, Union
from collections import Counter, defaultdict

import scipy.stats

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class Node:
    def __init__(self, feature, children):
        self.feature = feature
        self.children: Dict[str, Union[Node, Leaf]] = children

    def __repr__(self, level=0):
        ret = "\t" * level + repr(self.feature) + "\n"
        for child in self.children.values():
            ret += child.__repr__(level + 1)
        return ret

    def predict(self, instance: pd.Series):
        if not isinstance(instance, pd.Series):
            raise TypeError(f'Expected type pd.Series but got {type(instance).__name__}')

        value = instance.get(self.feature)
        if value is None:
            raise ValueError(f"Feature {self.feature} not found in the instance")

        child_node = self.children.get(value)
        if child_node is None:
            raise ValueError(f"No child node found for value {value} of feature {self.feature}")

        return child_node.predict(instance)

    def count_nodes(self):
        count = 1  # self
        for child in self.children.values():
            if not isinstance(child, (Node, Leaf)):
                raise TypeError(f'Expected type Node or Leaf but got {type(child).__name__}')

            if isinstance(child, Node):
                count += child.count_nodes()
        return count

class Leaf:
    def __init__(self, label):
        self.label = label

    def __repr__(self, level=0):
        return "\t"*level + "Leaf(" + repr(self.label) + ")" + "\n"

    def predict(self, instance: pd.Series):
        return self.label

    def count_nodes(self):
        return 1



class C45:
    def __init__(self, max_depth, discrete_features, validation_ratio=0.2, random_state=None):
        self._max_depth = max_depth
        self._discrete_features = discrete_features
        self._validation_ratio = validation_ratio
        self._root = None
        self._most_frequent_class = None
        self.random_state = random_state

    def __repr__(self):
        return f'C4.5(root={self._root})'

    def _is_continuous(self, feature):
        return feature not in self._discrete_features

    def _possible_splits(self, X: pd.DataFrame, feature):
        if self._is_continuous(feature):
            sorted_values = sorted(X[feature].unique())
            midpoints = [(sorted_values[i] + sorted_values[i + 1]) / 2 for i in range(len(sorted_values) - 1)]
            return midpoints
        else:
            return X[feature].unique()

    def _information_gain(self, X: pd.DataFrame, Y: pd.Series, x: str) -> float:
        entropy = lambda Y: -sum(
            [counts / len(Y) * np.log2(counts / len(Y)) for counts in np.unique(Y, return_counts=True)[1]])
        divided_entropy = sum([(X[x] == j).sum() / len(X) * entropy(Y[X[x] == j]) for j in X[x].unique()])
        information_gain = entropy(Y) - divided_entropy
        return information_gain

    def _split_information(self, X: pd.DataFrame, x: str) -> float:
        """
        Calculates the metric Split Information for a single attribute `x`
        :param X:
        :param Y:
        :param x:
        :return:
        """
        split_info = -sum(
            [(X[x] == j).sum() / len(X) * np.log2((X[x] == j).sum() / len(X)) for j in X[x].unique()])
        return split_info

    def _gain_ratio(self, X: pd.DataFrame, Y: pd.Series, x: str) -> float:
        """
        Calculates the metric Information Gain Ratio for a single attribute `x`
        :param X:
        :param Y:
        :param x:
        :return:
        """
        # Drop missing values before calculating gain ratio
        mask = X[x].notna()
        X_non_missing = X[mask]
        Y_non_missing = Y[mask]

        information_gain = self._information_gain(X_non_missing, Y_non_missing, x)
        split_info = self._split_information(X_non_missing, x)
        if split_info == 0:
            return 0
        gain_ratio = information_gain / split_info
        return gain_ratio

    def _max_gain_ratio(self, X: pd.DataFrame, Y: pd.Series, use_costs=False) -> str:
        if len(X) != len(Y):
            raise ValueError("X and Y must have the same number of rows")

        if use_costs and not hasattr(self, '_attribute_costs'):
            raise ValueError("Attribute costs not defined")

        features_gain_ratio = []
        for col in X.columns:
            if self._is_continuous(col):
                max_gain_ratio = 0
                best_midpoint = None

                for midpoint in self._possible_splits(X, col):
                    tmp_df = X.copy()
                    tmp_df[col] = ['<=' + str(midpoint) if val <= midpoint else '>' + str(midpoint) for val in X[col]]
                    gain_ratio = self._gain_ratio(tmp_df, Y, col)

                    # adjust gain ratio based on attribute cost if necessary
                    if use_costs:
                        gain_ratio /= self._attribute_costs[col]

                    if gain_ratio > max_gain_ratio:
                        max_gain_ratio = gain_ratio
                        best_midpoint = midpoint

                if best_midpoint is None:
                    raise ValueError(f"Found no valid split for continuous features {col}")

                X[col] = ['<=' + str(best_midpoint) if val <= best_midpoint else '>' + str(best_midpoint) for val in
                          X[col]]
                features_gain_ratio.append(max_gain_ratio)
            else:
                gain_ratio = self._gain_ratio(X, Y, col)

                # adjust gain ratio based on attribute cost if necessary
                if use_costs:
                    gain_ratio /= self._attribute_costs[col]

                features_gain_ratio.append(gain_ratio)

        if not features_gain_ratio:
            raise ValueError("No valid features were found.")

        return X.columns[features_gain_ratio.index(max(features_gain_ratio))]

    def _fit_algorithm(self, X: pd.DataFrame, Y: pd.Series, depth: int) -> Union[Node, Leaf]:
        if len(X) == 0 or len(X.columns) == 0:
            return Leaf(self._most_frequent_class)
        if depth == self._max_depth or Y.nunique() == 1:
            return Leaf(Counter(Y).most_common(1)[0][0])

        best_column = self._max_gain_ratio(X, Y)
        children = {}

        for value in X[best_column].unique():
            mask = X[best_column] == value
            children[value] = self._fit_algorithm(X[mask].drop(columns=best_column), Y[mask], depth + 1)

        return Node(best_column, children)

    def fit(self, X: pd.DataFrame, Y: pd.Series) -> None:
        X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=self._validation_ratio, random_state=self.random_state)

        self._most_frequent_class = Counter(Y).most_common(1)[0][0]
        self._root = self._fit_algorithm(X_train, Y_train, 0)
        self._prune(self._root, X_val, Y_val)

    def _error(self, Y_true: pd.Series, Y_pred: pd.Series) -> float:
        """
        Evaluate the error of the predictions
        """
        if len(Y_true) != len(Y_pred):
            raise ValueError("Both Series should have the same length")

        return (Y_true != Y_pred).mean()

    def _prune(self, node: Union[Node, Leaf], X_val: pd.DataFrame, Y_val: pd.Series) -> None:

        if not isinstance(node, Node):
            raise TypeError(f'Expected type Node but got {type(node).__name__}')

        if len(X_val) == 0:
            return

        for value, child in node.children.items():
            if isinstance(child, Node):
                self._prune(child, X_val[X_val[child.feature] == value], Y_val[X_val[child.feature] == value])

        original_children = node.children

        original_error = self._error(Y_val, self._predict_node(node, X_val))

        node.children = defaultdict(lambda: Leaf(Counter(Y_val).most_common(1)[0][0]))

        pruned_error = self._error(Y_val, self._predict_node(node, X_val))

        if original_error >= pruned_error:
            # If error is not increased, prune permanently
            return

         # If error is increased, revert pruning
        node.children = original_children

    def _predict_node(self, node: Union[Node, Leaf], X: pd.DataFrame) -> np.array:
        """
        Use the given node to predict the outputs
        """
        if isinstance(node, Leaf):
            return np.array([node.label] * len(X))
        elif isinstance(node, Node):
            # If an instance has a missing value for a feature that the tree wants to split on,
            # pass the instance down all branches of the tree, and then take a vote among the leaf nodes it ends up in.
            # results = []
            # for i, row in X.iterrows():
            #     if pd.isna(row[node.feature]) or row[node.feature] not in node.children:
            #         # Missing attribute, pass instance down all branches and collect results
            #         temp_results = [self._predict_node(child, pd.DataFrame([row]))[0] for child in node.children.values()]
            #         if not temp_results:
            #             results.append(self._most_frequent_class)
            #         else:
            #             results.append(Counter(temp_results).most_common(1)[0][0])
            #     else:
            #         # No missing attribute, pass instance down appropriate branch
            #         results.append(self._predict_node(node.children[row[node.feature]], pd.DataFrame([row]))[0])
            results = np.zeros(len(X), dtype=np.array([self._most_frequent_class]).dtype)
            for value, subX in X.reset_index(drop=True).groupby(node.feature):
                if pd.isna(value) or node.children.get(value) is None:
                    # Missing attribute, pass instance down all branches and collect results
                    temp_results = [self._predict_node(child, subX) for child in node.children.values()]
                    if not temp_results:
                        results[subX.index] = [self._most_frequent_class] * len(subX)
                    else:
                        results[subX.index] = scipy.stats.mode(np.stack(temp_results), axis=0).mode[0]
                else:
                    # No missing attribute, pass instance down appropriate branch
                    results[subX.index] = self._predict_node(node.children[value], subX)
            return results
        else:
            raise TypeError(f'Expected type Node or Leaf but got {type(node).__name__}')

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Predict the outputs for the given inputs
        """
        if self._root is None:
            raise ValueError("The tree has not been fitted yet")

        # try:
        return self._predict_node(self._root, X)
        # except Exception as e:
        #     print("Error in prediction: ", e)
        #     # If an error occurs during prediction, return the most frequent class
        #     return pd.Series([self._most_frequent_class] * len(X))
