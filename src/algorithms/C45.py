from collections import Counter, defaultdict
from typing import Dict, Union, Tuple, Optional

import numpy as np
import pandas as pd
import scipy.stats
from sklearn.model_selection import train_test_split


class CategoricalNode:
    def __init__(self, feature, children):
        self.feature = feature
        self.children: Dict[str, Union[CategoricalNode, ThresholdNode, Leaf]] = children

    def __repr__(self):
        return f'Node({repr(self.feature)}, {repr(self.children)})'


class ThresholdNode:
    def __init__(self, feature, threshold, left_child, right_child):
        self.feature = feature
        self.threshold = threshold
        self.left_child: Union[CategoricalNode, ThresholdNode, Leaf] = left_child
        self.right_child: Union[CategoricalNode, ThresholdNode, Leaf] = right_child

    def __repr__(self):
        return f'Node({repr(self.feature)}, {repr(self.threshold)}, {repr(self.left_child)}, {repr(self.right_child)})'


class Leaf:
    def __init__(self, label):
        self.label = label

    def __repr__(self):
        return f'Leaf({repr(self.label)})'


class C45:
    def __init__(self, max_depth, discrete_features, validation_ratio=0.2, random_seed=None, criterion: str = 'gain_ratio'):
        self._max_depth = max_depth
        self._discrete_features = discrete_features
        self._validation_ratio = validation_ratio
        self._root = None
        self._most_frequent_class = None
        self._criterion = criterion
        if criterion not in ['gain_ratio', 'inf_gain']:
            raise ValueError("unsupported criterion")
        self.random_seed = random_seed

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
    
    def _inf_gain(self, X: pd.DataFrame, Y: pd.Series, x: str) -> float:
        entropy = lambda Y: -sum(
            [counts / len(Y) * np.log(counts / len(Y)) for counts in np.unique(Y, return_counts=True)[1]])
        divided_entropy = sum([(X[x] == j).sum() / len(X) * entropy(Y[X[x] == j]) for j in X[x].unique()])
        information_gain = entropy(Y) - divided_entropy
        return information_gain
    
    def _criterion_fn(self, X: pd.DataFrame, Y: pd.Series, x: str) -> float:
        if self._criterion == 'gain_ratio':
            return self._gain_ratio(X, Y, x)
        elif self._criterion == 'inf_gain':
            return self._inf_gain(X, Y, x)
        assert 0
            

    def _best_split(self, X: pd.DataFrame, Y: pd.Series, use_costs=False) -> Tuple[str, Optional[float]]:
        if len(X) != len(Y):
            raise ValueError("X and Y must have the same number of rows")

        if use_costs and not hasattr(self, '_attribute_costs'):
            raise ValueError("Attribute costs not defined")

        split_results = []
        for col in X.columns:
            if self._is_continuous(col):
                max_value = 0
                best_midpoint = None

                for midpoint in self._possible_splits(X, col):
                    tmp_df = X.copy()
                    tmp_df[col] = ['<=' + str(midpoint) if val <= midpoint else '>' + str(midpoint) for val in X[col]]
                    value = self._criterion_fn(tmp_df, Y, col)

                    # adjust gain ratio based on attribute cost if necessary
                    if use_costs:
                        value /= self._attribute_costs[col]

                    if value > max_value:
                        max_value = value
                        best_midpoint = midpoint

                if best_midpoint is None:
                    # ignore feature, no viable splits
                    split_results.append((-np.inf, None))
                else:
                    split_results.append((max_value, best_midpoint))
            else:
                value = self._criterion_fn(X, Y, col)

                # adjust gain ratio based on attribute cost if necessary
                if use_costs:
                    value /= self._attribute_costs[col]

                split_results.append((value, None))

        if not split_results:
            raise ValueError("No valid features were found.")

        max_i = np.argmax([value for value, _ in split_results])
        if np.isinf(-split_results[max_i][0]):
            return None, None  # no features to split on, stopping tree construction
        return X.columns[max_i], split_results[max_i][1]

    def _fit_algorithm(self, X: pd.DataFrame, Y: pd.Series, depth: int) -> Union[CategoricalNode, ThresholdNode, Leaf]:
        if len(X) == 0:
            return Leaf(self._most_frequent_class)
        if depth == self._max_depth or Y.nunique() == 1 or len(X.columns) == 0:
            return Leaf(Counter(Y).most_common(1)[0][0])

        best_column, best_threshold = self._best_split(X, Y)
        if best_column is None:
            # no features to split on, stopping tree construction
            return Leaf(Counter(Y).most_common(1)[0][0])

        children = {}

        if self._is_continuous(best_column):
            left_mask = X[best_column] <= best_threshold
            left_child = self._fit_algorithm(X[left_mask], Y[left_mask], depth + 1)
            right_child = self._fit_algorithm(X[~left_mask], Y[~left_mask], depth + 1)
            return ThresholdNode(best_column, best_threshold, left_child, right_child)
        else:
            for value in X[best_column].unique():
                mask = X[best_column] == value
                children[value] = self._fit_algorithm(X[mask].drop(columns=best_column), Y[mask], depth + 1)
            return CategoricalNode(best_column, children)

    def fit(self, X: pd.DataFrame, Y: pd.Series) -> None:
        if self._validation_ratio == 0:
            X_train = X
            Y_train = Y
            X_val = Y_val = []
        else:
            X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=self._validation_ratio,
                                                              random_state=self.random_seed)

        self._most_frequent_class = Counter(Y).most_common(1)[0][0]
        self._root = self._fit_algorithm(X_train, Y_train, 0)
        if not isinstance(self._root, Leaf):
            self._prune(self._root, X_val, Y_val)

    def _error(self, Y_true: pd.Series, Y_pred: pd.Series) -> float:
        """
        Evaluate the error of the predictions
        """
        if len(Y_true) != len(Y_pred):
            raise ValueError("Both Series should have the same length")

        return (Y_true != Y_pred).mean()

    def _prune(self, node: Union[CategoricalNode, ThresholdNode, Leaf], X_val: pd.DataFrame, Y_val: pd.Series) -> None:

        if not isinstance(node, (CategoricalNode, ThresholdNode)):
            raise TypeError(f'Expected type Node but got {type(node).__name__}')

        if len(X_val) == 0:
            return

        if isinstance(node, CategoricalNode):
            for value, child in node.children.items():
                if isinstance(child, (CategoricalNode, ThresholdNode)):
                    self._prune(child, X_val[X_val[node.feature] == value], Y_val[X_val[node.feature] == value])

            original_children = node.children

            original_error = self._error(Y_val, self._predict_node(node, X_val))
            node.children = defaultdict(lambda: Leaf(Counter(Y_val).most_common(1)[0][0]))
            pruned_error = self._error(Y_val, self._predict_node(node, X_val))
            if original_error >= pruned_error:
                # If error is not increased, prune permanently
                return
            # If error is increased, revert pruning
            node.children = original_children
        else:
            if isinstance(node.left_child, (CategoricalNode, ThresholdNode)):
                self._prune(node.left_child, X_val[X_val[node.feature] <= node.threshold],
                            Y_val[X_val[node.feature] <= node.threshold])
            if isinstance(node.right_child, (CategoricalNode, ThresholdNode)):
                self._prune(node.right_child, X_val[X_val[node.feature] > node.threshold],
                            Y_val[X_val[node.feature] > node.threshold])

            original_left_child = node.left_child
            original_right_child = node.right_child

            original_error = self._error(Y_val, self._predict_node(node, X_val))
            node.left_child = node.right_child = Leaf(Counter(Y_val).most_common(1)[0][0])
            pruned_error = self._error(Y_val, self._predict_node(node, X_val))

            if original_error >= pruned_error:
                return
            node.left_child = original_left_child
            node.right_child = original_right_child

    def _predict_node(self, node: Union[CategoricalNode, ThresholdNode, Leaf], X: pd.DataFrame) -> np.array:
        """
        Use the given node to predict the outputs
        """
        if isinstance(node, Leaf):
            return np.array([node.label] * len(X))
        elif isinstance(node, CategoricalNode):
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
        elif isinstance(node, ThresholdNode):
            results = np.zeros(len(X), dtype=np.array([self._most_frequent_class]).dtype)
            left_mask = X[node.feature] <= node.threshold
            results[left_mask] = self._predict_node(node.left_child, X[left_mask])
            results[~left_mask] = self._predict_node(node.right_child, X[~left_mask])
            return results
        else:
            raise TypeError(f'Expected type Node or Leaf but got {type(node).__name__}')

    def predict(self, X: pd.DataFrame) -> np.array:
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
