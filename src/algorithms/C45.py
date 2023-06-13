from collections import Counter, defaultdict
from typing import Dict, Union, Tuple, Optional

import numpy as np
import pandas as pd
import scipy.stats
from sklearn.model_selection import train_test_split


class CategoricalNode:
    """
    Class to represent a node in a tree structure that splits on categorical features.
    """

    def __init__(self, feature, children):
        """
        CategoricalNode constructor.

        :param feature: str
            The categorical feature that this node represents.
        :param children: Dict[str, Union[CategoricalNode, ThresholdNode, Leaf]]
            The children of this node, represented as a dictionary mapping
            from string to either CategoricalNode, ThresholdNode or Leaf instances.
        """
        self.feature = feature
        self.children: Dict[str, Union[CategoricalNode, ThresholdNode, Leaf]] = children

    def __repr__(self):
        """
        String representation for the CategoricalNode object, used for pretty printing the tree structure.

        :param level: int, optional
            The current level in the tree, used for indentation.
        :return: str
            String representation of CategoricalNode object.
        """
        return f'Node({repr(self.feature)}, {repr(self.children)})'


class ThresholdNode:
    """
    Class to represent a node in a tree structure that splits on numerical features based on a threshold.
    """

    def __init__(self, feature, threshold, left_child, right_child):
        """
        ThresholdNode constructor.

        :param feature: str
            The numerical feature that this node represents.
        :param threshold: float
            The threshold value used for splitting the data.
        :param left_child: Union[CategoricalNode, ThresholdNode, Leaf]
            The left child node of this node, created when the feature value is less than or equal to the threshold.
        :param right_child: Union[CategoricalNode, ThresholdNode, Leaf]
            The right child node of this node, created when the feature value is greater than the threshold.
        """
        self.feature = feature
        self.threshold = threshold
        self.left_child: Union[CategoricalNode, ThresholdNode, Leaf] = left_child
        self.right_child: Union[CategoricalNode, ThresholdNode, Leaf] = right_child

    def __repr__(self):
        """
        String representation for the ThresholdNode object, used for pretty printing the tree structure.

        :param level: int, optional
            The current level in the tree, used for indentation.
        :return: str
            String representation of ThresholdNode object.
        """
        return f'Node({repr(self.feature)}, {repr(self.threshold)}, {repr(self.left_child)}, {repr(self.right_child)})'


class Leaf:
    """
    Leaf class that represents a leaf in a tree structure.
    """

    def __init__(self, label):
        """
        Leaf constructor.

        :param label: Any
            The class label that this leaf holds.
        """
        self.label = label

    def __repr__(self):
        """
        String representation for the Leaf object, used for pretty printing the tree structure.

        :param level: int, optional
            The current level in the tree, used for indentation.
        :return: str
            String representation of Leaf object.
        """
        return f'Leaf({repr(self.label)})'


class C45:
    """
    Class implementing the C4.5 decision tree algorithm.
    """

    def __init__(self, max_depth, discrete_features, validation_ratio=0.2, random_seed=None,
                 criterion: str = 'gain_ratio'):
        """
        Constructor for C4.5 decision tree algorithm.

        :param max_depth: int
            Maximum depth of the tree.
        :param discrete_features: list[str]
            List of feature names that are considered discrete.
        :param validation_ratio: float, optional
            Ratio of the data used for validation during training (default is 0.2).
        :param random_seed: int, optional
            Seed for the random number generator.
        """
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
        """
        String representation of the C45 object.

        :return: str
            A string representation of the C45 object.
        """
        return f'C4.5(root={self._root})'

    def _is_continuous(self, feature):
        """
        Checks if a feature is continuous.

        :param feature: str
            Name of the feature to be checked.
        :return: bool
            Returns True if feature is continuous, otherwise False.
        """
        return feature not in self._discrete_features

    def _possible_splits(self, X: pd.DataFrame, feature):
        """
        Determines possible splits for a feature.

        :param X: pd.DataFrame
            The DataFrame of features.
        :param feature: str
            Name of the feature for which to find possible splits.
        :return: list
            List of possible splits.
        """
        if self._is_continuous(feature):
            sorted_values = sorted(X[feature].unique())
            midpoints = [(sorted_values[i] + sorted_values[i + 1]) / 2 for i in range(len(sorted_values) - 1)]
            return midpoints
        else:
            return X[feature].unique()

    def _information_gain(self, X: pd.DataFrame, Y: pd.Series, x: str) -> float:
        """
        Calculates information gain for a given feature.

        :param X: pd.DataFrame
            The DataFrame of features.
        :param Y: pd.Series
            The Series of labels.
        :param x: str
            Name of the feature for which to calculate information gain.
        :return: float
            Information gain of the feature.
        """
        entropy = lambda Y: -sum(
            [counts / len(Y) * np.log2(counts / len(Y)) for counts in np.unique(Y, return_counts=True)[1]])
        divided_entropy = sum([(X[x] == j).sum() / len(X) * entropy(Y[X[x] == j]) for j in X[x].unique()])
        information_gain = entropy(Y) - divided_entropy
        return information_gain

    def _split_information(self, X: pd.DataFrame, x: str) -> float:
        """
        Calculates the metric Split Information for a single attribute `x`.

        :param X: pd.DataFrame
            The DataFrame of features.
        :param x: str
            Name of the feature for which to calculate split information.
        :return: float
            Split Information of the feature.
        """
        split_info = -sum(
            [(X[x] == j).sum() / len(X) * np.log2((X[x] == j).sum() / len(X)) for j in X[x].unique()])
        return split_info

    def _gain_ratio(self, X: pd.DataFrame, Y: pd.Series, x: str) -> float:
        """
        Calculates the metric Information Gain Ratio for a single attribute `x`.

        :param X: The dataframe containing the feature set.
        :param Y: The target values.
        :param use_costs: If True, adjusts the Gain Ratios by dividing them by the corresponding attribute costs.
        :return: A tuple containing the name of the attribute with the maximum gain ratio and the value of the best
                 threshold if the attribute is continuous.
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
        """
        Calculates the metric Information Gain for a single attribute `x`.

        :param X: The dataframe containing the feature set.
        :param Y: The target values.
        :param x: The attribute for which to calculate the gain ratio.
        :return: The calculated gain ratio for the attribute `x`.
        """
        entropy = lambda Y: -sum(
            [counts / len(Y) * np.log(counts / len(Y)) for counts in np.unique(Y, return_counts=True)[1]])
        divided_entropy = sum([(X[x] == j).sum() / len(X) * entropy(Y[X[x] == j]) for j in X[x].unique()])
        information_gain = entropy(Y) - divided_entropy
        return information_gain

    def _criterion_fn(self, X: pd.DataFrame, Y: pd.Series, x: str) -> float:
        """
        Criterion to use, optionally gain ratio or information gain.
        
        :param X: The dataframe containing the feature set.
        :param Y: The target values.
        :param x: The attribute for which to calculate the criterium.
        """
        if self._criterion == 'gain_ratio':
            return self._gain_ratio(X, Y, x)
        elif self._criterion == 'inf_gain':
            return self._inf_gain(X, Y, x)
        assert 0

    def _best_split(self, X: pd.DataFrame, Y: pd.Series, use_costs=False) -> Tuple[str, Optional[float]]:
        """
        Determines the attribute with the maximum criterium in the dataset as a best split.

        :param X: The dataframe containing the feature set.
        :param Y: The target values.
        :param use_costs: If True, adjusts the Gain Ratios by dividing them by the corresponding attribute costs.
        :return: A tuple containing the name of the attribute with the maximum gain ratio and the value of the best
                 threshold if the attribute is continuous.
        """
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
        """
        Fits the decision tree to the given dataset.

        :param X: The dataframe containing the feature set.
        :param Y: The target values.
        :param depth: Current depth of the tree.
        :return: The root of the constructed tree.
        """
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
        """
        Fits the model to the given dataset and performs validation if a validation ratio is set.

        :param X: The dataframe containing the feature set.
        :param Y: The target values.
        :return: None
        """
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
        Evaluates the error of the predictions.

        :param Y_true: The true target values.
        :param Y_pred: The predicted target values.
        :return: The calculated mean error.
        """
        if len(Y_true) != len(Y_pred):
            raise ValueError("Both Series should have the same length")

        return (Y_true != Y_pred).mean()

    def _prune(self, node: Union[CategoricalNode, ThresholdNode, Leaf], X_val: pd.DataFrame, Y_val: pd.Series) -> None:
        """
        Prunes the tree by replacing a node with a leaf node if it does not increase the prediction error.

        :param node: The current node to prune.
        :param X_val: The dataframe containing the validation feature set.
        :param Y_val: The validation target values.
        :return: None
        """

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
        Uses the given node to predict the outputs for the given feature set.

        :param node: The node to use for prediction.
        :param X: The dataframe containing the feature set to predict.
        :return: The predicted target values.
        """
        if isinstance(node, Leaf):
            return np.array([node.label] * len(X))
        elif isinstance(node, CategoricalNode):
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
        Predicts the target values for the given feature set using the fitted model.
        
        :param X: The dataframe containing the feature set to predict.
        :return: The predicted target values.
        """
        if self._root is None:
            raise ValueError("The tree has not been fitted yet")

        return self._predict_node(self._root, X)
