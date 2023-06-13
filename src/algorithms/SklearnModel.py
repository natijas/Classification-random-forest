import functools
from collections import Counter
from typing import Iterable, Any, Dict, List

import numpy as np
import pandas as pd


class SklearnModel:
    """
    A wrapper class for sklearn models that handles the conversion of categorical features to numeric values as
    required by sklearn. It also handles unknown feature values during prediction.
    """

    def __init__(self, model_class, discrete_feature_order: Dict[str, List[Any]], **kwargs):
        """
        Initializes the SklearnModel class.

        :param model_class: The sklearn model class to be used.
        :param discrete_feature_order: A dictionary mapping column names to their possible discrete values.
        :param kwargs: Additional keyword arguments for the sklearn model.
        """
        self.model = model_class(**kwargs)
        self.name2index = {}  # {column_name -> {str_value -> index}}
        self.name2index_y = {}  # {str_value -> index}
        self.index2name_y = {}  # {index -> str_value}
        self._most_frequent_class = None
        self.discrete_feature_order = discrete_feature_order

    def _sort_key(self, val, column):
        """
        Returns a key for sorting column values. This is used to assign numerical values to columns as required
        by sklearn, because sklearn decision tree uses comparisons in nodes, therefore the order is important.

        :param val: The value to be sorted.
        :param column: The column in which the value belongs.
        :return: The key for sorting.
        """

        if column not in self.discrete_feature_order:
            return val
        return self.discrete_feature_order[column].index(val)

    def fit(self, X: pd.DataFrame, Y: pd.Series):
        """
        Fits the sklearn model to the given dataset.

        :param X: The dataframe containing the feature set.
        :param Y: The target values.
        :return: None
        """
        self._most_frequent_class = Counter(Y).most_common(1)[0][0]

        # find a mapping between string feature value and numeric values (as required by sklearn)
        for column in X.columns:
            self.name2index[column] = {}
            for index, name in enumerate(
                    sorted(X[column].unique(), key=functools.partial(self._sort_key, column=column))):
                self.name2index[column][name] = index
        for index, name in enumerate(sorted(Y.unique(), key=functools.partial(self._sort_key, column='target'))):
            self.name2index_y[name] = index
            self.index2name_y[index] = name

        # substitute values using a found mapping
        X = X.copy()
        for column in X.columns:
            X[column] = X[column].apply(lambda value: self.name2index[column][value])
        Y = Y.apply(lambda value: self.name2index_y[value])

        self.model.fit(X, Y)

    def predict(self, X: pd.DataFrame) -> Iterable[Any]:
        """
        Predicts the target values for the given feature set using the fitted model.
        If any unknown feature value is encountered during prediction, it is assigned the most common class.

        :param X: The dataframe containing the feature set to predict.
        :return: The predicted target values.
        """
        # substitute values using a mapping
        X = X.copy()
        for column in X.columns:
            X[column] = X[column].apply(lambda value: self.name2index[column].get(value))

        # find samples that contains unknown feature value during fitting and assigned them the most common class
        mask = X.isna().apply(lambda row: row.any(), axis=1).values

        res = np.array([self._most_frequent_class] * len(X), dtype=object)
        if not mask.all():
            res[~mask] = [self.index2name_y[y] for y in self.model.predict(X[~mask])]
        return res
