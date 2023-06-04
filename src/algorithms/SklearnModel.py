import functools
from collections import Counter
from typing import Iterable, Any

import numpy as np
import pandas as pd


class SklearnModel:
    def __init__(self, model_class, **kwargs):
        self.model = model_class(**kwargs)
        self.name2index = {}  # {column_name -> {str_value -> index}}
        self.name2index_y = {}  # {str_value -> index}
        self.index2name_y = {}  # {index -> str_value}
        self._most_frequent_class = None

    @staticmethod
    def _sort_key(val, column):
        '''
        returns a key for sorting column values (to assign numerical values to columns as required by sklearn).
        Sklearn decision tree uses comparison is nodes therefore the order is important.
        '''
        cat2val = {
            'buying': ['low', 'med', 'high', 'vhigh'],
            'maint': ['low', 'med', 'high', 'vhigh'],
            'doors': ['2', '3', '4', '5more'],
            'persons': ['2', '4', 'more'],
            'lug_boot': ['small', 'med', 'big'],
            'safety': ['low', 'med', 'high'],
            'class': ['unacc', 'acc', 'good', 'vgood'],
        }

        return cat2val[column].index(val)

    def fit(self, X: pd.DataFrame, Y: pd.Series):
        self._most_frequent_class = Counter(Y).most_common(1)[0][0]

        # find a mapping between string feature value and numeric values (as required by sklearn)
        for column in X.columns:
            self.name2index[column] = {}
            for index, name in enumerate(
                    sorted(X[column].unique(), key=functools.partial(self._sort_key, column=column))):
                self.name2index[column][name] = index
        for index, name in enumerate(sorted(Y.unique(), key=functools.partial(self._sort_key, column='class'))):
            self.name2index_y[name] = index
            self.index2name_y[index] = name

        # substitute values using a found mapping
        X = X.copy()
        for column in X.columns:
            X[column] = X[column].apply(lambda value: self.name2index[column][value])
        Y = Y.apply(lambda value: self.name2index_y[value])

        self.model.fit(X, Y)

    def predict(self, X: pd.DataFrame) -> Iterable[Any]:
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
