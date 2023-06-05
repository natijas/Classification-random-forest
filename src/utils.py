import pandas as pd


def load_gender_dataset():
    """
    return a pair (
        data dataframe with required 'target' column,
        a mapping for discrete columns to value order,
    )
    """
    df = pd.read_csv('gender_classification_v7.csv')
    df = df.rename(columns={'gender': 'target'})

    return df, {
        'long_hair': [0, 1],
        'nose_wide': [0, 1],
        'nose_long': [0, 1],
        'lips_thin': [0, 1],
        'distance_nose_to_lip_long': [0, 1],
    }


def load_car_dataset():
    """
    return a pair (
        data dataframe with required 'target' column,
        a mapping for discrete columns to value order,
    )
    """

    URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data"
    col_names = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'target']
    df = pd.read_csv(URL, names=col_names)

    return df, {
        'buying': ['low', 'med', 'high', 'vhigh'],
        'maint': ['low', 'med', 'high', 'vhigh'],
        'doors': ['2', '3', '4', '5more'],
        'persons': ['2', '4', 'more'],
        'lug_boot': ['small', 'med', 'big'],
        'safety': ['low', 'med', 'high'],
        # 'target': ['unacc', 'acc', 'good', 'vgood'],
    }
