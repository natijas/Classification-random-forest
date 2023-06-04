from matplotlib import pyplot as plt
from sklearn.datasets import load_wine
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import StratifiedKFold
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.preprocessing import LabelEncoder
import seaborn as sns

from algorithms.ID3 import ID3
from algorithms.C45 import C45



URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data"
col_names = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']
df = pd.read_csv(URL, names=col_names)


def load_preprocessed_data():
    '''
    Loads data and splits them to X and Y
    '''
    modified_data = df.drop(columns=['class'], axis=1)
    target = df['class']

    return StandardScaler().fit_transform(modified_data.values), target.values


def main():
    global df
    cat2val = {
        'buying': ['low', 'med', 'high', 'vhigh'],
        'maint': ['low', 'med', 'high', 'vhigh'],
        'doors': ['2', '3', '4', '5more'],
        'persons': ['2', '4', 'more'],
        'lug_boot': ['small', 'med', 'big'],
        'safety': ['low', 'med', 'high'],
        'class': ['unacc', 'acc', 'good', 'vgood'],
    }

    # for column in df.columns:
    #     df[column] = [cat2val[column].index(cat) for cat in df[column]]
    #
    # X, Y = load_preprocessed_data()
    # print(Y)
    print(len(df))

    df = df.sample(500)
    X, Y = df[df.columns[:-1]], df['class']

    classifiers = [
        ('ID3', ID3(max_depth=5, random_state=42)),
        ('C45', C45(max_depth=5, discrete_features=X.columns, validation_ratio=0.2, random_state=42)),
    ]

    accuracies = []

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

    for name, clf in classifiers:
        accuracies.append([])
        for train_idx, val_idx in skf.split(X, Y):
            clf.fit(X.iloc[train_idx], Y.iloc[train_idx])
            Y_pred = clf.predict(X.iloc[val_idx])
            accuracy = accuracy_score(Y.iloc[val_idx], Y_pred)
            accuracies[-1].append(accuracy)
        print(f"{name} Accuracy: {np.mean(accuracies[-1]):.4f} +/- {np.std(accuracies[-1]):.4f}")

    names, acc_values = zip(*classifiers)
    plt.figure(figsize=(15, 5))
    sns.barplot(data=pd.DataFrame([(name, acc) for name, accs in zip(names, accuracies) for acc in accs]), x=0, y=1)
    plt.ylabel('Accuracy')
    plt.title('Comparison of Classifier Accuracies on the Gender Classification Dataset')


if __name__ == '__main__':
    main()


