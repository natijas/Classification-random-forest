import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier

from algorithms.C45 import C45
from algorithms.ID3 import ID3
from algorithms.SklearnModel import SklearnModel
from utils import load_car_dataset, load_gender_dataset


def main() -> None:
    df, discrete_columns = load_gender_dataset()

    X, Y = df.drop(columns=['target']), df['target']

    classifiers = [
        ('ID3', ID3(max_depth=5, features_to_use=list(set(discrete_columns.keys())))),
        ('C45',
         C45(max_depth=5, discrete_features=list(discrete_columns.keys()), validation_ratio=0.0, random_state=42)),
        ('DecisionTree', SklearnModel(DecisionTreeClassifier, max_depth=6, discrete_feature_order=discrete_columns)),
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
