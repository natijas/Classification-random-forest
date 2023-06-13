# Modified Random Forest Classification Algorithm

## Project Description

The project aims to implement a modified algorithm for generating a random forest classification. In this case, the algorithm uses the elements of the training set on which the model makes larger errors more frequently. The project is implemented in the Python programming language.

## Used Libraries

The project utilizes the following libraries:

- pandas: a library for easy data manipulation and analysis, providing necessary data structures.
- numpy: a library for fast matrix and vector operations.
- random: a library containing various pseudo-random number generators for different distributions.
- sklearn: a machine learning software library for Python.
- multiprocessing: a module for speeding up the process by utilizing multiprocessing. The project uses the Pool module from the multiprocessing package, and all processes are run in parallel on 24 cores.

## Project Stages

The project consists of the following stages:

1. Data Preprocessing:
   - Data cleaning, handling missing values, and converting categorical variables into numerical values.
   - Splitting the data into a training set and a validation set using cross-validation.

2. Implementation of the C4.5 Algorithm:
   - Implementation of the C4.5 algorithm, which is an improvement over the ID3 algorithm.
   - Generating decision trees that will constitute the random forest. The C4.5 algorithm takes into account the information gain and gain ratio when selecting the best attribute for tree splitting.

3. Implementation of the Modified Random Forest Algorithm:
   - Generating a specified number of trees (n_trees) using the C4.5 algorithm.
   - Updating the weights of misclassified records after each iteration. The weights of misclassified records are increased to increase the probability of their inclusion in the sample for the next tree.

4. Bootstrap Sampling:
   - For each tree in the random forest, sampling a bootstrap sample (with replacement) from the training set.
   - The sample size is equal to the original training set size, but misclassified records have higher weights to increase the probability of their inclusion.

## Experiments from Initial Documentation

The following experiments were conducted based on the initial documentation:

- Confusion Matrix Analysis: Confusion matrices were generated for each of the models to provide a detailed breakdown of their classification performance. This includes the number of true positives (TP), true negatives (TN), false positives (FP), and false negatives (FN) for each class.

- Effect of Difficult Sample Weighting on Imbalanced Data Sets: This experiment specifically analyzed the impact of the modified random forest algorithm with difficult sample weighting on imbalanced training data sets. The hypothesis is that this modified algorithm will perform better on imbalanced data sets, assuming a balanced validation/test data set.

- Hyperparameter Optimization: For each of the models, an experiment was conducted to search for optimal hyperparameters.

- ROC Curve Analysis: ROC curves (Receiver Operating Characteristic) were plotted for each of the three models (modified random forest with weight, modified random forest without weight, and Sklearn's Random Forest). This provides a visual representation of the true positive rate (sensitivity) against the false positive rate (1-specificity) for each model.
