# ppyPatternRecognition
**[Should read this in github]**

This is a library about AI model. All the model ideas are come from inclass lecture.

You can see the class github [here](https://github.com/ekapolc/Pattern_2024) which is created by Aj. Ekapol.

## Table of content
- [ppyPatternRecognition](#ppypatternrecognition)
  - [Table of content](#table-of-content)
  - [Provided model](#provided-model)
  - [How to install](#how-to-install)
  - [K-means](#k-means)
  - [Linear regression](#linear-regression)
  - [Logistic regression](#logistic-regression)
  - [Simple Naïve Bayes](#simple-naïve-bayes)

## Provided model
- [ppyPatternRecognition](#ppypatternrecognition)
  - [Table of content](#table-of-content)
  - [Provided model](#provided-model)
  - [How to install](#how-to-install)
  - [K-means](#k-means)
  - [Linear regression](#linear-regression)
  - [Logistic regression](#logistic-regression)
  - [Simple Naïve Bayes](#simple-naïve-bayes)

## How to install
1. Install python
1. Install library using `pip`
    ```bash
    pip install -U ppyPatternRecognition
    ```

## K-means
code [here](https://github.com/jirayuwat12/ppyPatternRecognition/tree/main/ppyPatternRecognition/clustering/kmeans.py)

example
```python
from ppyPatternRecognition import Kmeans

df = pd.read_csv(...)

k_means = Kmeans()

# fit the model
labeled_df = k_means.fit(df, k=3)

# print the label
print(labeled_df['label'])

# get the last centroid
print(k_means.last_centroid)
```
- `fit` method will return the dataframe with label column
- `last_centroid` is the last centroid of the model after fitting


## Linear regression
code [here](https://github.com/jirayuwat12/ppyPatternRecognition/tree/main/ppyPatternRecognition/regression/linear_regression.py)

example
```python
from ppyPatternRecognition import LinearRegression

df = pd.read_csv(...)
X_train, y_train = ...

linear_regression = LinearRegression()

# fit the model
linear_regression.fit(X_train, y_train, epochs=1000, lr=0.01)

# predict
y_pred = linear_regression.predict(X_test)
```
- `fit` method will return the model itself
- `predict` method will return the predicted value
  

## Logistic regression
code [here](https://github.com/jirayuwat12/ppyPatternRecognition/tree/main/ppyPatternRecognition/regression/logistic_regression.py)

example
```python
from ppyPatternRecognition import LogisticRegression

df = pd.read_csv(...)
X_train, y_train = ...

logistic_regression = LogisticRegression()

# fit the model
logistic_regression.fit(X_train, y_train, epochs=1000, lr=0.01)

# predict
y_pred = logistic_regression.predict(X_test)
```
- `fit` method will return the model itself
- `predict` method will return the predicted value

## Simple Naïve Bayes
code [here](https://github.com/jirayuwat12/ppyPatternRecognition/tree/main/ppyPatternRecognition/clustering/simple_naive_bayes.py)

example
```python
from ppyPatternRecognition import SimpleBayesClassifier

df = pd.read_csv(...)
X_train, y_train = ...

simple_naive_bayes = SimpleBayesClassifier()

# fit the model
simple_naive_bayes.fit(X_train, y_train)

# predict
y_pred = simple_naive_bayes.predict(X_test)
```
- `fit` method will return the model itself
- `fit_gaussian` method will return the model itself
- `predict` method will return the predicted value
- `predict_gaussian` method will return the predicted value
