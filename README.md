# ppyPatternRecognition
**[Should read this in github]**

This is a library about AI model. All the model ideas are come from inclass lecture.

You can see the class github [here](https://github.com/ekapolc/Pattern_2024) which is created by Aj. Ekapol.

## Provided model
- [K-nearest neighbor](#K-nearst-neighbor)
- [Linear regression](#Linear-regression)

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
