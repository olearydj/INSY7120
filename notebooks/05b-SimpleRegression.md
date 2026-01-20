---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.19.0
kernelspec:
  display_name: insy7120-py4sml
  language: python
  name: insy7120-py4sml
---

# Regression

+++

## Setup

+++

The following cell sets up the Colab environment. No changes are made if run locally.

```{code-cell} ipython3
# If running on Colab, set up the environment
import sys
if 'google.colab' in sys.modules:
    # !pip install requests wquantiles
    !mkdir -p /content/data
    %cd /content
    !wget -q https://raw.githubusercontent.com/olearydj/INSY7120/refs/heads/main/notebooks/common.py -O common.py

import common
```

```{code-cell} ipython3
%matplotlib inline

import matplotlib.pylab as plt
import numpy as np
import pandas as pd
```

## Simple Linear Regression

+++

This example is from HOML, chapter 1:
https://colab.research.google.com/github/ageron/handson-ml3/blob/main/01_the_machine_learning_landscape.ipynb

Load life satisfaction data and look at the first few rows.

```{code-cell} ipython3
# Download and prepare the data
data_root = "https://github.com/ageron/data/raw/main/"
lifesat = pd.read_csv(data_root + "lifesat/lifesat.csv")
lifesat.head()
```

The data is a table of life satisfaction rating for selected countries. Gross domestic product (GDP) is also given. Use `info` and `describe` dataframe methods to learn more about what we are working with.

```{code-cell} ipython3
lifesat.info()
```

```{code-cell} ipython3
lifesat.describe()
```

What is the relationship between GDP and life satisfaction?

- GDP is a numerical predictor
- Life satisfaction is a numerical outcome

Predict quantitative response on the basis of a single predictor → simple linear regression.

+++

### Scikit-learn Process

We'll use scikit-learn to perform this analysis. The general process is consistent across all model types:

1. Prepare the data, e.g. split out predictors (`X`) and outcome (`y`).
2. Initialize the selected model with desired parameters, e.g. `LinearRegression`
3. Fit the model using training data, e.g. `model.fit(X, y)`
4. Evaluate the model performance and tune / adjust until satisfied.
5. Use the predict method of the resulting model to get predictions for new data, e.g. `model.predict(x_0)`.

At most steps we will explore the results, adjust, and iterate as required.

+++

### Prepare the Data

We've already explored the data. No cleaning appears necessary, so let's extract `X` and `y` from `lifesat`.

```{code-cell} ipython3
X = lifesat[["GDP per capita (USD)"]].values
y = lifesat[["Life satisfaction"]].values
```

A few notes are in order.

First, about the notation. Capital `X` and lowercase `y` are used by convention. Here, capitalization is intended to denote a matrix, not that it is a random variable. There will normally be more than a single predictor, so `X` is assumed to be size `n x p`, where `n` is the number of observations and `p` is the number of predictors. Similarly, `y` is expected to be a single output for each set of `n` observed values, `X`.

In the particular example of Simple Linear Regression, `X` is a single column, but the capital notation is maintained by convention, signalling that it is the predictor.

Second, note the double-bracket notation above, e.g. `lifestat[["GDP per capita (USD)"]].values`. This is done to ensure that `X` and `y` remain 2-dimensional arrays (matricies) rather than 1-dimensinal arrays (vectors). Most scikit-learn estimators, including linear regression expect:

- `X` to be 2D with shape `(num_samples, num_features)`
- `y` can be either 1D or 2D, but it is often safer to keep y in the same format as `X`

The double-bracket notation should be interpreted as follows. The outer pair of brackets denote the standard column selection operation in pandas. The inner pair define a list of column lables. When a list of column labels is provided, pandas will always generate a 2D array of size `n x p`.

For example, if you wanted `X` to include both GDP and country, specify both columns:

```{code-cell} ipython3
lifesat[["GDP per capita (USD)", "Country"]].values
```

But in this case, `p` is 1 and `n` is 27, so both `X` and `y` 1 column and 27 rows:

```{code-cell} ipython3
X.shape
```

```{code-cell} ipython3
y.shape
```

Had the inner pair of brackets been omitted, `X` would be a 1D array, which is not suitable for scikit-learn.

```{code-cell} ipython3
lifesat["GDP per capita (USD)"].shape
```

Let's visualize the relationship between `X` and `y` with a scatter chart.

```{code-cell} ipython3
# Visualize the data
lifesat.plot(kind='scatter', grid=True,
             x="GDP per capita (USD)", y="Life satisfaction")
# specify ranges for both axes, recall that the underbar character is ignored
plt.axis([23_500, 62_500, 4, 9])
plt.show()
```

Based on the general shape of the data, this appears to be a good candidate for SLR.

+++

### Initialize the Model

Fit the data using the scikit-learn's `LinearRegression` class.

```{code-cell} ipython3
# import and instantiate the linear model
from sklearn.linear_model import LinearRegression
model = LinearRegression()
print(model)
```

To get more information on the model, use Python's `help()` system. When using a Jupyter Notebook, the alternative `?` syntax provides a more readable format.

```{code-cell} ipython3
LinearRegression?
```

### Fit the Training Data

Let's train the model!

```{code-cell} ipython3
# Train the model
model.fit(X, y)
```

```{code-cell} ipython3
model.fit?
```

```{code-cell} ipython3
model.coef_, model.intercept_
```

### Evaluate Model Performance

+++

Let's start by visualizing the line of best fit by combining the scatter plot with a line of the predictions.

```{code-cell} ipython3
plt.scatter(X, y)
plt.plot(X, model.predict(X), color='red')
plt.xlabel("GDP per capita (USD)")
plt.ylabel("Life satisfaction")
plt.axis([23_500, 62_500, 4, 9])
plt.grid(True)
plt.show()
```

Next let's calculate some common measures of model fit, starting with $R^2$.

```{code-cell} ipython3
r2_score = model.score(X, y)
print(f"R-squared: {r2_score:.4f}")
```

This result suggests that approximately 72.7% of the variance in life satisfaction ($y$) can be explained by differences in GDP per capita ($X$). The remaining 27.3% of variance can be attributed to:

- Reducible error
  - model form: linear may be too simple (bias)
  - predictors: may be missing factors important to life satisfaction
- Irreducible error
  - random effects: random fluctuations and inherent variablity in life satisfaction
  - measurement error: how do you measure the outcome consistently, accurately?
  - unidentified or unmeasureable factors

+++

Now let's look at three other measures, all based on residuals:

- Residual Standard Error (RSE), also known as the standard error of the regression: $$RSE = \sqrt{\frac{1}{n-2}RSS} \text{, where } RSS = \sum(y_i - \hat{y}_i)^2$$
- Mean Squared Error (MSE), the average of squared residuals: $$MSE = \frac{1}{n}\sum(y_i - \hat{y}_i)^2$$
- Root Mean Squared Error (RMSE), which is simply the square root of MSE: $$RMSE = \sqrt{MSE}$$

We covered RSE in the previous lecture as it is recommended by the primary text (ISL). It is like RMSE, but divides by $n-2$ instead of $n$ to account for estimating two parameters. This is more commonly used in the context of statistical inference, but there is no direct way to calculate it in scikit-learn.

It is easy enough (and instructive!) to calculate directly, thanks to numpy / pandas vectorized math:

```{code-cell} ipython3
# subtract the predictions from the known values to get the residuals
residuals = y - model.predict(X)

# take the sum of the residuals squared
rss = (residuals ** 2).sum()

# number of observations is just the length of y or x
n = len(y)

# calculate rse
rse = np.sqrt(rss / (n - 2))

print(f"RSE: {rse:.4f}")
```

RMSE is more commonly used in ML and predictive modeling. It is in the same scale of the response variable, making it easier to interpret.

```{code-cell} ipython3
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y, model.predict(X))
rmse = np.sqrt(mse)
print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
```

Note that:

$$MSE = \frac{1}{n}RSS \rightarrow RSS = n(MSE) \text{, and}$$
$$RSE = \sqrt{\frac{1}{n-2}RSS} \rightarrow RSE = \sqrt{\frac{n}{n-2}MSE}$$

This allows us to use MSE to calculate RSE directly.

```{code-cell} ipython3
rse = np.sqrt((n * mse) / (n - 2))
print(f"RSE (from MSE): {rse:.4f}")
```

RSE is slightly larger than RMSE, owing to the smaller divisor (`n-2` vs `n`). This adjustment makes RSE a more conservative (larger) estimate of model error.

We can also generate a residual plot, Q-Q plot, and histogram of residuals, though it is not essential to check the model assumptions for prediction.

```{code-cell} ipython3
# Create three subplots in a row
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))

# Residual plot
ax1.scatter(model.predict(X), residuals)
ax1.axhline(y=0, color='r', linestyle='--')
ax1.set_xlabel("Predicted values")
ax1.set_ylabel("Residuals")
ax1.set_title("Residual Plot")
ax1.grid(True)

# Q-Q plot
from scipy import stats
stats.probplot(residuals.ravel(), plot=ax2)
ax2.set_title("Q-Q Plot")

# Histogram of residuals
ax3.hist(residuals, bins=6, edgecolor='black')
ax3.set_xlabel("Residuals")
ax3.set_ylabel("Frequency")
ax3.set_title("Histogram of Residuals")

plt.tight_layout()
plt.show()
```

Brief interpretation:
1. Residual Plot: No obvious pattern, though slight fan shape might indicate minor heteroscedasticity (increasing variance with higher predicted values)
2. Q-Q Plot: Points follow diagonal line reasonably well, suggesting residuals are approximately normally distributed
3. Histogram: Roughly bell-shaped but with small sample size, hard to make strong claims about normality

For prediction purposes, these diagnostics suggest the model is adequate. The violations of assumptions aren't severe enough to invalidate predictions, though they might be more concerning if we were doing inferential analysis.

+++

### Make Predictions

Finally, let's make a prediciton for Cyprus, which had a per capita GDP of $37,655.20 in 2020.

```{code-cell} ipython3
# Make a prediction
X_new = [[37_655.2]]  # Cyprus' GDP per capita in 2020
model.predict(X_new)
```

Confirm that Cyprus was not in the original dataset using a membership test (`in`) on the values of the column.

```{code-cell} ipython3
"Cyprus" in lifesat["Country"].values
```
