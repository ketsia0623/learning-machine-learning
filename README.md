# learning-machine-learning
I'm learning machine learning!

# Machine Learning Intro
- **Machine Learning (ML)**: Statistical computer algorithms that improve automatically through data.
- Differs from traditional programming: Algorithms infer the best approach from the data itself.
- Subset of Artificial Intelligence.
- Main goals:
  - Understand problems solved by ML
  - Learn types of ML
  - Review the ML process for supervised learning

# Numpy

## 1. Introduction
- **NumPy**: Core Python library for numerical and scientific computing.
- Basis for many data science libraries (Pandas, SciPy, scikit-learn).
- Provides **N-dimensional arrays** and fast, vectorized operations.


## 2. Section Goals
1. Understand NumPy
2. Create arrays:
   - From Python lists
   - Using built-in functions
   - Generating random data
3. Retrieve array information (slicing & indexing)
4. Perform basic NumPy operations
5. Practice with exercises

## 3. What is NumPy?
- **N-dimensional array object**: `numpy.ndarray`
- Supports:
  - **Broadcasting**: Apply operations to entire arrays without loops
  - **Mathematical functions**: Linear algebra, statistics, trig, random

## 4. Why Use NumPy?
- More **memory-efficient** and **faster** than Python lists
- Simplifies array operations with **vectorized code**

## 5. Creating NumPy Arrays
```python
import numpy as np

# From Python list
arr = np.array([1, 2, 3])

# Built-in functions
zeros = np.zeros((2, 3))     # 2x3 array of zeros
ones = np.ones((3, 3))       # 3x3 array of ones
range_arr = np.arange(0, 10, 2)  # [0, 2, 4, 6, 8]
linspace_arr = np.linspace(0, 1, 5)  # 5 evenly spaced numbers

# Random data
rand_uniform = np.random.rand(2, 2)     # Uniform [0, 1)
rand_normal = np.random.randn(2, 2)     # Normal distribution
rand_int = np.random.randint(0, 10, 5)  # Random integers
```


# Machine Learning Concepts

## 1. ML Pathway
1. Real-world problem/question
2. Collect & store data
3. Clean & organize data
4. Exploratory Data Analysis (EDA)
5. Build ML models:
   - **Supervised Learning**: Predict outcomes
   - **Unsupervised Learning**: Discover patterns
6. Deliver a data product/service

## 2. Why Use Machine Learning?
- Solves problems such as:
  - Credit scoring
  - Insurance risk
  - Price forecasting
  - Spam filtering
  - Customer segmentation
- Can handle complex problems like handwriting recognition and spam detection.
- **Key requirement**: Good quality data.
- Most time spent on data cleaning/organization, not algorithm coding.

## 3. Types of Machine Learning
### Supervised Learning
- Uses **historical, labeled data** to predict a value.
- Two types:
  1. **Classification**: Predict categories (e.g., spam vs. not spam, tumor type)
  2. **Regression**: Predict continuous values (e.g., prices, loads, scores)

### Unsupervised Learning
- Uses **unlabeled data** to discover patterns (e.g., clustering customers by behavior).
- Harder to evaluate performance without known “correct” answers.

## 4. Supervised ML Process
1. **Collect & prepare data**  
   - Separate into **Features (X)** and **Label (y)**
2. **Split data** into:
   - Training set
   - Test set
3. **Train model** on training set
4. **Evaluate model** on test set
5. **Adjust hyperparameters** if needed
6. **Repeat** training & evaluation until performance is acceptable
7. **Deploy model** for real-world use

## 5. Example: House Price Prediction
- Features: Area, Bedrooms, Bathrooms
- Label: Price
- ML model learns feature importance from historical sales data to predict future prices.

*Introduction to Statistical Learning (ISLR)*



# Linear Regression
## 1. Overview
- First ML algorithm covered — one of the oldest.
- Topics:
  - Theory & history
  - Ordinary Least Squares (OLS)
  - Cost function
  - Gradient descent
  - Scikit-learn implementation
  - Polynomial regression
  - Regularization
  - Performance evaluation

## 2. History
- Originated in the 1700s for astronomy/navigation calculations.
- Contributors:
  - Roger Cotes (1722), Tobias Mayer (1750), Boscovich (1757), LaPlace (1788)
  - Legendre (1805) — published least squares
  - Gauss (1809) — claimed earlier invention
  - Robert Adrain (1808)
- Goal: Fit a straight-line relationship between variables to make predictions.

## 3. Core Concept
- **Simple Linear Regression**:  
  \( y = m x + b \) — predicts continuous target `y` from one feature `x`.
- **Goal**: Minimize residual errors between actual values and predictions.
- **Residuals**: \( y - \hat{y} \)  
  Squared to avoid canceling and simplify math (Sum of Squared Errors).

## 4. Ordinary Least Squares (OLS)
- Minimizes the sum of squared residuals.
- Analytical solution possible for **one feature**:
  \[
  m = \frac{\sum (x_i - \bar{x})(y_i - \bar{y})}{\sum (x_i - \bar{x})^2}, \quad b = \bar{y} - m \bar{x}
  \]
- For **multiple features**: Represent data as matrix \( X \), coefficients as \( \beta \).
- Not scalable for many features — use **gradient descent**

## 5. Cost Function
- Measures model error:
  \[
  J(\beta) = \frac{1}{2m} \sum_{i=1}^m (y_i - \hat{y}_i)^2
  \]
- **Goal**: Find \(\beta\) values minimizing \(J(\beta)\)

## 6. Gradient Descent
- Iteratively updates coefficients:
  \[
  \beta := \beta - \alpha \frac{\partial J}{\partial \beta}
  \]
  - \(\alpha\) = learning rate.
- Steps:
  1. Initialize \(\beta\) values.
  2. Calculate gradient.
  3. Update coefficients in direction of **negative gradient**.
  4. Repeat until convergence.

## 7. Scikit-Learn Workflow
```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

# Evaluate
mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
rmse = mse ** 0.5
```

### Scikit-Learn Overview

- NumPy has built-in capabilities for simple linear regression, but for more complex models, **Scikit-Learn** is needed.
- **Scikit-Learn**:
  - Library containing many machine learning algorithms.
  - Uses a generalized “estimator API” framework — importing, fitting, and using models is uniform across algorithms.
  - This uniform framework allows easy swapping of algorithms and testing various approaches.
  - Convenience tools:
    - Train/test split functions
    - Cross-validation tools
    - Variety of reporting metric functions
  - “One-stop shop” for many ML needs.

- **Philosophy**:
  - Focuses on applying models and performance metrics.
  - Pragmatic, industry-style approach (not academic model parameter explanation).
  - Academic users who prefer detailed statistical reporting (e.g., significance levels) may explore the `statsmodels` library.

### Scikit-Learn Framework for Supervised ML Process

1. Perform **Train/Test split**.
2. Four main components after split:
   - **X_train**, **X_test**
   - **y_train**, **y_test**
3. Use `train_test_split` (and more advanced cross-validation if needed):
   ```python
   from sklearn.model_selection import train_test_split
   X_train, X_test, y_train, y_test = train_test_split(X, y)


# Logistic Regression
