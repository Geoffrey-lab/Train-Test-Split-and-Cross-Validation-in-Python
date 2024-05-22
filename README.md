# Train-Test Split and Cross-Validation in Python

This repository contains a Jupyter Notebook that provides a comprehensive guide on implementing the train-test split and cross-validation techniques for evaluating machine learning models. The notebook includes detailed explanations, code examples, and visualizations to help you understand and apply these concepts effectively.

## Notebook Content Overview

### 1. Introduction to Train-Test Split
- Explanation of the train-test split technique
- Benefits and purposes of splitting the dataset

### 2. Two-Way Split
- Splitting the dataset into training and testing sets
- Visualizing the split data

### 3. Three-Way Split
- Introduction to the three-way split: training, validation, and testing sets
- The purpose of each subset in model training and evaluation

### 4. Cross-Validation
- Explanation of cross-validation and its importance
- Detailed look at K-fold cross-validation
- How to implement cross-validation in Python

### 5. Practical Implementation in Python
- Importing necessary libraries: `numpy`, `pandas`, `matplotlib`, `seaborn`, and `scikit-learn`
- Loading and exploring the dataset

### 6. Using `train_test_split` from scikit-learn
- Splitting the dataset into features (X) and response (y)
- Performing the train-test split and visualizing the results

### 7. Training and Evaluating a Linear Model
- Training a linear regression model using the training data
- Plotting the regression line over the training data
- Evaluating model performance on both training and testing sets using Mean Squared Error (MSE) and R-squared metrics

### 8. Addressing Overfitting
- Identifying overfitting through discrepancies in model performance on training and testing sets
- Introduction to techniques for mitigating overfitting

## Key Code Snippets

### Data Loading and Preparation
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# Load the dataset
df = pd.read_csv('https://raw.githubusercontent.com/Explore-AI/Public-Data/master/Data/regression_sprint/regression_sprint_data_1.csv', index_col=0)
df.head(10)
```

### Performing Train-Test Split
```python
# Split the dataset into features (X) and response (y)
y = df['ZAR/USD']
X = df.drop('ZAR/USD', axis=1)

# Perform the train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=50)

# Plot the splitting results
plt.scatter(X_train, y_train, color='green', label='Training')
plt.scatter(X_test, y_test, color='darkblue', label='Testing')
plt.legend()
plt.show()
```

### Training a Linear Model
```python
# Initialize and fit the linear regression model
lm = LinearRegression()
lm.fit(X_train, y_train)

# Extract model parameters
a = float(lm.intercept_)
b = lm.coef_
print("Slope:\t\t", b)
print("Intercept:\t", float(a))

# Generate values that fall along the regression line
gen_y_train = lm.predict(X_train)

# Plot the training data and the regression line
plt.scatter(X_train, y_train, color='green', label='Training data')
plt.plot(X_train, gen_y_train, color='red', label='Regression line')
plt.legend()
plt.show()
```

### Model Evaluation
```python
# Evaluate the model on the training data
print("Training:")
print('MSE:', metrics.mean_squared_error(y_train, gen_y_train))
print('R_squared:', metrics.r2_score(y_train, gen_y_train))

# Evaluate the model on the testing data
gen_y_test = lm.predict(X_test)
plt.scatter(X_test, y_test, color='darkblue', label='Testing data')
plt.plot(X_test, gen_y_test, color='red', label='Regression line')
plt.legend()
plt.show()

print("Testing:")
print('MSE:', metrics.mean_squared_error(y_test, gen_y_test))
print('R_squared:', metrics.r2_score(y_test, gen_y_test))
```

## Conclusion
The notebook provides a detailed introduction to the concepts of train-test split, three-way split, and cross-validation. It demonstrates how to implement these techniques in Python using practical examples and evaluates the performance of a linear regression model. By following this guide, you will gain a solid understanding of how to split datasets for training, validation, and testing, and how to assess model performance effectively.

Feel free to explore the notebook, modify the code, and apply these techniques to your own datasets. Contributions and feedback are welcome!
