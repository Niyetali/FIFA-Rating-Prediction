#!/usr/bin/env python
# coding: utf-8

# # Dependencies loading

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# # Data loading

# In[5]:


fifa_url = "https://raw.githubusercontent.com/Niyetali/FIFA-Rating-Prediction/main/output"

# importing datasets
X_train = pd.read_csv(f"{fifa_url}/X_train_fe.csv", sep=',')
X_test = pd.read_csv(f"{fifa_url}/X_test_fe.csv", sep=',')
y_train = pd.read_csv(f"{fifa_url}/y_train.csv", sep=',')
y_test = pd.read_csv(f"{fifa_url}/y_test.csv", sep=',')

pd.set_option('display.max_columns', None)

# Print shapes of the datasets
print(X_train.shape, X_test.shape)
print(y_train.shape, y_test.shape)


# In[7]:


X_train.head()


# ##### Let's drop the `Name` column:

# In[10]:


# Dropping the 'Name'
X_train = X_train.drop(columns=['Name'], errors='ignore')
X_test = X_test.drop(columns=['Name'], errors='ignore')


# # OLS model

# In[13]:


X_train = sm.add_constant(X_train) # add constant
X_test = sm.add_constant(X_test) # add constant

# Fit the OLS regression model
ols_model = sm.OLS(y_train, X_train).fit()

# Print the summary of the regression
print(ols_model.summary())


# In[15]:


# Make predictions
y_pred = ols_model.predict(X_test)


# # Model Evaluation

# In[18]:


# Calculate the metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# Calculate Adjusted R²
n = len(y_test)
p = X_test.shape[1] - 1

adj_r2 = 1 - ((1 - r2) * (n - 1) / (n - p - 1))

metrics = {"Metric": ["MAE", "MSE", "RMSE", "R²", "Adj. R²"],
            "Value": [mae, mse, rmse,r2, adj_r2]}

metrics = pd.DataFrame(metrics)

metrics


# In[20]:


# Plotting the metrics as a bar chart
plt.figure(figsize=(12, 6))
plt.barh(metrics['Metric'], metrics['Value'], color='steelblue')
plt.xlabel('Value', fontsize=12)
plt.title('Performance Metrics', fontsize=16)
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.show()


# Overall, the model's performance is quite strong for a linear regression. With an R² of approximately 0.93, it explains nearly 93% of the variance in the data, indicating a good fit. The MAE, MSE, and RMSE values show that the model's predictions are relatively close to the actual values, with an average error of around 1.66 units. The adjusted R² further confirms that the model is not overfitting and generalizes well. All in all, these results suggest that the linear regression model is performing well, especially considering the simplicity of the approach.
