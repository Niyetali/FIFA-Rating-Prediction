#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from category_encoders import TargetEncoder


# ## Data Loading

# In[4]:


fifa_url = "https://raw.githubusercontent.com/Niyetali/FIFA-Rating-Prediction/main/output"

# importing datasets
X_train = pd.read_csv(f"{fifa_url}/X_train.csv", sep=',')
X_test = pd.read_csv(f"{fifa_url}/X_test.csv", sep=',')
y_train = pd.read_csv(f"{fifa_url}/y_train.csv", sep=',')
y_test = pd.read_csv(f"{fifa_url}/y_test.csv", sep=',')

pd.set_option('display.max_columns', None)

# Print shapes of the datasets
print(X_train.shape, X_test.shape)
print(y_train.shape, y_test.shape)


# In[5]:


X_train.head(11)


# For the models we will use, we need to convert the categorical data into numerical format.

# # Feature Engineering

# ## Binary Encoding:

# In[9]:


X_train['Loaned From'].unique()


# The "Loaned From" column indicates whether a player is on loan (club name) or not ("Permanent"). We'll simplify this by applying binary encoding: `1` for players on loan and `0` for those not on loan. This transformation makes the data cleaner and easier to use for the models.

# In[12]:


# Transform "Loaned From" into binary "On_loan"
X_train['On_loan'] = (X_train['Loaned From'] != 'Permanent').astype(int)
X_test['On_loan'] = (X_test['Loaned From'] != 'Permanent').astype(int)

# Drop the original "Loaned From" column
X_train = X_train.drop(columns=['Loaned From'])
X_test = X_test.drop(columns=['Loaned From'])

# Display the first few rows of X_train
X_train.head()


# In[13]:


X_train['On_loan'].unique()


# In[14]:


X_test['On_loan'].unique()


# ## Label Encoding:

# ### `Body Type`

# In[17]:


X_train['Body Type'].value_counts()


# In[18]:


# Create a label (category) encoder object
le = LabelEncoder()

# Creating a dictionary to store LabelEncoder objects
le = {}

# Initialize LabelEncoder
le['Body_Type'] = LabelEncoder()

# Fit and transform 'Body Type'
X_train['Body_Type'] = le['Body_Type'].fit_transform(X_train['Body Type'])
X_test['Body_Type'] = le['Body_Type'].transform(X_test['Body Type'])

# Drop the original 'Body Type' column
X_train = X_train.drop(columns=['Body Type'])
X_test = X_test.drop(columns=['Body Type'])

# Print encoding and decoding for 'Body Type'
encoded_values = range(len(le['Body_Type'].classes_))
decoded_values = le['Body_Type'].classes_

for enc, dec in zip(encoded_values, decoded_values):
    print(f"{enc} -> {dec}")


# ### `Work Rate`

# In[20]:


X_train['Work Rate'].value_counts()


# The `Work Rate` column will be split into two new features: `Attacking_Work_Rate`, which represent the player's effort in attack, and `Defensive_Work_Rate`, which represents their effort in defense. This transformation simplifies the analysis of a player's style and contributions. Once these features are created, label encoding is applied to prepare them for use in the model.

# #### Before proceeding, we'll replace any occurrences of `NA / NA` in the `Work Rate` column with the mode of the column. Since it is small compared to the total, this is an effective solution:

# In[23]:


mode = X_train['Work Rate'].mode()[0]

X_train['Work Rate'] = X_train['Work Rate'].replace('N/A/ N/A', mode)
X_test['Work Rate'] = X_test['Work Rate'].replace('N/A/ N/A', mode)

X_train['Work Rate'].value_counts()


# #### Lets first split the `Work Rate` column into `Attacking_Work_Rate` and `Defensive_Work_Rate`:

# In[25]:


X_train[['Attacking_Work_Rate', 'Defensive_Work_Rate']] = X_train['Work Rate'].str.split('/', expand=True)
X_test[['Attacking_Work_Rate', 'Defensive_Work_Rate']] = X_test['Work Rate'].str.split('/', expand=True)

# And remove any extra spaces
X_train['Attacking_Work_Rate'] = X_train['Attacking_Work_Rate'].str.strip()
X_train['Defensive_Work_Rate'] = X_train['Defensive_Work_Rate'].str.strip()
X_test['Attacking_Work_Rate'] = X_test['Attacking_Work_Rate'].str.strip()
X_test['Defensive_Work_Rate'] = X_test['Defensive_Work_Rate'].str.strip()


# #### And apply Label Encoder:

# In[27]:


# Create a label (category) encoder object
le = LabelEncoder()

# Fit and transform 'Attacking_Work_Rate'
X_train['Attacking_Work_Rate'] = le.fit_transform(X_train['Attacking_Work_Rate'])
X_test['Attacking_Work_Rate'] = le.transform(X_test['Attacking_Work_Rate'])

# Create mapping for 'Attacking_Work_Rate'
attacking_work_rate_mapping = dict(zip(le.classes_, le.transform(le.classes_)))

# Fit and transform 'Defensive_Work_Rate'
X_train['Defensive_Work_Rate'] = le.fit_transform(X_train['Defensive_Work_Rate'])
X_test['Defensive_Work_Rate'] = le.transform(X_test['Defensive_Work_Rate'])

# Create mapping for 'Defensive_Work_Rate'
defensive_work_rate_mapping = dict(zip(le.classes_, le.transform(le.classes_)))

# Drop the original 'Work Rate' column
X_train = X_train.drop(columns=['Work Rate'])
X_test = X_test.drop(columns=['Work Rate'])

# Print mappings
print("Attacking Work Rate Mapping:", attacking_work_rate_mapping)
print("Defensive Work Rate Mapping:", defensive_work_rate_mapping)


# #### Looks good!!

# ## One-Hot Encoding

# ### `Position`

# In[31]:


X_train['Position'].unique()


# In[32]:


# One-hot encoding for the 'Position' column
X_train = pd.get_dummies(X_train, columns=['Position'], drop_first=True)
X_test = pd.get_dummies(X_test, columns=['Position'], drop_first=True)

X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

# Ensure binary values for the one-hot encoded columns are integers
for col in X_train.filter(like='Position_').columns:
    X_train[col] = X_train[col].astype(int)
    X_test[col] = X_test[col].astype(int)

# Rename columns to remove the 'Position_' prefix
X_train.rename(columns=lambda x: x.replace('Position_', ''), inplace=True)
X_test.rename(columns=lambda x: x.replace('Position_', ''), inplace=True)

print(X_train.head(11))


# ## Target Encoding

# ### `Nationality` & `Club`

# During the EDA, we observed that the `Nationality` and `Club` columns have many unique values. However, these columns might contain relevant information that impacts the target variable, which is the `Overall` rating. To capture this relationship, we'll apply target encoding and replace these categorical values with the mean value of the target for each category.

# In[36]:


# Create a target (category) encoder object
te = TargetEncoder(cols=['Nationality', 'Club'])

# Fit and transform 'Nationality' and 'Club' columns
X_train[['Nationality', 'Club']] = te.fit_transform(X_train[['Nationality', 'Club']], y_train)
X_test[['Nationality', 'Club']] = te.transform(X_test[['Nationality', 'Club']])

# Check the transformed columns in X_train
print(X_train[['Nationality', 'Club']].head(11))


# ## Reviewing the Data

# In[38]:


X_train.head()


# In[39]:


X_train.info()


# #### And for test:

# In[41]:


X_test.head()


# In[42]:


X_test.info()


# ## Additional Features:

# Now, let’s create new features using the ones we already have. This helps us discover new patterns and make our analysis even better!

# ### Additional Financial Features:

# In[46]:


# Shows how much a player is worth relative to their wage.
X_train['Value_to_Wage_Ratio'] = X_train['Value'] / X_train['Wage']
X_test['Value_to_Wage_Ratio'] = X_test['Value'] / X_test['Wage']

# Measures potential market gain/loss by comparing release clause to market value.
X_train['Surplus_Value'] = X_train['Release Clause'] - X_train['Value']
X_test['Surplus_Value'] = X_test['Release Clause'] - X_test['Value']

print(X_train[['Value_to_Wage_Ratio', 'Surplus_Value']].head(11))


# ### Additional Physical Features:

# In[48]:


## BMI and Physical
#  BMI: Calculates a player's body mass index.  
#  Physical: Approximates physical dominance using height and weight averages.

# BMI
X_train['BMI'] = X_train['Weight'] / (X_train['Height'] / 100) ** 2
X_test['BMI'] = X_test['Weight'] / (X_test['Height'] / 100) ** 2

# Physical
X_train['Physical'] = (X_train['Height'] + X_train['Weight']) / 2
X_test['Physical'] = (X_test['Height'] + X_test['Weight']) / 2

print(X_train[['BMI', 'Physical']].head(10))


# ### Additional Age-Related Features:

# We'll create an `Age_Bucket` feature that groups players into age categories: `Young`, `Prime`, and `Veteran`. After defining these categories, we'll convert them into numerical values using label encoding.

# In[51]:


# Define the age_bucket function
def age_bucket(age):
    
    if age <= 21:         # below or equal to 21y
        return 'Young'
    elif 22 <= age <= 30: # between 22y and 30y
        return 'Prime'
    else:                 # above 30y
        return 'Veteran'

# Fit and transform 'Age_Group'
X_train['Age_Group'] = X_train['Age'].apply(age_bucket)
X_test['Age_Group'] = X_test['Age'].apply(age_bucket)

# Create a label (category) encoder object
le = LabelEncoder()

# Fit and transform 'Age_Group' column
X_train['Age_Group'] = le.fit_transform(X_train['Age_Group'])
X_test['Age_Group'] = le.transform(X_test['Age_Group'])

# Create mapping for 'Age_Group'
age_group_mapping = dict(zip(le.classes_, le.transform(le.classes_)))

print(age_group_mapping)


# ### Potential Related Features

# In[53]:


X_train['Value_to_Potential'] = X_train['Value'] / X_train['Potential']
X_test['Value_to_Potential'] = X_test['Value'] / X_test['Potential']

X_train['Potential_Age_Ratio'] = X_train['Potential'] / X_train['Age']
X_test['Potential_Age_Ratio'] = X_test['Potential'] / X_test['Age']

print(X_train[['Value_to_Potential', 'Potential_Age_Ratio']].head(10))


# ### Reviewing the Data

# In[55]:


X_train


# In[56]:


X_test


# ## Feature Selection

# We simplify the dataset by keeping only the most relevant features, removing noise, and improving prediction accuracy. This ensures our analysis focuses on meaningful data for better results.

# In[59]:


# Columns related to Positions
exclude = [
    "ST", "GK", "RCB", "LCM", "LCB", "RCM", "LM", "LB", "RB", 
    "RM", "RW", "CDM", "CB", "LW", "RS", "LS", "LDM", "RDM", 
    "RWB", "CF", "RF", "LF", "LWB", "CM", "LAM", "RAM", "SUB", "RES"
]

# Lets drop the 'Names' column and Positions
X_train_filtered = X_train.drop(columns=[X_train.columns[0]] + exclude)

# And combine into a dataframe
data = X_train_filtered.copy()
data['Target'] = y_train

data.head(11)


# ### Next Steps - Correlation with target
# 
# We will start by checking the correlation of each feature with the target variable.

# In[61]:


# Correlation with Target

# Pearson correlation
pearson = data.corr()['Target'].sort_values(ascending=False)

# Spearman correlation
spearman = data.corr(method='spearman')['Target'].sort_values(ascending=False)


# Pearson is great for spotting linear relationships between two numbers. It's easy to use, widely trusted, and tells you how strong and in what direction the connection is—perfect for clean, normal data without outliers.

# In[63]:


print("Pearson Correlation:\n", pearson)


# Adding Spearman after Pearson helps catching non-linear relationships. If Pearson shows no strong connection, Spearman might reveal a monotonic trend, giving a fuller picture of how our variables relate. It’s like double-checking your work from a different angle!

# In[65]:


print("\nSpearman Correlation:\n", spearman)


# ### Checking Feature Correlations
# 
# For now, we’ll remove features with weak correlations to the target, such as `Height`, `On_loan`, and `Physical`, as they a minimal predictive value. This simplifies the dataset, allowing the model to focus on the most impactful data. However, we’ll keep the other features for now and examine how they correlate with each other. This step will help us identify redundancies and refine the feature set further, ensuring we retain only the most meaningful variables for modeling and avoid collinearity.

# In[67]:


# Columns to drop based on low correlation
drop = ['Height', 'On_loan', 'Physical']

X_train = X_train.drop(columns=drop, errors='ignore')
X_test = X_test.drop(columns=drop, errors='ignore')
data = data.drop(columns=drop, errors='ignore')


# ### Correlation Heatmap

# In[69]:


correlation_matrix = data.corr()

# Plot the heatmap
plt.figure(figsize=(16, 12))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", 
            cmap="coolwarm", cbar=True, annot_kws={"size": 8})
plt.xticks(rotation=45, ha='right')
plt.title("Correlation Heatmap")
plt.show()


# ### Removing Highly Correlated Features
# 
# Based on the correlation heatmap, we identified features with strong correlations among themselves. High correlations between features can lead to redundancy and multicollinearity, which may negatively impact the model's interpretability and performance. To simplify the dataset and avoid redundancy, we decided to remove the following features:
# 
# #### Features to be removed:
# 1. **`Value`**, **`Release Clause`**, **`Surplus_Value`**, **`Wage`**:
#    - Strongly correlated with `Value_to_Potential`.
#    - We retained `Value_to_Potential` because it has a stronger correlation with the target.
# 
# 2. **`Weight`**:
#    - Strong correlation with `BMI` (r > 0.80).
#    - We kept `BMI` as the primary physical attribute because it has a stronger correlation with         the target.
#    - 
# 3. **`Potential_Age_Ratio`**:
#    - Strong negative correlation with `Age` (r > -0.80).
#    - We kept `Age` because it has a nice positive correlation with the target.

# In[73]:


# Columns to drop based on high correlation with other features
drop = ['Value', 'Release Clause','Surplus_Value', 
        'Wage', 'Potential_Age_Ratio']

X_train = X_train.drop(columns=drop, errors='ignore')
X_test = X_test.drop(columns=drop, errors='ignore')
data = data.drop(columns=drop, errors='ignore')


# In[80]:


correlation_matrix = data.corr()

# Plot the heatmap
plt.figure(figsize=(16, 12))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", 
            cmap="coolwarm", cbar=True, annot_kws={"size": 8})
plt.xticks(rotation=45, ha='right')
plt.title("Correlation Heatmap")
plt.show()


# - We can also consider removing the `Weight` feature, as it shows a low correlation with the target variable and has some correlation with `BMI`. Additionally, it is slightly less predictive compared to other physical features.
# - Let's also remove `Value_to_Wage_Ratio`, as it has a low correlation with all features and the target, contributing little to no value.

# In[88]:


# Columns to drop based on high correlation with other features
drop = ['Weight', 'Value_to_Wage_Ratio']

X_train = X_train.drop(columns=drop, errors='ignore')
X_test = X_test.drop(columns=drop, errors='ignore')
data = data.drop(columns=drop, errors='ignore')


# In[91]:


correlation_matrix = data.corr()

# Plot the heatmap
plt.figure(figsize=(16, 12))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", 
            cmap="coolwarm", cbar=True, annot_kws={"size": 8})
plt.xticks(rotation=45, ha='right')
plt.title("Correlation Heatmap")
plt.show()


# ## Reviewing the Data

# In[100]:


X_train.columns


# In[104]:


X_test.columns


# In[107]:


print(X_train.shape, X_test.shape)


# In[132]:


X_train.isnull().sum()


# In[134]:


X_test.isnull().sum()


# ## And exporting it

# In[137]:


X_train.to_csv("X_train_fe.csv", index=False)
X_test.to_csv("X_test_fe.csv", index=False)

print("Datasets successfully exported as CSV files.")

