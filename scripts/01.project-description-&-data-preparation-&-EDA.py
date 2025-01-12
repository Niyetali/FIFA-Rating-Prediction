#!/usr/bin/env python
# coding: utf-8

# # Problem Description
# 
# **How do you uncover the stars behind the numbers?**
# 
# In football, player ratings aren’t just numbers—they’re a way to measure skill, potential, and reputation. But what if we could predict them? What if we could uncover what makes a player truly stand out?
# 
# This project takes on the challenge of predicting FIFA 23 player ratings using real data like age, club, position, height, and even contract details. From young talents full of potential to experienced stars, every player’s numbers tell a unique story.
# 
# But it’s not easy. A striker’s value isn’t judged the same way as a defender’s. Predicting ratings means understanding how different roles, skills, and attributes shape a player’s performance.
# 
# Why does this matter? Scouts could find hidden gems. Gamers could understand what makes their favorite players great. And fans could see football in a whole new light.
# 
# Can we crack the code behind these ratings? Let’s find out.

# ![FIFA 23](https://raw.githubusercontent.com/Niyetali/FIFA-Rating-Prediction/main/input/fifa-23.jpg)

# # Dataset Description
# 
# This dataset is focused on predicting the **Overall rating** of FIFA 23 players, which is the target variable. The remaining columns serve as features, providing real-world data to help uncover the patterns and factors that influence player ratings. Here’s a breakdown of the dataset:
# 
# ---
# 
# #### **Player Information**
# - **ID**: Unique identifier for each player.
# - **Name**: The player’s name.
# - **Age**: The player’s age.
# - **Photo**: Link to the player’s image.
# - **Nationality**: The country the player represents.
# - **Flag**: Link to the country’s flag.
# - **Height**: The player’s height.
# - **Weight**: The player’s weight.
# - **Body Type**: The player’s physical build.
# - **Real Face**: Indicates if the player’s face is represented in the game (binary):
#   Values: "Yes", "No".
# 
# ---
# 
# #### **Player Ratings and Attributes**
# - **Potential**: The player’s potential future rating.
# - **Special**: A unique attribute reflecting specialized skills.
# - **Preferred Foot**: The player’s dominant foot.
# - **International Reputation**: Global popularity rating.
# - **Weak Foot**: The player’s ability to use their weaker foot.
# - **Skill Moves**: The player’s ability to perform skill moves.
# - **Work Rate**: Effort levels in attack and defense.
# - **Position**: The player’s main playing position.
# 
# ---
# 
# #### **Club and Contract Information**
# - **Club**: The name of the club the player represents.
# - **Club Logo**: Link to the club’s logo.
# - **Joined**: Date the player joined their current club.
# - **Loaned From**: If applicable, the club the player is loaned from.
# - **Contract Valid Until**: The year the player’s contract expires.
# - **Release Clause**: The player’s release clause value.
# 
# ---
# 
# #### **Financial Information**
# - **Value**: The player’s market value.
# - **Wage**: Weekly wage of the player.
# 
# ---
# 
# #### **Miscellaneous**
# - **Kit Number**: The player’s jersey number.
# - **Best Overall Rating**: The highest rating achieved across all positions (mostly empty).
# 
# ---
# 
# #### **Target Variable**
# - **Overall**: The player’s current overall skill rating, which serves as the target variable for prediction.
# 
# # Dataset Characteristics
# - **Type**: Multivariate
# - **Focus**: Football Analytics
# - **Task**: Regression (predicting player ratings)
# - **Features**: Categorical, Numeric, Text
# - **Instances**: 17,000+ players
# - **Features**: 28 (plus target)
# 
# ---
# 
# # Challenges
# - **Complex Relationships**: Player stats, positions, and reputation interact in fascinating ways that need careful analysis.
# - **Wide Variety**: Features like nationality and club include many unique categories, adding complexity.
# - **Imbalanced Data**: Some attributes, like skill moves or reputation, are skewed and need fair handling.
# 
# ---
# 
# # Potential Applications
# - **Scouting Talent**: Spot rising stars and hidden gems based on key stats.
# - **Building Teams**: Find the perfect player for any role using data insights.
# - **Gaming Insights**: Understand how ratings are created to master team setups in FIFA.
# - **Analysis**: Explore how real-world performance translates into player ratings.

# # Dependencies loading

# In[557]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from math import pi

from sklearn.model_selection import train_test_split
from scipy.stats import chi2_contingency, pearsonr
from sklearn.preprocessing import LabelEncoder

# Utilities
import os
import warnings
warnings.filterwarnings('ignore')


# # Project setup

# In[560]:


fifa_url = 'https://raw.githubusercontent.com/Niyetali/FIFA-Rating-Prediction/refs/heads/main/input/FIFA23_official_data.csv'

# Configuration to display all columns (the dataset has many features)
pd.set_option('display.max_columns', None)


# # Data preparation

# In[563]:


df = pd.read_csv(fifa_url, sep=',')
df.head(10)


# In[564]:


df.shape


# # Dataset adjustment

# Let’s remove features that are intuitively unlikely to impact a footballer’s rating, simplifying the data.

# In[567]:


# List of columns to drop
columns_to_drop = ['Photo', 'Flag', 'Club Logo', 'Real Face', 'Kit Number',
                    'ID', 'Special', 'Joined', 'Contract Valid Until']

# Drop columns
df = df.drop(columns=columns_to_drop)

df.columns


# In[568]:


df.info()


# We can see that most columns are categorical, which means we’ll need to handle a significant amount of encoding to prepare the data for analysis.

# ## Handling Missing Values

# It goes without saying that it is crucial to deal with missing values to proceed further with dataset analysis and later, improve a model's performance. Thus, we will investigate whether to remove or fill these missing values.

# In[572]:


df.isnull().sum()


# In[573]:


# Best Overall Rating has too many missing values so,
# it would not make sense to keep this column.
df = df.drop(columns=['Best Overall Rating'])

# The original owner of the player might make sense to keep for now.
# Lets replace NaNs with Permanent for now, meaning players that are not on loan.
df['Loaned From'] = df['Loaned From'].fillna('Permanent')
df['Loaned From'] = df['Loaned From'].astype('category')

df['Loaned From'].head


# **Club**, **Body Type**, and **Position** have only a few missing values. Clubs can be filled with "Unknown," as this is unlikely to significantly impact our analysis. For Body Type, missing values could be filled based on **Height** and **Weight**—for example, shorter and heavier players might be categorized as "Stocky," while taller and lighter players could be labeled as "Lean." For **Position**, we can consider removing the rows with missing values.

# **BUT,** before addressing the missing values, we also need to clean and convert the variables into their appropriate data types.

# ### `Club`

# In[577]:


df['Club'] = df['Club'].astype('category')

# Setting "Unknown" as a new category 
df['Club'] = df['Club'].cat.add_categories('Unknown')

# Now lets fill the column
df['Club'] = df['Club'].fillna('Unknown')

df.isnull().sum()


# ### `Weight`

# In[579]:


df['Weight'].head(10)


# In[580]:


# Remove kg suffix 
df['Weight'] = df['Weight'].str.replace('kg', '').astype('int64')

df['Weight'].head(10)


# ### `Height`

# In[582]:


df['Height'].head(10)


# In[583]:


# Remove cm suffix 
df['Height'] = df['Height'].str.replace('cm', '').astype('int64')

df['Height'].head(10)


# ### `Body Type`

# In[589]:


df['Body Type'].unique()


# In[592]:


# Remove height suffix
df['Body Type'] = df['Body Type'].str.split('(', expand=True)[0].str.strip()
df['Body Type'] = df['Body Type'].astype('category')

df['Body Type'].unique()


# In[593]:


# Filling Body Type based on Height and Weight thresholds
def fill_body_type(row):    
    if pd.isnull(row['Body Type']):
        
        if row['Height'] > 185 and row['Weight'] > 80:
            return 'Stocky'
        elif row['Height'] < 175 and row['Weight'] < 70:
            return 'Lean'
        else:
            return 'Normal'
            
    return row['Body Type']

df['Body Type'] = df.apply(fill_body_type, axis=1)

df['Body Type'].unique()


# ### `Value`, `Wage` and `Release Clause`

# In[600]:


print(df[["Value", "Wage", "Release Clause"]])


# Lets convert `Value`, `Release Clause` and `Wage` into a float variable.

# In[603]:


def convert_currency(value):
    if isinstance(value, str):
        value = value.replace('€', '')  # first remove the euro sign
        
        if 'M' in value:  # for values in millions
            return float(value.replace('M', '')) * 1000000
        elif 'K' in value:  # for values in thousands
            return float(value.replace('K', '')) * 1000
        else:  # For values without suffix
            return float(value)
            
    return 0  # handle missing values

# applying the function to these features
df['Value'] = df['Value'].apply(convert_currency)
df['Wage'] = df['Wage'].apply(convert_currency)
df['Release Clause'] = df['Release Clause'].apply(convert_currency)

print(df[["Value", "Wage", "Release Clause"]])


# ### `Release Clause`

# The **Release Clause** column still has a few missing values. To handle this, we can fill the missing entries with the median of the column. This ensures the missing values are replaced with a representative number without skewing the data. Currently, there are 1,151 missing values in this column.

# In[608]:


release_clause_median = df['Release Clause'].median()

df['Release Clause'] = df['Release Clause'].fillna(release_clause_median)


# ### `Position`

# In[611]:


df['Position'].head(10)


# Next, let’s convert remaining relevant columns into categorical data types, starting with **Position**. Upon inspection, we can see that the **Position** column contains HTML code blended into the data. We’ll clean this column to remove any unwanted HTML code before proceeding.

# In[614]:


df['Position'] = df['Position'].apply(lambda x: x.split(">")[1] if isinstance(x, str) and ">" in x else x)

# Conversion
df['Position'] = df['Position'].astype('category')

# Verification 
df['Position'].unique()


# `Position` has only a few missing values so we can simply drop those instances.

# In[617]:


df = df.dropna(subset=['Position'])

df.isnull().sum()


# ### `Preferred foot`

# In[621]:


df["Preferred Foot"].head(10)


# #### Lets convert this feature into a binary variable:

# In[624]:


# Map "Right" to 1 and "Left" to 0
df['Preferred Foot'] = df['Preferred Foot'].map({'Right': 1, 'Left': 0})

df['Preferred Foot'].unique()


# ### `Work Rate` & `Body Type`

# In[628]:


df['Work Rate'] = df['Work Rate'].astype('category')
df['Body Type'] = df['Body Type'].astype('category')


# We converted objects to categories to save memory and speed up operations, especially when working with repeated or predefined values, making data handling faster and more efficient.

# ## Reviewing the Data

# ##### A final look at the dataset ensures it's clean and ready for splitting and further analysis.

# In[633]:


df.head(10)


# In[635]:


df.info()


# In[637]:


df.isnull().sum()


# In[639]:


df.shape


# # Dataset Splitting

# We split the dataset into:
# 
# - **Training (& Validation) dataset**: 80% of the data.
# - **Test dataset**: 20% of the data for final evaluation.
# 
# Test dataset will be used only for the final predictions! We assume that during the entire study Training does not have access to it and do not study its statistical properties.

# In[643]:


# Define features (X) and target (y)
X = df.drop('Overall', axis = 1)
y = df['Overall']

# Split into training and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[645]:


print(X_train.shape, X_test.shape)


# In[647]:


print(y_train.shape, y_test.shape)


# # EDA (Exploratory Data Analysis)

# ## Descriptive analyses of the data

# In[651]:


X_train.head(10)


# In[653]:


y_train.head(10)


# In[655]:


X_train.info()


# In[657]:


y_train.info()


# ## Target variable analysis

# ### Unique Values

# In[661]:


y_train.unique()


# ### Value Counts

# In[664]:


y_train.value_counts()


# ### Basic Statistics

# In[667]:


print(f"Mean: {y_train.mean()}")
print(f"Median: {y_train.median()}")
print(f"Mode: {y_train.mode()}")
print(f"Standard Deviation: {y_train.std()}")
print(f"Variance: {y_train.var()}")


# ### Top 10 Players by Overall Rating

# In[670]:


X_train_overall = X_train.copy()
X_train_overall['Overall'] = y_train

# Sort by Overall and select the top 10
xtrain_overall = X_train_overall.sort_values(['Overall'], ascending=False).head(10)

xtrain_overall[['Name', 'Overall', 'Potential', 'Club', 'Nationality', 'Age', 'Position']].style.background_gradient(cmap='Greens')


# **These are the Top 10 Players by Overall Rating** ->  R. Lewandowski, K. Mbappé, K. Benzema, L. Messi, M. Neuer,  
# M. Salah, V. van Dijk, Cristiano Ronaldo, N. Kanté, H. Kane.
# 
# ---
# 
# #### Interesting Facts:
# 
# - **3 French players** (K. Mbappé, K. Benzema, N.Kanté) make it to the top 10 list!  
# - **Paris Saint-Germain shines** with **2 stars** (K. Mbappé & L. Messi) making the cut.  
# - **No surprise! Top players come from elite clubs** like Bayern München, PSG, Real Madrid, and Liverpool.  
# - **Liverpool proves its strength** with 2 of their best (M. Salah & V. van Dijk) featured in the list.  
# - **Top positions dominate:** Most players in the top 10 are **forwards**, reflecting their influence on the game.  

# ### Bottom 10 Players by Overall Rating

# In[674]:


X_train_overall = X_train.copy()
X_train_overall['Overall'] = y_train

# Sort by Overall and select the top 10
xtrain_overall = X_train_overall.sort_values(['Overall'], ascending=True).head(10)

xtrain_overall[['Name', 'Overall', 'Potential', 'Club', 'Nationality', 'Age', 'Position']].style.background_gradient(cmap='Reds')


# As expected, the list is dominated by **reserves and substitutes** playing in **less competitive leagues**.

# ### Distribution of Players Ratings

# In[678]:


# Bar plot for Overall distribution
plt.figure(figsize=(12, 6))
sns.barplot(x=y_train.value_counts().index, y=y_train.value_counts().values)

plt.title('Overall Rating Distribution', fontsize=14)
plt.xlabel('Overall Rating')
plt.ylabel('Frequency')
plt.show()


# ## Features variables analysis

# ### `Potential`

# In[682]:


X_train['Potential'].mean()


# ### TOP 10 Players by Highest Potential

# In[685]:


X_train_potential = X_train.copy()
X_train_potential['Overall'] = y_train

# Sort by Potential and select the top 10
xtrain_potential = X_train_potential.sort_values(['Potential'], ascending=False).head(10)

# Select relevant columns and apply a gradient style
xtrain_potential[['Name', 'Overall', 'Potential', 'Club', 'Nationality', 'Age', 'Position']].style.background_gradient(cmap='Greens')


# **These are the Top 10 Players by Potential Rating** -> K. Mbappé, E. Haaland, Pedri, P. Foden, F. de Jong,  
# Vinícius Jr., G. Donnarumma, K. Benzema, Ederson, R. Lewandowski.
# 
# ---
# 
# #### Interesting Facts:
# 
# - **2 French players** (K. Mbappé and K. Benzema) feature in the top 10, showcasing France’s depth in talent.  
# - **PSG dominates** with **2 stars** (K. Mbappé & G. Donnarumma), proving their strong investments in future talent.
# - **Real Madrid CF** also makes a strong showing with 2 players (Vinícius Jr. and K. Benzema), continuing their legacy of nurturing top talent.    
# - **FC Barcelona’s** famed **La Masia** academy continues to nurture top talent like Pedri and F. de Jong, while **Pep Guardiola’s Manchester City** exemplifies how youth development translates into success at the highest level.  
# - **Younger players take the spotlight**—unlike the top Overall ratings, most players here are under 25, signaling the next generation of stars.  
# - **Strikers and attacking players** (K. Mbappé, E. Haaland) lead the list, but we also see midfielders (Pedri, F. de Jong) and goalkeepers (G. Donnarumma, Ederson), adding diversity to the mix.

# ### Bottom 10 Players by Highest Potential

# In[689]:


X_train_potential = X_train.copy()
X_train_potential['Overall'] = y_train

# Sort by Potential and select the top 10
xtrain_potential = X_train_potential.sort_values(['Potential'], ascending=True).head(10)

# Select relevant columns and apply a gradient style
xtrain_potential[['Name', 'Overall', 'Potential', 'Club', 'Nationality', 'Age', 'Position']].style.background_gradient(cmap='Reds')


# As expected, the list is dominated by **reserves and substitutes** playing in **less competitive leagues**.

# ### Distribution of Players Potential

# In[693]:


# Bar plot for Potential distribution
plt.figure(figsize=(12, 6))
sns.barplot(x=X_train['Potential'].value_counts().index, y=X_train['Potential'].value_counts().values, color='lightblue')

plt.title('Potential Distribution', fontsize=14)
plt.xlabel('Potential Rating')
plt.ylabel('Frequency')
plt.show()


# ### `Value`

# In[695]:


X_train.sort_values(['Value'],ascending=False)[:10]


# Just by looking at it, there seems to be some correlation between a player's potential and their market value, suggesting that higher potential often translates to a greater valuation in transfermarket.

# In[699]:


# Scatter plot for Value vs Overall (target)
plt.figure(figsize=(12, 6))
plt.scatter(X_train['Value'], y_train, alpha=0.6, color='#87CEEB')

plt.title('Value vs Overall Rating', fontsize=12)
plt.xlabel('Value', fontsize=12)
plt.ylabel('Overall Rating', fontsize=10)
plt.grid(axis='both', linestyle='--', alpha=1)
plt.show()


# Top players are worth so much more than the average player—it’s a huge gap in value!

# ### `Weak Foot`, `Skill Moves` and `International Reputation`

# In[703]:


# To find the range of values for Weak Foot, Skill Moves, and International Reputation
attributes = ['Weak Foot', 'Skill Moves', 'International Reputation']

for attr in attributes:
    print(f"{attr} - Min: {X_train[attr].min()}, Max: {X_train[attr].max()}")


# In[705]:


# Filter players with 5 stars for Weak Foot and Skill Moves
X_train.loc[
    (X_train['Weak Foot'] == X_train['Weak Foot'].max()) & 
    (X_train['Skill Moves'] == X_train['Skill Moves'].max())]


# Weak Foot and Skill Moves - R. Cherki, Cesinha, J. Corona

# In[708]:


# Filter players with 5 stars for Skill Moves and International Reputation
X_train.loc[
    (X_train['Skill Moves'] == X_train['Skill Moves'].max()) & 
    (X_train['International Reputation'] == X_train['International Reputation'].max())]


# Skill Moves and International Reputation - Cristiano Ronaldo, Z. Ibrahimović

# In[711]:


# Filter players with 5 stars for Weak Foot and International Reputation
X_train.loc[
    (X_train['Weak Foot'] == X_train['Weak Foot'].max()) & 
    (X_train['International Reputation'] == X_train['International Reputation'].max())]


# If no players have 5 stars for both **Weak Foot** and **International Reputation**, it’s impossible to have players with 5 stars in all three attributes. Surprisingly, there are very few players who excel with 5 stars in more than one attribute, highlighting how rare this level of versatility is in football.

# ##### Out of curiosity, let's check how top players look for Weak Foot, International Reputation, and Skill Moves:

# In[715]:


# Filter players with Overall >= 89
players_above_89 = X_train[y_train >= 89]

# Define attributes for radar chart
attributes = ['International Reputation', 'Weak Foot', 'Skill Moves']
data = players_above_89[['Name'] + attributes].set_index('Name')

# Number of variables and angles for radar chart
angles = np.linspace(0, 2 * np.pi, len(attributes), endpoint=False).tolist()
angles += angles[:1]

# Initialize the radar chart
plt.figure(figsize=(10, 10))
ax = plt.subplot(111, polar=True)

# Plot each player's data
for player, row in data.iterrows():
    values = row.tolist() + row.tolist()[:1]
    ax.plot(angles, values, linewidth=1.5, linestyle='solid', label=player)
    ax.fill(angles, values, alpha=0.2)

# Chart appearance
ax.set_yticks(range(1, 6))
ax.set_yticklabels(['1', '2', '3', '4', '5'], fontsize=10, color='grey')
ax.set_ylim(0, 5)
ax.set_xticks(angles[:-1])
ax.set_xticklabels(attributes, fontsize=12, color='darkblue')

# Adjust title and legend
plt.title('Overall Rating ≥ 89', size=16, color='navy', y=1.1, weight='bold')
plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10, frameon=False)
ax.grid(color='grey', linestyle='--', linewidth=0.5, alpha=0.7)

# Show the chart
plt.tight_layout()
plt.show()


# ### `Height` and `Weight`

# In[718]:


# Heaviest players
heaviest_player = X_train.sort_values(['Weight'], ascending=False).head(10)[['Name', 'Weight']]
print(heaviest_player)


# In[720]:


# Tallest players
tallest_player = X_train.sort_values(['Height'], ascending=False).head(10)[['Name', 'Height']]
print(tallest_player)


# In[722]:


# Scatter plot for Height vs Weight
plt.figure(figsize=(12, 6))
plt.scatter(X_train['Height'], X_train['Weight'], alpha=0.7, color='lightblue')
plt.title('Height vs Weight: Outliers', fontsize=12)
plt.xlabel('Height (cm)', fontsize=10)
plt.ylabel('Weight (kg)', fontsize=10)
plt.grid(axis='both', linestyle='--', alpha=0.5)
plt.show()


# ### `Nationality`

# In[725]:


X_train['Nationality'].unique()


# In[727]:


X_train['Nationality'].value_counts().head(20)


# In[729]:


X_train['Nationality'].value_counts().head(10).plot(kind='bar', figsize=(12, 6))
plt.title('Top 10 Nationalities')
plt.ylabel('Number of Players')
plt.show()


# England is the country that contributes the highest number of players.

# ### Top 5 Players by Overall Rating from Leading Football Nations

# In[733]:


X_train_with_overall = X_train.copy()
X_train_with_overall['Overall'] = y_train

# Top 5 players from each country in the top 5 list
top_countries = ['England', 'Germany', 'Spain', 'France', 'Argentina']

# Create a dictionary to store results
top_players_by_country = {}

for country in top_countries:
    # Filter players by country
    country_data = X_train_with_overall.loc[X_train_with_overall['Nationality'] == country]
    
    # Sort by Overall and select the top 5
    top_players_by_country[country] = country_data.sort_values(['Overall'], ascending=False).head(5)

# Display the results for each country
for country, players in top_players_by_country.items():
    print(f"\nTop 5 Players from {country}:")
    print(players[['Name', 'Overall', 'Potential', 'Club', 'Position']])


# ### `Age`

# In[736]:


X_train['Age'].unique()


# In[738]:


X_train['Age'].value_counts().head(20)


# In[740]:


# Age bins and labels
age_bins = [0, 20, 23, 28, 32, 100]
labels = ['<20', '20-23', '24-28', '29-32', '33+']

# Counting occurrences of each group
age_group = pd.cut(X_train['Age'], bins=age_bins, labels=labels).value_counts()

# Pie chart
plt.figure(figsize=(12, 6))
plt.pie(age_group, labels=age_group.index, autopct='%1.1f%%', 
        colors=sns.color_palette("Blues", len(labels)), startangle=90)

plt.title('Age Group Distribution')
plt.show()


# The dataset predominantly consists of young players.

# ##### Let’s check how **`Age`** affects player ratings:

# In[744]:


data_train = X_train.copy()
data_train['Overall'] = y_train  

# Filter for players under 41 years old
filtered_dataset = data_train[data_train['Age'] < 41]

# Group by Age and calculate the mean Overall Rating
age_rating = filtered_dataset.groupby('Age')['Overall'].mean().reset_index()

# Plot the relationship between Age and Overall Rating
plt.figure(figsize=(12, 6))
plt.plot(age_rating['Age'], age_rating['Overall'], marker='o', color='blue')
plt.title('Influence of Age on Overall Rating', fontsize=14)
plt.xlabel('Age', fontsize=12)
plt.ylabel('Average Overall Rating', fontsize=12)
plt.grid(alpha=0.5)
plt.show()


# Better ratings at later stages in career.

# ### `Body Type`

# In[748]:


X_train['Body Type'].unique()


# In[750]:


X_train['Body Type'].value_counts()


# In[752]:


# Filter players with Body Type 'Unique'
unique_body = X_train[X_train['Body Type'] == 'Unique'][['Name', 'Height', 'Weight']]

print(unique_body.head(25))


# In[754]:


# Plot distribution of players by body type
plt.figure(figsize=(10, 6))
sns.barplot(x=X_train['Body Type'].value_counts().index, y=X_train['Body Type'].value_counts().values, palette='Blues')

plt.title('Player Distribution by Body Type', fontsize=12)
plt.xlabel('Body Type')
plt.ylabel('Number of Players')
plt.xticks(rotation=45)
plt.show()


# The majority of players have a **Normal** or **Lean** body type. Players classified as having a **Unique** body type don’t fit neatly into the standard categories, making them distinct in their physical attributes.

# ### `Position`

# In[758]:


X_train['Position'].unique().tolist()


# In[760]:


# count positions
count_position = X_train['Position'].value_counts()

# Plot the positions
plt.figure(figsize=(12, 5))
sns.barplot(y=count_position.index, x=count_position.values, palette="Blues")
plt.title('Distribution of Players by Position', fontsize=14)
plt.show()


# Predominantly substitutes and reserves!

# ##### Let’s check how **`Position`** affects player ratings:

# In[764]:


# Visualize the relationship between Average Ranking and Position
X_train.assign(Overall=y_train).groupby('Position')['Overall'].mean().sort_values(ascending=False).plot(
    kind='bar', figsize=(12, 6), title='Average Overall Rating by Position'
)
plt.xlabel('Preferred Position')
plt.ylabel('Average Overall Rating')
plt.xticks(rotation=45)
plt.show()


# The bar plot reveals that **Center Forwards (CF)** and **Right Attacking Midfielders (RAM)** tend to have higher average ratings compared to other positions. As expected, substitutes and reserves rank lower on average. 
# 
# Interestingly, as we move from attacking positions to defensive roles, there’s a slight but consistent decrease in average ratings—except for **goalkeepers (GK)**, who surprisingly rank quite high.

# #### Let's see in more detail:

# In[768]:


# Matrix
position_matrix = X_train.assign(Overall=y_train).pivot_table(index='Position', values='Overall', aggfunc='mean').sort_values(by='Overall', ascending=False)

# Plot the matrix as a heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(position_matrix, annot=True, cmap='Blues', cbar=True, fmt='.2f')
plt.title('Average Overall Rating by Position', fontsize=12)
plt.ylabel('Position')
plt.show()


# ### `Preferred Foot`

# In[770]:


X_train['Preferred Foot'].value_counts()


# In[771]:


# Calculate percentages for each foot preference
foot_percentages = X_train['Preferred Foot'].value_counts(normalize=True) * 100

# And create a DataFrame
foot = pd.DataFrame({'Percentage': [foot_percentages.get(1, 0), foot_percentages.get(0, 0)]}, index=['Right Foot', 'Left Foot'])
foot.style.background_gradient(cmap='Blues')


# In[775]:


# Plot the results
plt.figure(figsize=(12, 6))
sns.barplot(x=foot_percentages.index, y=foot_percentages.values, palette='Blues')
plt.title('Preferred Foot Distribution (%)')
plt.ylabel('Percentage')
plt.show()


# As expected, there are significantly more right-footed players than left-footed ones.

# In[778]:


# Calculate the average Overall rating for each Preferred Foot
foot_avg = X_train.assign(Overall=y_train).groupby('Preferred Foot')['Overall'].mean()

# Plot the results
plt.figure(figsize=(12, 6))
sns.barplot(x=foot_avg.index, y=foot_avg.values, palette='Blues')
plt.title("Average Overall Rating by Preferred Foot", fontsize=12)
plt.xticks([0, 1], ['Left Foot', 'Right Foot'])
plt.ylabel("Average Overall Rating")
plt.show()


# The plot reveals a **very small advantage** for left-footed players in terms of average overall ratings. However, the difference is so small that it is almost unnoticeable. This suggests that **Preferred Foot** might not be a very significant feature in predicting player ratings.

# ## Pairwise variables analysis

# ### `Nationality` vs `Wage`

# In[783]:


# Filter top 10 nationalities and plot wage distribution
top_nationalities = X_train['Nationality'].value_counts().nlargest(10).index

plt.figure(figsize=(12, 6))
sns.boxplot(data=X_train[X_train['Nationality'].isin(top_nationalities)],
            x='Nationality', y='Wage', palette='Blues')

plt.title('Wage Distribution by Top 10 Nationalities')
plt.xticks(rotation=45)
plt.xlabel('Nationality')
plt.ylabel('Wage (€)')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


# The plot shows how wages vary across the top 10 nationalities in football. Players from Brazil, England, and Spain generally have the highest wages, likely reflecting their strong football cultures and talent pools. Nationalities like Japan and Uruguay show lower median wages, suggesting fewer players earning at the top end. The presence of outliers (dots) indicates a few star players earning significantly more than others in their nationality group.

# ### `Nacionality` vs `Position`

# In[787]:


top_nation = X_train[X_train['Nationality'].isin(top_nationalities)]

# Number of instances
pos_nation_counts = top_nation.groupby(['Nationality', 'Position']).size().reset_index(name='Count')

# Plot
plt.figure(figsize=(16, 10))
sns.barplot(data=pos_nation_counts, x='Nationality', y='Count', hue='Position', palette='viridis')

# Add titles and labels
plt.title('Distribution of Players by Nationality and Position', fontsize=16)
plt.xlabel('Nationality', fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.xticks(rotation=45, ha='right', fontsize=14)
plt.legend(title='Position', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()


# ### `Nacionality` vs `Age`

# In[789]:


# Create a bar plot for Age by Nationality

plt.figure(figsize=(10, 30))
sns.barplot(y=X_train['Nationality'], x=X_train['Age'], ci=None, palette="rainbow")

plt.xlabel('Average Age', fontsize=10)
plt.ylabel('Nationality', fontsize=10)
plt.show()


# Average age of players from Curacao and Monserratis is greater than 31, while that of Singapore and Malawi is less than 20!!!!!

# ### `Wage` vs `Age`

# In[792]:


plt.figure(figsize=(16, 10))

sns.violinplot(data=X_train, x='Age', y='Wage', palette='viridis')
plt.title('Wage Distribution by Age', fontsize=12)
plt.xlabel('Age', fontsize=10)
plt.ylabel('Wage (in €)', fontsize=10)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


# This violin plot shows that wages peak between ages 29 and 33, where players are most likely to earn very high amounts, with some exceeding €300,000. After 33, wages generally decrease, with fewer players earning exceptionally high salaries. While a few older players still earn over €100,000, most see their wages drop as they near the end of their careers.

# ### `Position` vs `Preferred Foot`

# In[796]:


# Grouping by Position and Preferred Foot
pos_foot = X_train.groupby("Position")["Preferred Foot"].value_counts().reset_index(name='Count')

# Plot
plt.figure(figsize=(14, 8))
sns.barplot(x='Position', y='Count', data=pos_foot, hue='Preferred Foot', palette='Blues')

plt.title('Preferred Foot Distribution by Position', fontsize=14)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# Surprisingly, left-back is the only position where left-footed players outnumber right-footed players—and by a significant margin. Interestingly, right-footed players dominate other left-sided positions.

# ### Histogram plots

# In[799]:


# Columns to plot
to_plot = ['Height', 'Weight', 'Age', 'Potential', 
    'Value', 'Wage', 'Release Clause']

fig, axes = plt.subplots(ncols=3, nrows=int(np.ceil(len(to_plot) / 3)), figsize=(20, 20))

# Loop through the columns
for col, ax in zip(to_plot, axes.flat):
    sns.histplot(X_train[col], ax=ax, kde=True).set(title=f"PDF of: {col}", xlabel="")

# Remove any extra empty subplots
for ax in axes.flat[len(to_plot):]:
    ax.set_visible(False)

plt.tight_layout()
plt.show()


# ### Boxplots

# In[801]:


# Columns to plot
to_plot = ['Height', 'Weight', 'Age', 'Potential', 'Value', 'Wage', 'Release Clause']

fig, axes = plt.subplots(ncols=3, nrows=int(np.ceil(len(to_plot) / 3)), figsize=(20, 20))

# Loop through the columns
for col, ax in zip(to_plot, axes.flat):
    sns.boxplot(data=X_train, y=col, ax=ax)
    ax.set_title(f"Boxplot of: {col}")

# Remove any extra empty subplots
for ax in axes.flat[len(to_plot):]:
    ax.set_visible(False)

plt.tight_layout()
plt.show()


# 1. **Height and Weight**:  
#    Most players are between 170-190 cm tall and weigh 60-90 kg. These are typical ranges for footballers.
# 
# 2. **Age**:  
#    Football is a young person’s game! Most players are in their early 20s, with fewer older players in the mix.
# 
# 3. **Potential**:  
#    The majority of players have a "good but not great" potential (70-80), with only a few standing out as exceptional.
# 
# 4. **Value, Wage, and Release Clause**:  
#    These are all highly skewed. Most players earn modestly and have low values, while a few stars are off the charts in terms of value, salary, and release clauses.

# In[804]:


# Columns to plot
to_plot = ['Height', 'Weight', 'Age', 'Potential', 'Value', 'Wage', 'Release Clause']

fig, axes = plt.subplots(ncols=3, nrows=int(np.ceil(len(to_plot) / 3)), figsize=(20, 20))

# Loop through the columns
for col, ax in zip(to_plot, axes.flat):
    sns.regplot(x=X_train[col], y=y_train, ax=ax, scatter_kws={'alpha': 0.5}, line_kws={'color': 'red'})
    ax.set_title(f"Relationship of {col} with Target")

# Remove any extra empty subplots
for ax in axes.flat[len(to_plot):]:
    ax.set_visible(False)

plt.tight_layout()
plt.show()


# 1. **Height**:  
#    Taller players don’t always have higher ratings—height doesn’t seem to matter much for overall quality.
# 
# 2. **Weight**:  
#    Heavier players show a slight edge, but weight isn’t a strong factor in determining ratings.
# 
# 3. **Age**:  
#    Older players tend to have better ratings, but some younger stars break the trend.
# 
# 4. **Potential**:  
#    The higher the potential, the higher the rating—this is a clear and strong relationship.
# 
# 5. **Value**:  
#    High-value players often have high ratings, but there’s a lot of variability at lower values.
# 
# 6. **Wage**:  
#    Big earners tend to have high ratings, but there are some overpaid players who don’t match up!
# 
# 7. **Release Clause**:  
#    Players with expensive release clauses usually have better ratings, but there are exceptions.

# ## Statistical Analysis

# ### Basic Statistics

# In[808]:


X_train.describe()


# ### Chi-square Test

# In[811]:


# List categorical variables
categorical_columns = X_train.select_dtypes(include=['category','object']).columns

# Chi-Square test
for column in categorical_columns:
    crosstab = pd.crosstab(X_train[column], y_train)
    chi2, p, dof, expected = chi2_contingency(crosstab)
    print(f"Chi-Square Test for {column}: p-value = {p}")


# 1. **Name**:  
# 🚫 No relationship with the overall rating (p = 0.617). Names are just identifiers!
# 
# 2. **Nationality**, **Club**, **Work Rate**, **Body Type**, **Position**:  
# ✅ Strong relationship (p ≈ 0.0). These factors significantly impact a player’s rating, reflecting skills, physical traits, and playing environment.
# 
# 3. **Loaned From**:  
# 🚫 No relationship (p = 1.0). Loan status doesn’t impact ratings.

# ### Pearson Correlation

# In[815]:


# List of numerical variables
numerical_columns = X_train.select_dtypes(include=['int', 'float']).columns

# Pearson correlation
for column in numerical_columns:
    correlation, p_value = pearsonr(X_train[column], y_train)
    print(f"Pearson Correlation between {column} and y_train: {correlation:.4f}, p-value: {p_value:.4g}")


# ### Correlation plot

# In[817]:


X_train_target = X_train.copy()
X_train_target['Overall'] = y_train

# List of numerical variables (including the target)
numerical_features = X_train_target.select_dtypes(include=['int64', 'float64'])

# Correlation matrix
correlation_matrix = numerical_features.corr()

# Correlations of numerical features with the 'Overall' target
correlation_with_target = correlation_matrix[['Overall']].drop('Overall')

# Plot the heatmap
plt.figure(figsize=(8, 10))
sns.heatmap(correlation_with_target, annot=True, cmap='coolwarm', cbar=True, fmt='.2f', linewidths=0.5)
plt.tight_layout()
plt.show()


# - **Most Predictive Features**:  
#   `Potential`, `Wage`, `Value`, and `Age` are strongly linked to `Overall Rating`—these are your top features for predictive modeling!
# 
# - **Least Predictive Features**:  
#   `Height` and `Preferred Foot` barely affect ratings, so their contribution is minimal.

# ## Final Decision

# We’ll focus on the top features for modeling:  
# - **Categorical**: `Club`, `Position`, `Work Rate`, `Body Type`, and `Nationality` all have strong links to `Overall`.  
# - **Numerical**: `Potential`, `Wage`, `Value`, and `Age` are the strongest predictors.  
# 
# On the other hand, **Preferred Foot** adds little value and will be removed. **Height** and **Weight** also have weak correlations and will be deprioritized, but kept for now.  
# 
# For **Loan Status**, although it has no relationship with `Overall` (`p = 1.0`), we’ll try converting it to a binary variable (0 for "Loaned" and 1 for "Permanent" or vice-versa) to see if it adds some value.

# In[821]:


X_train= X_train.drop(columns=['Preferred Foot'])
X_test= X_test.drop(columns=['Preferred Foot'])


# ##### Lets give it another look:

# In[823]:


X_train.head()


# In[824]:


X_test.head()


# In[825]:


print(X_train.shape, X_test.shape)


# In[826]:


print(y_train.shape, y_test.shape)


# ## Export

# In[828]:


X_train.to_csv("X_train.csv", index=False)
X_test.to_csv("X_test.csv", index=False)
y_train.to_csv("y_train.csv", index=False)
y_test.to_csv("y_test.csv", index=False)

print("Datasets successfully exported as CSV files.")

