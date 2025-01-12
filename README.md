# **FIFA 23 Player Rating Prediction**
How do you uncover the stars behind the numbers?

In football, player ratings aren’t just numbers—they’re a way to measure skill, potential, and reputation. But what if we could predict them? This project takes on the challenge of predicting FIFA 23 player ratings using real-world data like age, club, position, and contract details. From young talents full of potential to experienced stars, every player’s data tells a unique story.

---

## **Project Goal**
The goal of this project is to develop machine learning models to predict a player’s **Overall Rating** in FIFA 23 based on player characteristics and performance metrics. By understanding the factors that contribute to player ratings, we can:

- Help scouts find hidden gems.
- Enable gamers to build better teams.
- Provide fans with deeper insights into what makes a player great.

We will use the following machine learning models:

- **Baseline OLS Regression**: Simple, interpretable benchmark.
- **AdaBoost**: Combines weak learners to create a strong predictive model.
- **XGBoost**: A powerful, scalable gradient boosting method.
- **MLP (Multi-Layer Perceptron)**: A deep learning model to capture complex, non-linear relationships.

---

## **Dataset Overview**
This dataset contains 17,000+ instances with 28 features. The features can be grouped into the following categories:

### **Player Information**
- **ID**: Unique identifier for each player.
- **Name**: Player’s name.
- **Age**: Player’s age (numeric).
- **Nationality**: The country the player represents.
- **Height** and **Weight**: Physical attributes (numeric).
- **Body Type**: Player’s build (categorical).
- **Real Face**: Binary indicator of face representation in the game.

### **Player Ratings and Attributes**
- **Potential**: Predicted future performance (numeric).
- **Special**: Specialized skill rating.
- **Preferred Foot**: Dominant foot (categorical).
- **Weak Foot**: Ability to use the weaker foot (ordinal).
- **Skill Moves**: Proficiency with skill moves (ordinal).
- **Position**: Primary playing position (categorical).

### **Club and Contract Information**
- **Club**: Current club affiliation (categorical).
- **Contract Valid Until**: Year of contract expiration (numeric).
- **Release Clause**: Value of the release clause (numeric).

### **Financial Information**
- **Value**: Market value (numeric).
- **Wage**: Weekly wage (numeric).

### **Target Variable**
- **Overall**: The current overall skill rating, serving as the target variable for regression.

---

## **Challenges**
- **Complex Relationships**: Player roles and attributes interact differently for strikers, midfielders, and defenders.
- **Wide Variety**: Features like nationality and club have many unique values, increasing complexity.
- **Imbalanced Data**: Attributes like skill moves and reputation are skewed and require careful handling.

---

## **Why It Matters**
- **Scouting Talent**: Spotting rising stars and undervalued players.
- **Team Optimization**: Building better squads by identifying the ideal players for specific roles.
- **Gaming Insights**: Understanding the logic behind player ratings to enhance gameplay strategy.

---

## **Project Structure**
```plaintext
├── README.md
├── data
│   ├── input               # Raw data files
│   ├── processed           # Preprocessed data
│   └── models_output       # Model predictions and metrics
├── models                  # Scripts for model training and evaluation
├── notebooks               # Exploratory and modeling notebooks
│   ├── 01. data-preparation-and-eda.ipynb
│   ├── 02. feature-engineering.ipynb
│   ├── 03. baseline-ols.ipynb
│   ├── 04. adaboost-model.ipynb
│   ├── 05. xgboost-model.ipynb
│   ├── 06. mlp-model.ipynb
│   ├── 07. model-comparison-summary.ipynb
├── requirements.txt        # Conda environment dependencies
└── requirements_pypi.txt   # Additional pip dependencies

