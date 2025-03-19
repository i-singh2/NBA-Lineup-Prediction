# Machine Learning: NBA Lineup Prediction Model

## SOFE 4620U - Machine Learning and Data Mining

## Group 20:

Inder Singh - 100816726

Justin Fisher - 100776303

Rohan Radadiya - 100704614

## Overview

This NBA Lineup Prediction project is meant to predict the fifth home player, ` home_4 ` in NBA matchups using historical data from 2007 to 2015.
Built using XGBoost, the model is trained on NBA data from 2007-2015 to analyze lineup effectiveness and assist in strategic decision-making.

## Key Features

- Predicts the optimal fifth player in an NBA lineup.
- Uses XGBoost for high-performance multi-class classification.
- Trained on historical NBA matchups (2007-2015) for accurate insights.
- Implements label encoding for team/player categorization.
- Optimized with GPU acceleration (CUDA) for fast training.
- Provides validation accuracy to assess model performance.

## Dataset

The dataset contains historical NBA lineup data, including allowed features such as:

- `game` - Unique game identifier
- `season` - NBA season year
- `home_team` - Home team name
- `away_team` - Away team name
- `starting_min` - Minutes played by the lineup
- `home_0 to home_3` - First four players in the home team lineup
- `away_0 to away_4` - First five players in the away team lineup
- `outcome` - Determines whether the game resulted in a win or loss for the home team (1 meaning win, -1 meaning loss)

## Setup and Running Instructions

### Prerequisites
- **Python** 3.8 or newer
- **Jupyter Notebook** or **JupyterLab**

### Installing Dependencies

- `pip install pandas numpy xgboost scikit-learn`
- `pip install openpyxl`

### Running the Project
1. Clone this repository or download the provided notebook and datasets.
2. Ensure your directory structure matches the layout provided above.
3. Open the notebook using Jupyter Notebook or JupyterLab:
   ```bash
   jupyter notebook nba_lineup_prediction.ipynb
4. Execute notebook cells sequentially from top to bottom to reproduce results.

## Workflow

### 1. Import libraries
- Load essential Python libraries including pandas, numpy, xgboost, and sklearn.
- These libraries handle data processing, machine learning training, and evaluation.

### 2. Extract allowed features from Matchup-Metadata:
- Filter the dataset to include only the features permitted for the project.
- Ensures compliance with dataset restrictions while maximizing predictive power.

### 3. Feature selection
- Identify the allowed features for predicting the optimal fifth player in a lineup.

### 4. Load and merge NBA Matchup Data
- Load and merge data of historical NBA matchups (2007-2015) and filter the allowed features.

### 5. Handle Missing Values
- Identify and remove missing values to prevent model biases.

### 6. Label Encode Columns
- Convert categorical data (e.g., team names, player names) into numerical format.
- Uses LabelEncoder to maintain consistency in machine learning models.

### 7. Remove Rare Target Classes for Better Model Learning
- Eliminate target classes (players) with very few occurrences to improve model generalization.
- Ensures that predictions are reliable and not skewed by limited data.

### 8. Train-Test Split
- Split data into training (80%) and testing (20%) datasets.
- Helps evaluate the model's ability to generalize to unseen data.

### 9. Move rows where outcome == -1 from test to train
- Some rows were placed in the test set with a -1 outcome (indicating an unknown player).
- These rows are moved back to the training set to ensure proper learning, and compliance with assignment instructions (test set can only contain outcome = 1)

### 10. Handling Missing Classes in Training Set
- This prevents the model from failing on completely unseen labels.

### 11. Train XGBoost Model
- XGBoost (Extreme Gradient Boosting) is trained using optimized hyperparameters.
- The model is trained using a multi-class classification objective (multi:softprob).

### 12. Model Testing Predictions
- The trained model is used to predict the fifth player in an NBA lineup for the test set.
- Predictions are compared to actual historical data.

### 13. Model Evaluation
- Evaluate model accuracy using multiclass log loss (mlogloss) and classification accuracy.
- Analyzes performance across training and validation datasets.

### 14. Final Results
- Displays the model’s final accuracy and loss metrics.
- Insights into how well the model performs and potential improvements.
- Initial validation: 79% accuracy
- Final validation using the provided test set: 0.003% accuracy
