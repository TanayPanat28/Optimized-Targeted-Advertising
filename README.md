# Optimized-Targeted-Advertising

## Description
This project analyzes and models marketing data to understand customer behavior and improve decision-making. It includes data cleaning, exploratory data analysis (EDA), dimensionality reduction using PCA and Factor Analysis, and model development using various machine learning algorithms. The models aim to predict customer responses and derive actionable insights.

## Features
1) Data Cleaning: Handling missing values and encoding categorical variables.
2) Exploratory Data Analysis: Visualizing data distributions, counts, and relationships.
3) Dimensionality Reduction:

    Principal Component Analysis (PCA) to reduce features while retaining 95% variance or selecting the first five components.
    Factor Analysis to extract underlying latent factors.

4) Machine Learning Models: Training and evaluating Logistic Regression, Random Forest, Gradient Boosting, and Neural Networks.
5) Visualization: Custom bar charts for categorical data and PCA scree plots.

## Requirements 
1) Python 3.7+
2) Libraries:
    pandas
    numpy
    matplotlib
    seaborn
    scikit-learn
    factor-analyzer

## Model Evaluation and Results

The dataset used in this project is slightly skewed, requiring a focus on metrics that better account for class imbalances. Instead of relying solely on accuracy, we prioritize the F1 Score as it balances precision and recall.

Best Model: 
Logistic Regression was identified as the best-performing model with the highest F1 Score of 88.08%.
