# R_Charles_Book_Club---Predictive-Modeling

Charles Book Club Customer Purchase Prediction

This project focuses on developing classification models to predict customer purchasing behavior for Charles Book Club, a hypothetical company. The goal is to determine whether a customer will purchase a new book, "The Art History of Florence," based on historical purchasing data.

The dataset consists of 4000 customers with various features such as previous purchases across different book categories, recency, frequency, and monetary value of transactions. After data cleaning, exploratory data analysis, and dimensionality reduction (via correlation analysis and PCA), multiple models were explored, including logistic regression, k-Nearest Neighbors (k-NN), and neural networks.

Key steps:

# Data Preprocessing: 
Normalized data, handled class imbalance, and split into training and validation sets.
# Modeling: 
Logistic regression provided high sensitivity but poor specificity, whereas k-NN achieved a better balance between sensitivity and specificity, making it the most effective model.
# Outcome: 
The k-NN model was deployed to optimize targeted direct mail campaigns, improving marketing efficiency by focusing on the most likely book purchasers.

This project highlights the application of classification models in targeted marketing and customer segmentation strategies.
