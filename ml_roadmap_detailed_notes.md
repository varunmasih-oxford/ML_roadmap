# Machine Learning Detailed Notes

---

# Phase 1: Core Machine Learning Concepts

## 1. What is Machine Learning?
Machine Learning (ML) is a field of Artificial Intelligence that allows computers to learn patterns from data and make predictions or decisions without being explicitly programmed.

### Traditional Programming vs Machine Learning
| Traditional Programming | Machine Learning |
|------------------------|------------------|
| Rules + Data → Output | Data + Output → Model |

---

## 2. Types of Machine Learning

### 2.1 Supervised Learning
Supervised learning uses labeled data, meaning both input features and output labels are available.

Examples:
- House price prediction
- Student marks prediction
- Spam email detection

Common Algorithms:
- Linear Regression
- Logistic Regression
- K-Nearest Neighbors (KNN)
- Decision Tree
- Random Forest
- Support Vector Machine (SVM)

---

### 2.2 Unsupervised Learning
Unsupervised learning works with unlabeled data. The model identifies hidden patterns and structures in the data.

Examples:
- Customer segmentation
- Market basket analysis
- Grouping similar documents

Common Algorithms:
- K-Means Clustering
- Hierarchical Clustering
- DBSCAN
- Principal Component Analysis (PCA)

---

### 2.3 Reinforcement Learning (Introduction)
In reinforcement learning, an agent learns by interacting with an environment and receiving rewards or penalties.

Applications:
- Game playing
- Robotics
- Recommendation systems

---

## 3. Train-Test Split
The dataset is divided into two parts:
- Training data: Used to train the model
- Testing data: Used to evaluate model performance

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

---

## 4. Overfitting and Underfitting

Overfitting:
- Model performs very well on training data
- Performs poorly on unseen data
- Learns noise instead of pattern

Underfitting:
- Model is too simple
- Cannot capture underlying patterns
- Poor performance on both training and testing data

---

## 5. Bias and Variance

Bias:
- Error due to oversimplified assumptions
- Leads to underfitting

Variance:
- Error due to model sensitivity to training data
- Leads to overfitting

Bias-Variance tradeoff aims to balance both.

---

## 6. Model Evaluation Basics

Common evaluation concepts:
- Accuracy
- Error rate
- Generalization

---

# Supervised Learning Algorithms

## 7. Regression Algorithms

### 7.1 Linear Regression
Used to predict continuous numerical values.

Equation:
```
y = mx + c
```

Assumptions:
- Linear relationship
- No multicollinearity
- Homoscedasticity

Use cases:
- House price prediction
- Salary estimation

---

### 7.2 Multiple Linear Regression
Uses multiple independent variables to predict a single dependent variable.

---

### 7.3 Polynomial Regression
Used when the relationship between input and output is non-linear.

---

## 8. Classification Algorithms

### 8.1 Logistic Regression
Used for binary classification problems.

Examples:
- Pass/Fail prediction
- Spam/Not Spam

---

### 8.2 K-Nearest Neighbors (KNN)
Classifies data based on similarity with nearest neighbors.

Key parameter:
- K value

---

### 8.3 Decision Tree
Tree-based model that splits data using conditions.

Advantages:
- Easy to understand
- Works with non-linear data

Disadvantages:
- Can overfit

---

### 8.4 Random Forest
An ensemble of multiple decision trees.

Benefits:
- Reduces overfitting
- Higher accuracy

---

### 8.5 Naive Bayes
Based on Bayes theorem and probability.

Works well for:
- Text classification
- Spam detection

---

### 8.6 Support Vector Machine (SVM)
Creates an optimal decision boundary using hyperplanes.

---

## 9. Model Evaluation Metrics

Classification Metrics:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

Regression Metrics:
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R-squared

---

# Phase 2: Unsupervised Learning

## 10. What is Unsupervised Learning?
Unsupervised learning works on unlabeled data and finds hidden structures without predefined output.

---

## 11. Clustering

### 11.1 K-Means Clustering
Groups data into K clusters based on similarity.

Steps:
1. Choose number of clusters (K)
2. Assign data points to nearest centroid
3. Update centroids
4. Repeat until convergence

---

### 11.2 Hierarchical Clustering
Creates a tree-like structure of clusters.

Types:
- Agglomerative
- Divisive

---

### 11.3 DBSCAN
Density-based clustering algorithm.

Advantages:
- Handles noise
- Finds arbitrary shaped clusters

---

## 12. Dimensionality Reduction

### 12.1 Principal Component Analysis (PCA)
Reduces number of features while retaining maximum variance.

Benefits:
- Improves performance
- Reduces overfitting
- Visualization of high-dimensional data

---

## 13. Applications of Unsupervised Learning

- Customer segmentation
- Anomaly detection
- Recommendation systems
- Data compression

---
