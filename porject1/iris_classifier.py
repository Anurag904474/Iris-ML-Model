# -*- coding: utf-8 -*-
"""
Created on Wed May 21 21:07:42 2025

@author: Anurag
"""

# iris_classifier.py
# Developed by: Anurag Vishwakarma

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Load the Iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
target_names = iris.target_names

# Convert to DataFrame for easy handling
df = pd.DataFrame(X, columns=feature_names)
df['target'] = y

# Print dataset info
print("Sample Data:\n", df.head())

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the KNN model
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Evaluate performance
print("\nModel Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix Heatmap
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, cmap="Blues", xticklabels=target_names, yticklabels=target_names)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("iris_output.png")  # Save the confusion matrix plot
plt.show()

# Scatter plot for Sepal Length vs Sepal Width
colors = ['red', 'blue', 'green']
plt.figure(figsize=(6, 4))
for i, color in zip([0, 1, 2], colors):
    plt.scatter(df[df['target'] == i]['sepal length (cm)'],
                df[df['target'] == i]['sepal width (cm)'],
                label=target_names[i],
                color=color)
plt.title("Sepal Length vs Width")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Sepal Width (cm)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("iris_scatter.png")  # Save scatter plot
plt.show()
