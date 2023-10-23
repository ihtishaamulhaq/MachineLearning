#Import the necessary libraries

from sklearn.datasets import load_breast_cancer
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import GaussianNB
import seaborn as sns

import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.metrics import confusion_matrix

# Load the breast cancer dataset from Sklearn
data= load_breast_cancer(as_frame=True)

# The data matrix
dataset=data.data
print(dataset)

# Target names 
targetName=data.target_names
print(targetName)

# Column names / feature names
column=data.feature_names
print(column)

# Description about the dataset
print(data.DESCR)

# Show the first five records
print(data.frame.head())

# Target values/ labels
targets=data.target
print(targets)

# count plot for negative and positive cases
sns.countplot(x='target', data=data, color="salmon", palette = "Set2")
plt.xlabel('Diagnosis (0 = Malignant, 1 = Benign)')
plt.title('Target Distribution ')
plt.savefig('Distribution_Plot.png')
plt.show()

correlation=dataset.corr()
# Corelation matrix
plt.figure(figsize=(20, 20))
sns.heatmap(correlation, annot=True, fmt='.2f', cmap='coolwarm',
            cbar=False)
plt.title('Correlation Matrix')
plt.savefig('Correlation_matrix_heatmap.png')
plt.show()

# Split the Dataset into train and test dataset 
i_train, i_test, j_train, j_test = train_test_split(dataset, targets, test_size=0.2, random_state=56)

#  Gaussian Naive Bayes classifier instance
gnbc = GaussianNB()

# Define the parameter grid for var_smoothing
param_grid={'var_smoothing':np.logspace(-10,-1, 10)}

# GridSearchCV instance with the classifier and  grid parameter
grid_search_cv=GridSearchCV(gnbc, param_grid, cv=3, scoring='accuracy')
grid_search_cv.fit(i_train, j_train)

# Get the best parameters and accuracy on the training
best_parameter = grid_search_cv.best_params_
best_accuracy = grid_search_cv.best_score_
print("Best parameters found: ", best_parameter)
print("Best accuracy found: ", round(best_accuracy * 100, 2))
  
# Make predictions on the test set using the best parameters
best_gnbc = grid_search_cv.best_estimator_
predicted = best_gnbc.predict(i_test)

# Compute the confusion matrix
ConfusionMatrix = confusion_matrix(j_test,predicted)
sns.heatmap(ConfusionMatrix, annot=True, cmap='Blues', fmt='d', 
            cbar=False)
plt.ylabel('Prediction',fontsize=14)
plt.xlabel('Actual ',fontsize=14)
plt.title('Confusion Matrix',fontsize=16)
plt.savefig('Confusion_matrix.png')
plt.show()

# Calculate the performane measures like accuracy, precision, recall,and
# f1_score

accuracy = accuracy_score(j_test, predicted)
precision=precision_score(j_test, predicted)
recall=recall_score(j_test, predicted)
f1=f1_score(j_test, predicted)

# use the round() fuction 
scores=round(accuracy,2)
precision=round(precision,2)
recall=round(recall,2)
f1=round(f1,2)

# Print the performance scoes
print("Accuracy   =", scores, "%")
print("Precision  =", precision, "%")
print("Recall     =", recall, "%")
print("F1-Score   =", f1, "%")
