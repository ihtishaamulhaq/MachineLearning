# In this program we use the random forest classifier for digit classification.

# -*- coding: utf-8 -*-
"""
Created on Wed May 24 18:12:06 2023

@author: Ihtishaam
"""

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import seaborn as sns
import matplotlib.pyplot as plt

# load the dataset from Sklearn (datasets library)
A, b=datasets.load_digits(return_X_y=True)
print(A.shape)
print(b.shape)



# Split the dataset into training and testing 
I_train, I_test, J_train, J_test=train_test_split(A, b, test_size=0.3) 

# train the model
R_F_Classifier=RandomForestClassifier(random_state=62)
R_F_Classifier.fit(I_train, J_train)

# model testing
Prediction=R_F_Classifier.predict(I_test)

# calculate the confusion matrix
conf_mat=metrics.confusion_matrix(J_test, Prediction)

# Display the confusion matrix
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='plasma', cbar=False)
plt.xlabel("actual")
plt.ylabel("prediction")
plt.title("Confusion Matrix on Digit Dataset",fontsize=16)
# plt.show()
plt.savefig("confusion Matrix.png")

# compute the metrics like Accuracy, Precision, Recall, F1-score
score=metrics.accuracy_score(J_test, Prediction)
print("Accuracy  :",round(score*100,2), "%")
score1=metrics.recall_score(J_test, Prediction, average='weighted')
print("Recall    :",round(score1*100,2), "%")
score2=metrics.precision_score(J_test, Prediction, average='macro')
print("Precision :",round(score2*100,2), "%")
score3=metrics.f1_score(J_test, Prediction, average='macro')
print("F1-Score  :",round(score3*100,2), "%")

