# -*- coding: utf-8 -*-
"""
Created on Sun Oct 15 10:54:12 2023

@author: Ihtishaam
"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from imblearn.under_sampling import ClusterCentroids 
from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import warnings


# Ignore FutureWarning from sklearn
warnings.simplefilter(action='ignore', category=FutureWarning)

# Ignore the UserWarning about the memory leak
warnings.filterwarnings("ignore", category=UserWarning, 
                        module="sklearn.cluster._kmeans")

#  Read CSV file 
dataset= pd.read_csv("E:\PythonCode\Credit_Card_Fraud\creditcard.csv")

#  Dataset information
dataset.info()

# Find the null values 
null_count=dataset.isnull().sum()
print(null_count)

# finding the number of sample in each class
number_of_samples=dataset['Class'].value_counts()
print(number_of_samples)

# indepdent and response variable
x_dataset=dataset.drop('Class', axis=1)
print(x_dataset.columns)
y_dataset=dataset['Class']
print(y_dataset)

# show the count plot 
plt.figure(figsize=(8,8))
sns.countplot(data=dataset,x='Class', palette='YlOrRd')
plt.title('Number of Cases in Each Class')
plt.savefig('Count_plot_casees.png')
plt.show()

# our dataset is imbalance. balance the dataset first
under_sampled_model=ClusterCentroids(sampling_strategy='majority', random_state=23)
under_sample_x, under_sample_y=under_sampled_model.fit_resample(x_dataset, y_dataset)
under_sample_dataset=pd.concat([pd.DataFrame(under_sample_x),pd.DataFrame(under_sample_y)], axis=1)

#  Show the number of samples after balanceing the dataset
classes=under_sample_dataset['Class']
plt.figure(figsize=(8,8))
sns.countplot(data=under_sample_dataset,x='Class', palette='OrRd')
plt.title('Count Plot aftr Balance the Dataset')
plt.savefig('balance_countplot.png')
plt.show()

# Split the indepdent and response variable after balance the Dataset
indepdent_variable=under_sample_dataset.drop('Class', axis=1)
responce_variable=under_sample_dataset['Class']

# Split the dataset into training and test Dataset
x_train, x_test, y_train, y_test=train_test_split(indepdent_variable,
                                                  responce_variable,
                                                  test_size=0.2,
                                                  random_state=23)

# Decision Tree model 
Decision_tree_clf=DecisionTreeClassifier(criterion='entropy',
                                         max_depth=5, 
                                         min_samples_leaf=10,
                                         min_samples_split=7,
                                         class_weight='balanced'
                                         )
# train and test the model
Decision_tree_clf.fit(x_train, y_train)
predict=Decision_tree_clf.predict(x_test)

# Confusion matrix
con_matrix=confusion_matrix(y_test, predict)
sns.heatmap(con_matrix, annot=True, cbar=False, fmt='d', cmap='Blues')
plt.xlabel('Actual labels', fontsize=14)
plt.ylabel('Predicted label', fontsize=14)
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')
plt.show()

# calculate and print the accuracy
Accuracy1=accuracy_score(y_test, predict)
Round_accuracy=round((Accuracy1*100), 2)
print("Accuracy = ",Round_accuracy,"%")

# classification report
classfiy_report=classification_report(y_test, predict)
print(classfiy_report)












