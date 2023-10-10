# -*- coding: utf-8 -*-
"""
Created on Sat Oct  7 17:27:44 2023

@author: Ihtishaam
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import ClusterCentroids

from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.svm import SVC

from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import classification_report

import warnings


# Ignore FutureWarning from sklearn
warnings.simplefilter(action='ignore', category=FutureWarning)

# Ignore the UserWarning about the memory leak
warnings.filterwarnings("ignore", category=UserWarning, 
                        module="sklearn.cluster._kmeans")


# read the csv file
diabetes_data=pd.read_csv('E:\PythonCode\Pima_Indians_Diabetes\diabetes.csv')

# Names of the columns
column_name=diabetes_data.columns
print(column_name)

# find the shape of the data
shape_=diabetes_data.shape
print(shape_)

# Show the random samples from the dataset
random_sample=diabetes_data.sample(10, random_state=101)
print(random_sample)

# Count the negative and positive cases
negative_cases, positive_cases=diabetes_data['Outcome'].value_counts()
print("Negative cases =",negative_cases, "\n",
      "Positive Cases =", positive_cases)

# countplot  for the Outcome/ target variable
sns.countplot(x='Outcome', data=diabetes_data, palette='YlGnBu')
plt.title('Count Plot on Negative and Positive Cases')
plt.savefig('countplot_negative.png')
plt.show()

# countplot to check age wise cases
plt.figure(figsize=(16,10))
sns.countplot(x='Age', hue='Outcome', data=diabetes_data, palette='YlOrRd')
plt.title('Age Wise Cases')
plt.savefig('Age_wise_cases.png')
plt.show()

# Solve the class imbalance problem Using SMOTE 
balance_model=SMOTE(sampling_strategy='minority', random_state=42)
oversampled_X, oversampled_Y = balance_model.fit_resample(diabetes_data.drop('Outcome',
                                axis=1), diabetes_data['Outcome'])
oversampled_diabetes = pd.concat([pd.DataFrame(oversampled_X), 
                                  pd.DataFrame(oversampled_Y)],axis=1)

# countplot  for the Outcome/ target variable
sns.countplot(x='Outcome', data=oversampled_diabetes, palette='OrRd')
plt.title('Count Plot after Oversampling')
plt.savefig('countplot_oversampled.png')
plt.show()

# 
Features=oversampled_diabetes.drop(['Outcome'],axis=1)
Labels=oversampled_diabetes['Outcome']

# Divide the dataset into training and test data
X_train, X_test, Y_train, Y_test = train_test_split(Features,Labels,test_size=0.2,
                                                  random_state=23)
# intilize and train the model
AdaBoo_clf=AdaBoostClassifier(n_estimators=70, random_state=23,
                              learning_rate=0.7, algorithm="SAMME.R",)
AdaBoo_clf.fit(X_train, Y_train)

# Check on test Dataset
predict=AdaBoo_clf.predict(X_test)

# Compute the confusion matrix
Confusion_m=confusion_matrix(Y_test, predict)
sns.heatmap(Confusion_m,cmap='Blues', annot=True, cbar=False, fmt='d')
plt.xlabel("Actual", fontsize=14)
plt.ylabel("Predicted", fontsize=14)
plt.title("Confusion Matrix")
plt.savefig("confusion_plot_oversample.png")
plt.show()

#  find the Accuracy
Accuracy=accuracy_score(Y_test, predict)
Accuracy_=round((Accuracy*100),2)
print(Accuracy_)

# print the classification report
class_repot=classification_report(Y_test, predict)
print(class_repot)

# Solve the class imbalance problem Using SMOTE 
balance_model1=ClusterCentroids(sampling_strategy='majority', random_state=42)
undersampled_X, undersampled_Y = balance_model1.fit_resample(diabetes_data.drop('Outcome',
                                axis=1), diabetes_data['Outcome'])
undersampled_diabetes = pd.concat([pd.DataFrame(undersampled_X), 
                                  pd.DataFrame(undersampled_Y)],axis=1)

# countplot  for the Outcome/ target variable
sns.countplot(x='Outcome', data=undersampled_diabetes, palette='BuPu')
plt.title('Count Plot after Undersampling')
plt.savefig('countplot_underSampled.png')
plt.show()

# 
Features2=undersampled_diabetes.drop(['Outcome'],axis=1)
Labels2=undersampled_diabetes['Outcome']

# Divide the dataset into training and test data
i_train, i_test, j_train, j_test = train_test_split(Features2,Labels2,test_size=0.2,
                                                  random_state=23)
# intilize and train the model
Bagging_clf=BaggingClassifier(estimator=SVC(), n_estimators=50,
                              random_state=41)
                              
Bagging_clf.fit(i_train, j_train)

# Check on test Dataset
prediction=Bagging_clf.predict(i_test)

# Compute the confusion matrix
Confusion_Bag=confusion_matrix(j_test, prediction)
sns.heatmap(Confusion_Bag, cmap='OrRd', annot=True, cbar=False, fmt='d')
plt.xlabel("Actual Label", fontsize=14)
plt.ylabel("Predicted", fontsize=14)
plt.title("Confusion Matrix on Undersample Dataset")
plt.savefig("confusion_plot_undersample.png")
plt.show()

#  find the Accuracy
Accuracy1=accuracy_score(j_test, prediction)
Accuracy_1=round((Accuracy1*100),2)
print(Accuracy_1)

# print the classification report
class_repot_1=classification_report(j_test, prediction)
print(class_repot_1)