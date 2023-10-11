# -*- coding: utf-8 -*-
"""
Created on Sun Sep 24 12:10:18 2023

@author: Ihtishaam
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from sklearn.metrics import precision_score, recall_score

import seaborn as sns
import matplotlib.pyplot as plt

# import the Customer Churn CSV file 
customer_data=pd.read_csv('E:\PythonCode\Customer_Churn_Prediction\Customer_Churn.csv')

# show the first ten rows 
print(customer_data.head(10))

# print the data information and shape
print(customer_data.info()) 
print(customer_data.shape)

# Columns name 
print(customer_data.columns)

# Using the info() function, we saw the data information like column  
# names, Non-null counts their data types, and memory usage
# here we realize that  RowNumber, CustomerId, and Surname
# are not useful, that's why we remove these columns.
dropped_columns=['RowNumber', 'CustomerId', 'Surname']
customer_data.drop(dropped_columns, axis=1, inplace=True)
print(customer_data.head()) # Now we have 11 columns

# Check the Null and Duplicate Values
print(customer_data.isnull().sum())
print(customer_data.duplicated().sum())

# Some columns have categorical data, show the unique values in the 
# following columns: Geography and Gender
print(customer_data['Geography'].unique())
print(customer_data['Gender'].unique())


# Now we use the  Data encoding technique to replace the 
# categorical data into numerical data , we use LabelEncoder from 
# Sklearn module 
"""
Label Encoding:
Label encoding is a technique used in machine learning to convert 
categorical data into numerical data by assigning each unique category
a numerical label.The labels are typically assigned in a sequential 
manner, with the first category receiving a label of 0, the second 
category receiving a label of 1, and so on.
"""
data_encoder=LabelEncoder()
customer_data['Geography']=data_encoder.fit_transform(customer_data['Geography'])
customer_data['Gender']=data_encoder.fit_transform(customer_data['Gender'])
print(customer_data.head())

# Split the indepedent and dependent variable
A=customer_data.drop(['Exited'], axis=1) # Input 
b=customer_data['Exited'] # Target 
print(A.shape)
print(b.shape)


# Split the data into training and test dataset
j_train, j_test, k_train, k_test=train_test_split(A, b, test_size=0.2, random_state=23)


# Gender exitence count plot
sns.countplot(x='Gender', hue='Exited', data=customer_data, palette='mako')
plt.savefig("Gender_Exitene.png")
plt.show()

# count the gender location (Geography)
sns.countplot(x='Geography', hue='Exited', data=customer_data, palette='mako')
plt.savefig("customer_location_count_plot.png")
plt.show()

#  Model training and testing (Decision Tree)
D_tree=DecisionTreeClassifier(random_state=2, min_samples_leaf=4, 
                              max_depth=9,min_samples_split=5)
D_tree.fit(j_train, k_train)
prediction=D_tree.predict(j_test)

# Draw and Display the Confusion matrix
cm=confusion_matrix(k_test, prediction)
sns.heatmap(cm, annot=True, cmap="Blues", fmt='d', cbar=False)
plt.title('CONFUSION MATRIX Using Decision Tree', fontsize=16)
plt.xlabel('Actual', fontsize=14)
plt.ylabel('Predicted', fontsize=14)
plt.savefig("Confusion_Matrix_for_DecisionTree.png")
plt.show()

# Compute the Accuracy score
Accuracy=accuracy_score(k_test, prediction)
Accuracy_score=round(Accuracy*100,2)
print("Accuracy = ", Accuracy_score)

# Compute the precision score
precision=precision_score(k_test, prediction)
precision_score=round(precision*100,2)
print("Precision = ", precision_score)

# Compute the recall score
recall=recall_score(k_test, prediction)
Recall_score=round(recall*100,2)
print("Recall = ", Recall_score)

# Compute the F1 score
F1=f1_score(k_test, prediction)
F1_score=round(F1*100,2)
print("F1_score = ", F1_score)

# count plot the number of exited or not exited customers
print(customer_data['Exited'].value_counts())
sns.countplot(customer_data, x='Exited')
plt.savefig("Count_plot.png")
plt.show()

print(customer_data['Exited'].value_counts())
sns.countplot(customer_data, x='Exited')
plt.savefig("Count_plot.png")
plt.show()

df_exited=customer_data[customer_data['Exited']==1]
df_not_exited=customer_data[customer_data['Exited']==0]
print(df_exited)
print(df_not_exited)

no_exited=df_not_exited.sample(n=2037)
print(no_exited.shape)

df_new=pd.concat([df_exited, no_exited],axis=0)
print(df_new.shape)

data=df_new.drop(['Exited'], axis=1)
print(data.shape)
label=df_new['Exited']
print(label.shape)

x_train, x_test, y_train, y_test=train_test_split(data, label, test_size=0.3, random_state=23)
print(x_train.shape)

#  Model training and testing (KNeighborsClassifier)
KNN_Classifier=KNeighborsClassifier(n_neighbors=14, weights='distance',
                                    algorithm='auto',leaf_size=20, p=2)
KNN_Classifier.fit(x_train,y_train)
Y_predict=KNN_Classifier.predict(x_test)


# compute and display the confusion matrix
cm_k=confusion_matrix(y_test,Y_predict)
sns.heatmap(cm_k, annot=True, cmap='plasma', fmt='d', cbar=False)
plt.title("Confusion Matrix using K Nearest Neighbors", fontsize=14)
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.savefig("Confusion_Matrix_for_KNN.png")
plt.show()

# Compute the Accuracy score
classification_report_k=classification_report(y_test, Y_predict)
print("********** Classification Reopt ********** = ","\n", classification_report_k,)

