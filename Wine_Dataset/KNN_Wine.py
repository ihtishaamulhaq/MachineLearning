# In this file, we use the wine dataset from the Sklearn Dataset library

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import matplotlib.pyplot as plt
from termcolor import colored
from sklearn.metrics import confusion_matrix
import seaborn as sns

# load the dataset
dataset= datasets.load_wine()

# Print the descriptip about the dataset
print(wine.DESCR)

# print the Wine data
print(dataset.data,"\n")
A=dataset.data

#  type of the dataset
print(type(dataset),"\n")

# find the name of features
print(dataset.feature_names,"\n") 

# number of target classes
print("Total number of classes",dataset.target_names,"\n")

# target values 
print(dataset.target,"\n")
b=dataset.target
# total observations and number of features 
print(dataset.data.shape,"\n")

# dataset is divided into train and test data (ratio 70:30)
i_train, i_test, j_train, j_test = train_test_split(A, b, 
                                    test_size=0.3, random_state=43)

# printing the shape of training and test data
print(i_train.shape,"\n")
print(i_test.shape,"\n")

# Knn classifier 
knn = KNeighborsClassifier(n_neighbors=5, weights='distance', 
                           algorithm='auto',leaf_size=40, p=2)

# train the model
knn.fit(i_train,j_train)

# test the model 
y_predict = knn.predict(i_test)

# draw scatter plot 
print (colored ("** Scatter Plot of Wine Dataset **"))
plt.figure(figsize = (8,5))
plt.scatter(i_test[:,0], i_test[:,1], c=y_predict, s=100, edgecolors='black')
plt.title("Predicted values with k=5", fontsize=20)
plt.show()

# Compute the confusion matrix
ConfusionMatrix = confusion_matrix(j_test,y_predict)
  
# Plot the confusion matrix.
sns.heatmap(ConfusionMatrix, annot=True)
plt.ylabel('Predict Class',fontsize=12)
plt.xlabel('Actual Class',fontsize=12)
plt.title('Confusion Matrix',fontsize=18)
plt.show()
print("\n")

# Compute the Evaluation Score
score = metrics.accuracy_score(j_test, y_predict) 
score1 = metrics.recall_score(j_test, y_predict, average='micro')
score2 = metrics.precision_score(j_test, y_predict, average='micro')

# round the value upto two decimal places
scores = round(score*100,2)
scores1 = round(score1*100,2)
scores2 = round(score2*100,2)

# Print the results
print("Accuracy =",scores, "%")
print("Recall   =",scores1, "%")
print("Precision=",scores2, "%")
