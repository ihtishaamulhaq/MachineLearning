#Import the necessary libraries

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import seaborn as sns

import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.metrics import confusion_matrix

 
# Load the breast cancer dataset from Sklearn
A, b = load_breast_cancer(return_X_y=True)

# Split the Dataset into train and test dataset 
i_train, i_test, j_train, j_test = train_test_split(A, b, test_size=0.3, random_state=56)

# Train the model
gnbc = GaussianNB()  
gnbc.fit(i_train, j_train)
  
# Predict using the test data
z_pred = gnbc.predict(i_test)

# Compute the confusion matrix

ConfusionMatrix = confusion_matrix(j_test,z_pred)
sns.heatmap(ConfusionMatrix, annot=True)
plt.ylabel('Prediction Class',fontsize=14)
plt.xlabel('Actual Class',fontsize=14)
plt.title('Confusion Matrix',fontsize=16)
plt.show()
print("\n")

# Calculate the performane measures like accuracy, precision, recall,and f1_score

accuracy = accuracy_score(j_test, z_pred)
precision=precision_score(j_test, z_pred)
recall=recall_score(j_test, z_pred)
f1=f1_score(j_test, z_pred)

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
