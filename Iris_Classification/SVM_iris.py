from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import datasets, svm
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# load the dataset from sklearn libraray
dataset= datasets.load_iris(as_frame=True)

# finding the column names or feature names
column_name=dataset.feature_names
print(column_name)

# Show the first five and last five rows
features=dataset.data
print(features)

# Show the target values
labels=dataset.target
print(labels)

# show the target names
print(dataset.target_names)

# use the train_test_split function to divide the data into taraining and test data. 
J_train, J_test, k_train, k_test = train_test_split(
    features, labels, test_size=0.3, random_state=0)

print(J_test.shape, k_test.shape)

# Here we use the SVM classifier with Linear Kernal
SVM = svm.SVC(kernel='linear', C=1)
SVM.fit(J_train, k_train)

# printing the score
print("SVM with linear kernel results= ",SVM.score(J_test, k_test))
Prediction=SVM.predict(J_test)

# Cross Validation Applied on iris Dataset 
scores = cross_val_score(SVM, features, labels, cv=5)
print("Result using Cross Validation = ",scores)

# calculate the accuracy
ACC=accuracy_score(k_test, Prediction)
print("\n","Accuracy is = ",round(ACC*100,2),"%")

# calculate the confusion matrix
conf_metrics=confusion_matrix(k_test, Prediction)

# draw the confusion matrix
sns.heatmap(conf_metrics, annot=True, fmt='d', cbar=False)
plt.ylabel("Predicted", fontsize=16)
plt.xlabel("Actual", fontsize=16)
plt.title("Confusion Matrix for Digit Dataset",fontsize=18)
plt.savefig("confusion_matrix.png")
plt.show()
