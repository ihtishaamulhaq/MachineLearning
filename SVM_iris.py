from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import datasets, svm

# load the dataset from sklearn libraray
J, k = datasets.load_iris(return_X_y=True)

# finding the shape of dataset
J.shape, k.shape

# use the train_test_split function to divide the data into taraining and test data. 
J_train, J_test, k_train, k_test = train_test_split(
    J, k, test_size=0.2, random_state=0)

print(J_test.shape, k_test.shape)

# Here we use the SVM classifier with Linear Kernal
SVM = svm.SVC(kernel='linear', C=1)
SVM.fit(J_train, k_train)

# printing the score
print("SVM with linear kernel results= ",SVM.score(J_test, k_test))
prediction=SVM.predict(J_test)

# Cross Validation Applied on iris Dataset 
scores = cross_val_score(SVM, J, k, cv=5)
print("Result using Cross Validation = ",scores)

# calculate the confusion matrix
conf_metrics=metrics.confusion_matrix(k_test, Prediction)

# draw the confusion matrix
sns.heatmap(conf_metrics, annot=True)
plt.ylabel("Predicted", fontsize=16)
plt.xlabel("Actual", fontsize=16)
plt.title("Confusion Matrix for Digit Dataset",fontsize=18)
plt.show()
