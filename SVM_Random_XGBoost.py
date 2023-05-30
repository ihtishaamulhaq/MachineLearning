
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import datasets, svm
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

# load the dataset from sklearn libraray
X, y = datasets.load_iris(return_X_y=True)

# finding the shape of dataset
X.shape, y.shape

# use the train_test_split function to divide the data into taraining and test data. 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=0)

X_train.shape, y_train.shape

print(X_test.shape, y_test.shape)

# Here we use the SVM classifier with Linear Kernal
SVM = svm.SVC(kernel='linear', C=1)
SVM.fit(X_train, y_train)

print("SVM with linear kernel results= ",SVM.score(X_test, y_test))


# random forest Regressor implemenattion on iris Dataset

n_features=4 
x_train_val, x_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)
x_train, x_val, y_train, y_val = train_test_split(
    x_train_val, y_train_val, test_size=0.2)
model = RandomForestRegressor(50, max_depth=15, max_features=n_features)
model.fit(x_train, y_train)
print("Random Forest Regressor results= ",model.score(x_val, y_val))

#  XGBoost Classifier with learning rate 0.1, maximum depth 1, and 330 estimaters

model2 = XGBClassifier(objective='multiclass:softmax', learning_rate = 0.1,
              max_depth = 1, n_estimators = 330)
model2.fit(x_train, y_train)
preds = model2.predict(x_test)
print("XGBoost results= ",sum(preds==y_test)/len(y_test))

# Cross Validation Applied on iris Dataset 

scores = cross_val_score(clf, X, y, cv=5)
print("Result using Cross Validation = ",scores)
