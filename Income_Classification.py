import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics

# read the CSV file using read_csv()
dataset=pd.read_csv('income_evaluation.csv')

# head () is used to show the first 5 samples
print(dataset.head())

# total number of records
print("<<<< Total number of records >>>>\n",dataset.shape)

#this dataset consists of 32561 observations and 15 features.
print(dataset.columns)

# stractural information about  and transform
print(dataset.info())

# statistical summary of dataset
print(dataset.describe(),"\n")

# check unique values from all objects
print(dataset.select_dtypes(include='object').nunique(), "\n")


# print the unique value from each column(type = object)
for column in dataset.select_dtypes(include=['object']).columns:
    print(f"{column}: {dataset[column].unique()}")
    
# label encoding for each column
for column in dataset.select_dtypes(include=['object']).columns:
    # initiliazing a label encoder object
    l_encoder=preprocessing.LabelEncoder()
    
    l_encoder.fit(dataset[column].unique())
    
    dataset[column]=l_encoder.transform(dataset[column])
    
    print(f"{column}: {dataset[column].unique()}\n")

# select the features (J) and the target variable (K)
J=dataset.drop(' income', axis=1)
K=dataset[' income']

# split the dataset into training and test data
J_train, J_test, K_train, K_test=train_test_split(
    J, K, test_size=0.3)


# finding the best hyperparameters
D_tree=DecisionTreeClassifier(class_weight='balanced')
hyperpara_grid={
    'random_state' :[0,1,2,3,42],
    'min_samples_leaf':[1,2,3,4,5],
    'min_samples_split':[1,2,3,4,5],
    'max_depth':[2,4,6,8,9]
    }

grid_search=GridSearchCV(D_tree, hyperpara_grid, cv=5)
grid_search.fit(J_train, K_train)
print(grid_search.best_params_)

# train the model
D_tree=DecisionTreeClassifier(random_state=2, min_samples_leaf=2, 
                              max_depth= 9 ,min_samples_split=2)
D_tree.fit(J_train, K_train)

# confusion matrix 
prediction=D_tree.predict(J_test)
cm=metrics.confusion_matrix(K_test, prediction)
sns.heatmap(cm, annot=True)
plt.title('Confusion Matrix', fontsize=16)
plt.xlabel('Actual',fontsize=16)
plt.ylabel('Predicted', fontsize=16)
plt.show()

# Find the metrics on test Dataset
print("\n*******  Performance Measures Evalution *******\n")

# find the acuracy 
accuracy=metrics.accuracy_score(K_test, prediction)
print("Accuracy  : ",round(accuracy*100,2), "%\n")

# Find the Jaccord Score
J_card_score=metrics.jaccard_score(K_test, prediction)
print("Jacccard Score : ", round(J_card_score*100,2), "%\n")

# Find the Recall Score
recall=metrics.recall_score(K_test, prediction)
print("Recall Score : ", round(recall*100,2), "%\n")
precision=metrics.precision_score(K_test, prediction)
print("Precision Score : ", round(precision*100,2), "%")
