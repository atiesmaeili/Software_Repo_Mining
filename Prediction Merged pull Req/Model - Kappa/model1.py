
# Predictive Model
## Import Essential


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold, cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score , precision_score, recall_score
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


#Read Data
data = pd.read_excel("Final_data.xlsx")



X = data.iloc[:, :-1] # All columns except the last one
y = data.iloc[:, -1] # The last column
X = pd.get_dummies(X)

# Decision Tree Classifier
clf = DecisionTreeClassifier(random_state=42)

#k-fold Cross Validation
X, y = datasets.load_iris(return_X_y=True)
k_folds = KFold(n_splits = 50)
scores = cross_val_score(clf, X, y, cv = k_folds)

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

clf.fit(X_train, y_train)

# Create a confusion matrix
conf_mat = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8,8))
sns.heatmap(conf_mat, annot=True, cmap='Blues')

plt.title('Confusion Matrix of Decision Tree')
plt.xlabel('Predicted')
plt.ylabel('Actual')

# Show the plot
plt.show()


print("Confusion Matrix:")
print(conf_mat)

# Calculate the accuracy score of the model
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)
print("Cross Validation Scores: ", scores)
print("Average CV Score: ", scores.mean())
print("Scores of train: ", clf.score(X_train, y_train))
print("Test accuracy:", clf.score(X_test, y_test))

### Logistic Regression


# Separate the target variable from the features
X = data.drop('Merged', axis=1)
y = data['Merged']


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a logistic regression object
log_reg = LogisticRegression()

#k-fold Cross Validation
X, y = datasets.load_iris(return_X_y=True)
k_folds_2 = KFold(n_splits = 50)
scores = cross_val_score(clf, X, y, cv = k_folds)

# Train the model on the training data
log_reg.fit(X_train, y_train)


# Predict the class labels for the testing data
y_pred = log_reg.predict(X_test)

# Evaluate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)

# Create a confusion matrix
conf_mat = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_mat)
# Create a confusion matrix
conf_mat = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8,8))
sns.heatmap(conf_mat, annot=True, cmap='Blues')

plt.title('Confusion Matrix of Logistic Regression')
plt.xlabel('Predicted')
plt.ylabel('Actual')

# Show the plot
plt.show()

# Calculate the accuracy score of the model
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)
print("Cross Validation Scores: ", scores)
print("Average CV Score: ", scores.mean())
print("Scores of train: ", log_reg.score(X_train, y_train))
print("Test accuracy:", log_reg.score(X_test, y_test))

#Random Forest
X = data.drop('Merged', axis=1)
y = data['Merged']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)

rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

#k-fold Cross Validation
X, y = datasets.load_iris(return_X_y=True)
k_folds = KFold(n_splits = 50)
scores = cross_val_score(clf, X, y, cv = k_folds)

# Create a confusion matrix
conf_mat = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_mat)

# Create a confusion matrix
conf_mat = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8,8))
sns.heatmap(conf_mat, annot=True, cmap='Blues')

plt.title('Confusion Matrix of RandomForest')
plt.xlabel('Predicted')
plt.ylabel('Actual')

# Show the plot
plt.show()

# Calculate the accuracy score of the model
accuracy = accuracy_score(y_test, y_pred)
print(f"The accuracy of the Random Forest model is: {accuracy}")
print("Cross Validation Scores: ", scores)
print("Average CV Score: ", scores.mean())
print("Scores of train: ", rf.score(X_train, y_train))
print("Test accuracy:", rf.score(X_test, y_test))



