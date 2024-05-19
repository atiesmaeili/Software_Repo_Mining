import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score , precision_score, recall_score
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier

"""# Predictive Model"""

# Load the CSV file into a Pandas dataframe
data = pd.read_csv('merged_file.csv')
data = data[["commit_msg","goal"]]
data = data.dropna()

#data["target"] = data["difference"].apply(lambda x: 0 if x<0 else 1)
# Extract the target variable
y = data['goal']
# y = y.dropna()
# Extract the feature columns
X = data["commit_msg"]
# X = X.dropna()
#df.drop('readability_status', axis=1)

# Vectorize the text data in the feature columns using CountVectorizer
vectorizer = CountVectorizer()
X_vec = vectorizer.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)

# Train a random forest classifier on the training set
rf = RandomForestClassifier()
rf.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = rf.predict(X_test)

# Evaluate the performance of the model using accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Create a confusion matrix
conf_mat = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_mat)
plt.figure(figsize=(8,8))
sns.heatmap(conf_mat, annot=True, cmap='Blues')

plt.title('Total')
plt.xlabel('Predicted')
plt.ylabel('Actual')

# Show the plot
plt.show()

# Evaluate the performance of the model on the test set
accuracy = accuracy_score(y_test, y_pred)

# calculate precision and recall
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

# print the results
print("Precision:", precision)
print("Recall:", recall)

"""# Cross Project

### codec and vfs as train , bcel as test
"""

# Load the data
train_data = pd.read_csv('codec-vfs.csv')
test_data = pd.read_csv('bcel.csv')

# Create a CountVectorizer object
vectorizer = CountVectorizer()

# Vectorize the text data in the training dataset
X_train = vectorizer.fit_transform(train_data['commit_msg'])

# Vectorize the text data in the test dataset
X_test = vectorizer.transform(test_data['commit_msg'])

# Get the target variable in the training and test datasets
y_train = train_data['goal']
y_test = test_data['goal']

# Create a random forest classifier with 100 trees
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = rf.predict(X_test)

# Evaluate the performance of the model using accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Create a confusion matrix
conf_mat = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_mat)
plt.figure(figsize=(8,8))
sns.heatmap(conf_mat, annot=True, cmap='Blues')

plt.title('test: bcel')
plt.xlabel('Predicted')
plt.ylabel('Actual')

# Show the plot
plt.show()

# Evaluate the performance of the model on the test set
accuracy = accuracy_score(y_test, y_pred)

# calculate precision and recall
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

#print the results
print("Precision:", precision)
print("Recall:", recall)

"""### bcel and vfs as train , codec as test"""

# Load the data
proj2_train = pd.read_csv('bcel-vfs.csv')
proj2_test = pd.read_csv('codec.csv')

# Create a CountVectorizer object
vectorizer = CountVectorizer()

# Vectorize the text data in the training dataset
X_train = vectorizer.fit_transform(train_data['commit_msg'])

# Vectorize the text data in the test dataset
X_test = vectorizer.transform(test_data['commit_msg'])

# Get the target variable in the training and test datasets
y_train = train_data['goal']
y_test = test_data['goal']

# Create a random forest classifier with 100 trees
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = rf.predict(X_test)

# Evaluate the performance of the model using accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Create a confusion matrix
conf_mat = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_mat)
plt.figure(figsize=(8,8))
sns.heatmap(conf_mat, annot=True, cmap='Blues')

plt.title('test: codec')
plt.xlabel('Predicted')
plt.ylabel('Actual')

# Show the plot
plt.show()



# Evaluate the performance of the model on the test set
accuracy = accuracy_score(y_test, y_pred)

# calculate precision and recall
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

# print the results
print("Precision:", precision)
print("Recall:", recall)

"""### codec and bcel as train , vfs as test"""

# Load the data
proj3_train = pd.read_csv('codec-bcel.csv')
proj3_test = pd.read_csv('vfs.csv')

# Create a CountVectorizer object
vectorizer = CountVectorizer()

# Vectorize the text data in the training dataset
X_train = vectorizer.fit_transform(train_data['commit_msg'])

# Vectorize the text data in the test dataset
X_test = vectorizer.transform(test_data['commit_msg'])

# Get the target variable in the training and test datasets
y_train = train_data['goal']
y_test = test_data['goal']

# Create a random forest classifier with 100 trees
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = rf.predict(X_test)

# Evaluate the performance of the model using accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Create a confusion matrix
conf_mat = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_mat)
plt.figure(figsize=(8,8))
sns.heatmap(conf_mat, annot=True, cmap='Blues')

plt.title('test:vfs')
plt.xlabel('Predicted')
plt.ylabel('Actual')

# Show the plot
plt.show()

# Evaluate the performance of the model on the test set
accuracy = accuracy_score(y_test, y_pred)

# calculate precision and recall
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
# print the results
print("Precision:", precision)
print("Recall:", recall)

