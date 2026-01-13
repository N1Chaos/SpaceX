# =====================================
# SpaceX Landing Prediction - Python Script
# =====================================

# 1️⃣ Import des librairies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import preprocessing
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

sns.set(style="whitegrid")

# Fonction pour afficher la matrice de confusion
def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    ax = plt.subplot()
    sns.heatmap(cm, annot=True, ax=ax, fmt='d', cmap='Blues')
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix')
    ax.xaxis.set_ticklabels(['did not land', 'land'])
    ax.yaxis.set_ticklabels(['did not land', 'landed'])
    plt.show()

# =====================================
# 2️⃣ Charger les datasets
# =====================================
URL1 = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/datasets/dataset_part_2.csv"
URL2 = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/datasets/dataset_part_3.csv"

data = pd.read_csv(URL1)
X = pd.read_csv(URL2)

print("Data loaded:")
print(data.head())
print(X.head())

# =====================================
# 3️⃣ TASK 1: Create Y from data['Class']
# =====================================
Y = data['Class'].to_numpy()
print("Shape of Y:", Y.shape)

# =====================================
# 4️⃣ TASK 2: Standardize X
# =====================================
transform = preprocessing.StandardScaler()
X = transform.fit_transform(X)
print("X standardized")

# =====================================
# 5️⃣ TASK 3: Split into training and test sets
# =====================================
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("Y_test shape:", Y_test.shape)

# =====================================
# 6️⃣ TASK 4: Logistic Regression with GridSearchCV
# =====================================
parameters_lr = {"C":[0.01,0.1,1], 'penalty':['l2'], 'solver':['lbfgs']}
lr = LogisticRegression()
logreg_cv = GridSearchCV(lr, parameters_lr, cv=10)
logreg_cv.fit(X_train, Y_train)

print("Logistic Regression - Best parameters:", logreg_cv.best_params_)
print("Logistic Regression - Best CV accuracy:", logreg_cv.best_score_)

# =====================================
# TASK 5: Accuracy and confusion matrix for Logistic Regression
# =====================================
accuracy_lr = logreg_cv.score(X_test, Y_test)
print("Logistic Regression Test Accuracy:", accuracy_lr)

yhat_lr = logreg_cv.predict(X_test)
plot_confusion_matrix(Y_test, yhat_lr)

# =====================================
# TASK 6: SVM with GridSearchCV
# =====================================
parameters_svm = {'kernel':['linear','rbf','poly','sigmoid'],
                  'C': np.logspace(-3,3,5),
                  'gamma': np.logspace(-3,3,5)}
svm = SVC()
svm_cv = GridSearchCV(svm, parameters_svm, cv=10)
svm_cv.fit(X_train, Y_train)

print("SVM - Best parameters:", svm_cv.best_params_)
print("SVM - Best CV accuracy:", svm_cv.best_score_)

# TASK 7: SVM accuracy and confusion matrix
accuracy_svm = svm_cv.score(X_test, Y_test)
print("SVM Test Accuracy:", accuracy_svm)

yhat_svm = svm_cv.predict(X_test)
plot_confusion_matrix(Y_test, yhat_svm)

# =====================================
# TASK 8: Decision Tree with GridSearchCV
# =====================================
parameters_tree = {'criterion': ['gini', 'entropy'],
                   'splitter': ['best', 'random'],
                   'max_depth': [2*n for n in range(1,10)],
                   'max_features': ['auto', 'sqrt'],
                   'min_samples_leaf': [1,2,4],
                   'min_samples_split': [2,5,10]}
tree = DecisionTreeClassifier()
tree_cv = GridSearchCV(tree, parameters_tree, cv=10)
tree_cv.fit(X_train, Y_train)

print("Decision Tree - Best parameters:", tree_cv.best_params_)
print("Decision Tree - Best CV accuracy:", tree_cv.best_score_)

# TASK 9: Decision Tree accuracy and confusion matrix
accuracy_tree = tree_cv.score(X_test, Y_test)
print("Decision Tree Test Accuracy:", accuracy_tree)

yhat_tree = tree_cv.predict(X_test)
plot_confusion_matrix(Y_test, yhat_tree)

# =====================================
# TASK 10: KNN with GridSearchCV
# =====================================
parameters_knn = {'n_neighbors': list(range(1,11)),
                  'algorithm':['auto','ball_tree','kd_tree','brute'],
                  'p':[1,2]}
KNN = KNeighborsClassifier()
knn_cv = GridSearchCV(KNN, parameters_knn, cv=10)
knn_cv.fit(X_train, Y_train)

print("KNN - Best parameters:", knn_cv.best_params_)
print("KNN - Best CV accuracy:", knn_cv.best_score_)

# TASK 11: KNN accuracy and confusion matrix
accuracy_knn = knn_cv.score(X_test, Y_test)
print("KNN Test Accuracy:", accuracy_knn)

yhat_knn = knn_cv.predict(X_test)
plot_confusion_matrix(Y_test, yhat_knn)

# =====================================
# TASK 12: Compare Models
# =====================================
accuracies = {
    "Logistic Regression": accuracy_lr,
    "SVM": accuracy_svm,
    "Decision Tree": accuracy_tree,
    "KNN": accuracy_knn
}

best_model = max(accuracies, key=accuracies.get)
print("Best performing model:", best_model, "with accuracy:", accuracies[best_model])
