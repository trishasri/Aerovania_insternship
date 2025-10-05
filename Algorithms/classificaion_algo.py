from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# Load dataset
dataset = load_iris()
X, y = dataset.data, dataset.target

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


#logistic regression
log_reg = LogisticRegression(max_iter=200)
log_reg.fit(X_train, y_train)
pred = log_reg.predict(X_test)
print("\nLogistic Regression")
print("Accuracy:", accuracy_score(y_test, pred))
print(classification_report(y_test, pred, target_names=dataset.target_names))

#decision tree
tree = DecisionTreeClassifier()
tree.fit(X_train, y_train)
pred = tree.predict(X_test)
print("\nDecision Tree")
print("Accuracy:", accuracy_score(y_test, pred))
print(classification_report(y_test, pred, target_names=dataset.target_names))

#Random Forest
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
pred = rf.predict(X_test)
print("\nRandom Forest")
print("Accuracy:", accuracy_score(y_test, pred))
print(classification_report(y_test, pred, target_names=dataset.target_names))

#SVM
svm = SVC()
svm.fit(X_train, y_train)
pred = svm.predict(X_test)
print("\nSupport Vector Machine")
print("Accuracy:", accuracy_score(y_test, pred))
print(classification_report(y_test, pred, target_names=dataset.target_names))

#KNN
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
pred = knn.predict(X_test)
print("\nK-Nearest Neighbors")
print("Accuracy:", accuracy_score(y_test, pred))
print(classification_report(y_test, pred, target_names=dataset.target_names))

#GNB
nb = GaussianNB()
nb.fit(X_train, y_train)
pred = nb.predict(X_test)
print("\nNaive Bayes")
print("Accuracy:", accuracy_score(y_test, pred))
print(classification_report(y_test, pred, target_names=dataset.target_names))



