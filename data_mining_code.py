#Predicting the Result of the match based on team winning the Toss. We have considered the following columns "Toss_Decision","Match_Winner_Id","City_Name",
#"Team_Name_Id","Opponent_Team_Id", for predicting of the result





import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

data=match_data[["Toss_Decision","Match_Winner_Id","City_Name","Team_Name_Id","Opponent_Team_Id"]]

# Drop rows with missing values
data = data.dropna()

# Convert categorical columns to numerical using Label Encoding
label_encoder = LabelEncoder()
data["Toss_Decision"] = label_encoder.fit_transform(data["Toss_Decision"])
data["City_Name"] = label_encoder.fit_transform(data["City_Name"])

# Separate features (X) and target variable (y)
X = data.drop("Match_Winner_Id", axis=1)
y = data["Match_Winner_Id"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize different classifiers
rf_classifier = RandomForestClassifier(random_state=42)
gb_classifier = GradientBoostingClassifier(random_state=42)
svm_classifier = SVC(random_state=42)

# Train and evaluate Random Forest classifier
rf_classifier.fit(X_train, y_train)
rf_predictions = rf_classifier.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_predictions)
print("Random Forest Classifier Accuracy:", rf_accuracy)
print("Classification Report:")
print(classification_report(y_test, rf_predictions))

# Train and evaluate Gradient Boosting classifier
gb_classifier.fit(X_train, y_train)
gb_predictions = gb_classifier.predict(X_test)
gb_accuracy = accuracy_score(y_test, gb_predictions)
print("\nGradient Boosting Classifier Accuracy:", gb_accuracy)
print("Classification Report:")
print(classification_report(y_test, gb_predictions))

# Train and evaluate Support Vector Machine classifier
svm_classifier.fit(X_train, y_train)
svm_predictions = svm_classifier.predict(X_test)
svm_accuracy = accuracy_score(y_test, svm_predictions)
print("\nSupport Vector Machine Classifier Accuracy:", svm_accuracy)
print("Classification Report:")
print(classification_report(y_test, svm_predictions))

# Logistic Regression
logreg_classifier = LogisticRegression(random_state=42)
logreg_classifier.fit(X_train, y_train)
logreg_predictions = logreg_classifier.predict(X_test)
logreg_accuracy = accuracy_score(y_test, logreg_predictions)
print("\nLogistic Regression Accuracy:", logreg_accuracy)
print("Classification Report:")
print(classification_report(y_test, logreg_predictions))

# K-Nearest Neighbors
knn_classifier = KNeighborsClassifier()
knn_classifier.fit(X_train, y_train)
knn_predictions = knn_classifier.predict(X_test)
knn_accuracy = accuracy_score(y_test, knn_predictions)
print("\nK-Nearest Neighbors Accuracy:", knn_accuracy)
print("Classification Report:")
print(classification_report(y_test, knn_predictions))

# Naive Bayes
nb_classifier = GaussianNB()
nb_classifier.fit(X_train, y_train)
nb_predictions = nb_classifier.predict(X_test)
nb_accuracy = accuracy_score(y_test, nb_predictions)
print("\nNaive Bayes Accuracy:", nb_accuracy)
print("Classification Report:")
print(classification_report(y_test, nb_predictions))

# Decision Tree
dt_classifier = DecisionTreeClassifier(random_state=42)
dt_classifier.fit(X_train, y_train)
dt_predictions = dt_classifier.predict(X_test)
dt_accuracy = accuracy_score(y_test, dt_predictions)
print("\nDecision Tree Classifier Accuracy:", dt_accuracy)
print("Classification Report:")
print(classification_report(y_test, dt_predictions))

# AdaBoost
adaboost_classifier = AdaBoostClassifier(random_state=42)
adaboost_classifier.fit(X_train, y_train)
adaboost_predictions = adaboost_classifier.predict(X_test)
adaboost_accuracy = accuracy_score(y_test, adaboost_predictions)
print("\nAdaBoost Classifier Accuracy:", adaboost_accuracy)
print("Classification Report:")
print(classification_report(y_test, adaboost_predictions))
