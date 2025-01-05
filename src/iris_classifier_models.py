import os
import pandas as pd
import numpy as np
import mlflow
from math import sqrt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression 
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import r2_score,mean_squared_error,accuracy_score, confusion_matrix,log_loss
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report

print("Processing started")
columns=["SepalLengthCm","SepalWidthCm","PetalLengthCm","PetalWidthCm","Species"]
base_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

dataset = pd.read_csv(base_folder + '\data\iris_data.csv', names=columns)
#print(dataset.head())

label_encoder = LabelEncoder() 
dataset['Species']= label_encoder.fit_transform(dataset['Species']) 

dataset['Species'].unique()

X = dataset.iloc[:, [2,3]].values
y = dataset.iloc[:, 4].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .20, random_state = 2)
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)

mlflow.set_experiment("iris_DecisionTreeClassifier")
with mlflow.start_run():
    # Model1 - Decision Tree Classifier

    #Trial #1
    #dt_random_state = 0
    #dt_criterion = 'entropy'
    
    #Trial #2
    #dt_random_state = 45
    #dt_criterion = 'gini'

    #Trial #3
    dt_random_state = 20
    dt_criterion = 'log_loss'

    classifier = DecisionTreeClassifier(criterion = dt_criterion, random_state = dt_random_state)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    rmse = sqrt(mean_squared_error(y_test, y_pred))

    mlflow.log_param("data_size", len(dataset))
    mlflow.log_param("train_size", len(X_train))
    mlflow.log_param("test_size", len(X_test))
    mlflow.log_param("feature_count", len(X_train[0]))
    mlflow.log_param("target_count", len(np.unique(y)))
    mlflow.log_param("criterion", 'entropy')
    mlflow.log_param("random_state", dt_random_state)

    mlflow.log_metric("Accuracy", accuracy_score( y_test,y_pred))
    mlflow.log_metric("Precision", precision_score(y_test,y_pred, average='weighted'))
    mlflow.log_metric("Recall", recall_score(y_test,y_pred, average='weighted'))
    mlflow.log_metric("f1-Score", f1_score(y_test,y_pred, average='weighted'))
    mlflow.log_metric("RMSE value", rmse)
    mlflow.log_metric("R2 Score", r2_score(y_test, y_pred))
    mlflow.sklearn.log_model(classifier, 'DecisionTreeClassifier')

mlflow.set_experiment("iris_KNeighborsClassifier")
with mlflow.start_run():
    # Model2 - KNN Classifier
    
    #Trial #1
    #kn_neighbors = 3
    #kn_algorithm = 'auto'
    
    #Trial #2
    #kn_neighbors = 5
    #kn_algorithm = 'brute'

    #Trial #3
    kn_neighbors = 10
    kn_algorithm = 'ball_tree'

    model = KNeighborsClassifier(n_neighbors = kn_neighbors, algorithm = kn_algorithm)
    model.fit(X_train,y_train)
    predict = model.predict(X_test)
    #for checking the model accuracy
    rmse = sqrt(mean_squared_error(y_test, predict))

    mlflow.log_param("data_size", len(dataset))
    mlflow.log_param("train_size", len(X_train))
    mlflow.log_param("test_size", len(X_test))
    mlflow.log_param("feature_count", len(X_train[0]))
    mlflow.log_param("target_count", len(np.unique(y)))
    mlflow.log_param("n_neighbors", kn_neighbors)
    mlflow.log_param("algorithm", 'auto')

    mlflow.log_metric("Accuracy", accuracy_score(y_test,predict))
    mlflow.log_metric("Precision", precision_score(y_test,predict, average='weighted'))
    mlflow.log_metric("Recall", recall_score(y_test,predict, average='weighted'))
    mlflow.log_metric("f1-Score", f1_score(y_test,predict, average='weighted'))
    mlflow.log_metric("RMSE value", rmse)
    mlflow.log_metric("R2 Score", r2_score(y_test, predict))
    
    mlflow.sklearn.log_model(model, 'KNeighborsClassifier')

mlflow.set_experiment("iris_LogisticRegression")
with mlflow.start_run():    
    # Model3 - Logistic Regression
    
    #Trial #1
    #lr_random_state = 42
    #lr_solver = 'lbfgs'
    
    #Trial #2
    #lr_random_state = 18
    #lr_solver = 'newton-cholesky'

    #Trial #3
    lr_random_state = 45
    lr_solver = 'sag'
    
    model = LogisticRegression(random_state= lr_random_state, solver= lr_solver)
    model.fit(X_train,y_train) 
    prediction=model.predict(X_test) 
    rmse = sqrt(mean_squared_error(y_test, prediction))

    mlflow.log_param("data_size", len(dataset))
    mlflow.log_param("train_size", len(X_train))
    mlflow.log_param("test_size", len(X_test))
    mlflow.log_param("feature_count", len(X_train[0]))
    mlflow.log_param("target_count", len(np.unique(y)))
    mlflow.log_param("random_state", lr_random_state)
    mlflow.log_param("solver", lr_solver) 


    mlflow.log_metric("Accuracy", accuracy_score(y_test,prediction))
    mlflow.log_metric("Precision", precision_score(y_test,prediction, average='weighted'))
    mlflow.log_metric("Recall", recall_score(y_test,prediction, average='weighted'))
    mlflow.log_metric("f1-Score", f1_score(y_test,prediction, average='weighted'))
    mlflow.log_metric("RMSE value", rmse)
    mlflow.log_metric("R2 Score", r2_score(y_test, prediction))
    
    mlflow.sklearn.log_model(model, 'LogisticRegression')

print("Processing completed")
exit(0)