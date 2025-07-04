import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier  # ✅ Use RandomForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

import dagshub
dagshub.init(repo_owner='PranavThakur1', repo_name='mlflow-dagshub-demo', mlflow=True)

# Set MLflow tracking URI and experiment
mlflow.set_tracking_uri("https://dagshub.com/PranavThakur1/mlflow-dagshub-demo.mlflow")  # ✅ use .mlflow not .git
mlflow.set_experiment("iris-rf")  # ✅ Changed experiment name to separate from iris-dt

# Load the iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define parameters for Random Forest
n_estimators = 100
max_depth = 3

# Start MLflow run
with mlflow.start_run():
    # Train Random Forest model
    rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    rf.fit(X_train, y_train)

    # Predict
    y_pred = rf.predict(X_test)

    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)

    # Log parameters and metrics
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_metric("accuracy", accuracy)

    # Confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=iris.target_names,
                yticklabels=iris.target_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")

    cm_filename = "confusion_matrix_rf.png"
    plt.savefig(cm_filename)
    mlflow.log_artifact(cm_filename)

    # Log model
    mlflow.sklearn.log_model(rf, "random_forest")

    # Tags
    mlflow.set_tag("author", "pranav")
    mlflow.set_tag("model", "random forest")

    # Print result
    print("Accuracy:", accuracy)
