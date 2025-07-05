import os

# Step 1: Set environment variables before importing mlflow
os.environ["MLFLOW_TRACKING_USERNAME"] = "PranavThakur1"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "f1539c3d1e3fbd4f576f8bb2c44496775ebb726e"

import mlflow
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib  # Used to save model as .pkl

# Step 2: Set tracking URI to your DagsHub repo
mlflow.set_tracking_uri("https://dagshub.com/PranavThakur1/mlflow-dagshub-demo.mlflow")

# Step 3: Load dataset
iris = load_iris()
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Start MLflow run
with mlflow.start_run():
    # Train model
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    # Make predictions and evaluate
    y_pred = rf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    # Log metric
    mlflow.log_metric("accuracy", acc)

    # Log parameters (optional)
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("random_state", 42)

    # Save model and log it as artifact
    joblib.dump(rf, "random_forest_model.pkl")
    mlflow.log_artifact("random_forest_model.pkl")

    # Log confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=iris.target_names,
                yticklabels=iris.target_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix_rf.png")
    mlflow.log_artifact("confusion_matrix_rf.png")

    # Tag the run for easy filtering
    mlflow.set_tag("author", "pranavv")
    mlflow.set_tag("experiment", "iris-rf")
    mlflow.set_tag("model", "random_forest")

    print("Accuracy:", acc)
