import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

import dagshub 
dagshub.init(repo_owner='PranavThakur1', repo_name='mlflow-dagshub-demo', mlflow=True)



# Set MLflow tracking URI and experiment
mlflow.set_tracking_uri("https://github.com/PranavThakur1/mlflow-dagshub-demo.git")
mlflow.set_experiment("iris-dt")

# Load the iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define a parameter for the Decision Tree model
max_depth = 1

# Start MLflow run
with mlflow.start_run():
    # Train the model
    dt = DecisionTreeClassifier(max_depth=max_depth)
    dt.fit(X_train, y_train)

    # Make predictions
    y_pred = dt.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # Log metrics and parameters
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_metric("accuracy", accuracy)

    # Create a confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=iris.target_names,
                yticklabels=iris.target_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")

    # Save the plot as a PNG file
    cm_filename = "confusion_matrix.png"
    plt.savefig(cm_filename)

    # Log the confusion matrix image as artifact
    mlflow.log_artifact(cm_filename)

    # Log the model
    mlflow.sklearn.log_model(dt, "decision_tree")

    # Add tags (optional)
    mlflow.set_tag("author", "nitish")
    mlflow.set_tag("model", "decision tree")

    # Print accuracy
    print("Accuracy:", accuracy)
