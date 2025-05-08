
from data_preprocess import X_train, X_test, y_train, y_test, loan_intent_encoder


from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

def evaluate_model(y_pred, y_test):
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    print(f"Accuracy: {round(acc, 3)}")
    print(f"F1 Score: {round(f1, 3)}")
    print(f"Precision: {round(precision, 3)}")
    print(f"Recall: {round(recall, 3)}")


# Random Forest
from sklearn.ensemble import RandomForestClassifier

n_estimators = 150
max_depth = 9
max_features = 6
random_state = 42

rf_model = RandomForestClassifier(n_estimators = n_estimators,
                                  max_depth = max_depth,
                                  max_features = max_features,
                                  random_state = random_state)

rf_model.fit(X_train, y_train)

print("Model trained successfully!")

y_pred = rf_model.predict(X_test)

evaluate_model(y_test, y_pred)

# Save model
import joblib
# joblib.dump(rf_model, "trained_model/rf_model_loan_default_pred.pkl")

# save encoder
joblib.dump(loan_intent_encoder, "trained_model/loan_intent_encoder.pkl")



################### MLflow related code START #################################

import mlflow

# Set the tracking URI to the server
from custom_utils import mlflow_tracking_uri, experiment_name

mlflow.set_tracking_uri(mlflow_tracking_uri)

# Set an experiment name, unique and case-sensitive
# It will create a new experiment if the experiment with given doesn't exist
exp = mlflow.set_experiment(experiment_name = experiment_name)  # "Loan-Defaulter-Prediction"

# Start RUN
mlflow.start_run(experiment_id= exp.experiment_id)        # experiment id under which to create the current run

# Log parameters
mlflow.log_param("n_estimators", n_estimators)
mlflow.log_param("max_depth", max_depth)
mlflow.log_param("max_features", max_features)
mlflow.log_param("random_state", random_state)


# Log performance metrics
training_accuracy = accuracy_score(rf_model.predict(X_train), y_train)
testing_accuracy = accuracy_score(rf_model.predict(X_test), y_test)
mlflow.log_metric("training_accuracy", training_accuracy)
mlflow.log_metric("testing_accuracy", testing_accuracy)


# Log loan_intent_encoder as an artifact
mlflow.log_artifact("trained_model/loan_intent_encoder.pkl", artifact_path="label_encoder")


# Log model

# Load current 'production' model via 'models'
import mlflow.pyfunc
from custom_utils import registered_model_name
model_name = registered_model_name

client = mlflow.tracking.MlflowClient()

try:
    # Capture the test-accuracy-score of the existing prod-model
    prod_model_info = client.get_model_version_by_alias(name=model_name, alias="production")         # 1fetch prod-model info
    prod_model_run_id = prod_model_info.run_id                   # run_id of the run associated with prod-model
    prod_run = client.get_run(run_id=prod_model_run_id)          # get run info using run_id
    prod_accuracy = prod_run.data.metrics['testing_accuracy']    # get metrics values

    # Capture the version of the last-trained model
    latest_model_info = client.get_model_version_by_alias(name=model_name, alias="last-trained")           # fetch latest-model info
    latest_model_version = int(latest_model_info.version)              # latest-model version
    new_version = latest_model_version + 1                      # new model version

except Exception as e:
    print(e)
    new_version = 1


# Criterion to Log trained model
if new_version > 1:
    if prod_accuracy < testing_accuracy:
        print("Trained model is better than the existing model in production, will use this model in production!")
        better_model = True
    else:
        print("Trained model is not better than the existing model in production!")
        better_model = False
    first_model = False
else:
    print("No existing model in production, registering a new model!")
    first_model = True


# Register new model/version of model
mlflow.sklearn.log_model(sk_model = rf_model, 
                        artifact_path="trained_model",
                        registered_model_name=model_name,
                        )

# Add 'last-trained' alias to this new model version
client.set_registered_model_alias(name=model_name, alias="last-trained", version=str(new_version))


if first_model or better_model:
    # Promote the model to production by adding 'production' alias to this new model version
    client.set_registered_model_alias(name=model_name, alias="production", version=str(new_version))
else:
    # Don't promote this new model version
    pass


# End an active MLflow run
mlflow.end_run()

################### MLflow related code END #################################
