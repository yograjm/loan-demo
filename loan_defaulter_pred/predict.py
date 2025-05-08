
import pandas as pd
from custom_utils import home_ownership_mapping, loan_grade_mapping, default_on_file_mapping

import joblib

# Load model
# rf_model = joblib.load("trained_model/rf_model_loan_default_pred.pkl")

################### MLflow related code START #################################

import mlflow 
import mlflow.pyfunc
from custom_utils import mlflow_tracking_uri, registered_model_name

mlflow.set_tracking_uri(mlflow_tracking_uri)

# Create MLflow client
client = mlflow.tracking.MlflowClient()

# Load model via 'models'
model_name = registered_model_name              #"loan-defaulter-pred-model"

model_info = client.get_model_version_by_alias(name=model_name, alias="production")

print(f'Model version fetched: {model_info.version}')

rf_model = mlflow.pyfunc.load_model(model_uri = f"models:/{model_name}@production")

# models:/sklearn-loan-defaulter-model@production


## Load label_encoder

# Run ID
run_id = model_info.run_id

# Download artifact path
artifact_path = "label_encoder/loan_intent_encoder.pkl"

# Download artifact to a local path
local_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=artifact_path)

# Load encoder
loan_intent_encoder = joblib.load(local_path)

################### MLflow related code END #################################


# Load encoder
# loan_intent_encoder = joblib.load("trained_model/loan_intent_encoder.pkl")


# Inference
sample_input = {'person_age': 30,
                'person_income': 1000000,
                'person_home_ownership': home_ownership_mapping['RENT'],
                'person_emp_length': 6.0,
                'loan_intent': loan_intent_encoder.transform(['DEBTCONSOLIDATION'])[0],
                'loan_grade': loan_grade_mapping['B'],
                'loan_amnt': 500000,
                'loan_int_rate': 11.89,
                'loan_percent_income': 0.5,
                'cb_person_default_on_file': default_on_file_mapping['N'],
                'cb_person_cred_hist_length': 4}

sample_input_df = pd.DataFrame(sample_input, index=[0])



def make_prediction(sample_input_df):
    prediction = rf_model.predict(sample_input_df)
    label = "Likely to default" if prediction[0] == 1 else "Less likely to default"
    return label


if __name__ == "__main__":
    pred = make_prediction(sample_input_df)
    print(pred)
