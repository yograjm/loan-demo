import numpy as np
import pandas as pd

df = pd.read_csv("dataset/credit_risk_dataset.csv")


############## DVC related code START ############################

# import os

# # Provide credentials and save them as environment variables. Later, whenever needed, access the credentials using environment variables only.
##


# import dvc.api
# import pandas as pd

# gtlb_username = os.environ['GTLB_USERNAME']
# gtlb_access_token = os.environ['GTLB_ACCESS_TOKEN']

# repo_name = "loan-data-repo"     # Change as per your GitLab repository name

# repo_url = 'https://' + gtlb_username + ':' + gtlb_access_token + '@gitlab.com/' + gtlb_username + '/' + repo_name
# # print(repo_url)

# # Data version to retrieve
# data_revision = 'v1.1'

# # Configurations to access remote storage
# remote_config = {
#     'password': os.environ["VM_PASSWORD"]
#     }

# # Open data file using dvc-api and load the dataset
# with dvc.api.open('data/credit_risk_dataset.csv', repo=repo_url, rev=data_revision, remote_config=remote_config) as file:  #remote_config=remote_config) as file:
#     df = pd.read_csv(file)

# print(df.tail())

############## DVC related code END ############################


# Handle missing values
df['loan_int_rate'] = df['loan_int_rate'].fillna(df['loan_int_rate'].mean())
df['person_emp_length'] = df['person_emp_length'].fillna(df['person_emp_length'].mean())


# Handle categorical columns
from custom_utils import home_ownership_mapping, loan_grade_mapping, default_on_file_mapping

df['person_home_ownership'] = df['person_home_ownership'].map(home_ownership_mapping)
df['loan_grade'] = df['loan_grade'].map(loan_grade_mapping)
df['cb_person_default_on_file'] = df['cb_person_default_on_file'].map(default_on_file_mapping)

# Label Encoder
from sklearn.preprocessing import LabelEncoder

loan_intent_encoder = LabelEncoder()

loan_intent_encoder.fit(df['loan_intent'])

df['loan_intent'] = loan_intent_encoder.transform(df['loan_intent'])


from sklearn.model_selection import train_test_split

# Fetures
X = df.drop('loan_status', axis=1)
# Target
y = df['loan_status']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
