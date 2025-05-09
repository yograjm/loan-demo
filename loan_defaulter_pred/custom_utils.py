
# Handle categorical columns
home_ownership_mapping = {'MORTGAGE': 0, 'RENT': 1, 'OWN': 2, 'OTHER': 3}
loan_grade_mapping = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6}
default_on_file_mapping = {'N': 0, 'Y': 1}

mlflow_tracking_uri = "https://redesigned-space-sniffle-7454xj5gj6xf95r-5000.app.github.dev/"

experiment_name = "Loan-Defaulter-Prediction"

registered_model_name = "sklearn-loan-defaulter-model"
