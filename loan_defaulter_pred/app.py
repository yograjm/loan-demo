
"""## Gradio Implementation"""

import pandas as pd
import gradio
import gradio as gr

# Load encoder
from predict import loan_intent_encoder, make_prediction

# Import mappings
from custom_utils import home_ownership_mapping, loan_grade_mapping, default_on_file_mapping


# Input elements
in_person_age = gr.Number(label="Age",value=30)
in_person_income = gr.Number(label="Annual Income", value=1000000)
in_person_home_ownership = gr.Radio(choices=["MORTGAGE", "RENT", "OWN", "OTHER"], label="Home Ownership")
in_person_emp_length = gr.Number(label="Employment Length", value=6)
in_loan_intent = gr.Radio(choices=list(loan_intent_encoder.classes_), label="Loan Intent")
in_loan_grade = gr.Radio(choices=["A", "B", "C", "D", "E", "F", "G"], label="Loan Grade")
in_loan_amnt = gr.Number(label="Loan Amount", value=500000)
in_loan_int_rate = gr.Number(label="Interest Rate", value=12)
in_loan_percent_income = gr.Number(label="Percent Income", value=0.5)
in_cb_person_default_on_file = gr.Radio(choices=["N", "Y"], label="Historical Default")
in_cb_person_cred_hist_length = gr.Number(label="Credit History Length", value=4)


# Output element
out_loan_status = gr.Textbox(label="Prediction")


# Function
def predict_loan_status(person_age, person_income, person_home_ownership, person_emp_length, loan_intent, loan_grade, loan_amnt, loan_int_rate, loan_percent_income, cb_person_default_on_file, cb_person_cred_hist_length):
    sample_input = {'person_age': person_age,
                    'person_income': person_income,
                    'person_home_ownership': home_ownership_mapping[person_home_ownership],
                    'person_emp_length': person_emp_length,
                    'loan_intent': loan_intent_encoder.transform([loan_intent]),
                    'loan_grade': loan_grade_mapping[loan_grade],
                    'loan_amnt': loan_amnt,
                    'loan_int_rate': loan_int_rate,
                    'loan_percent_income': loan_percent_income,
                    'cb_person_default_on_file': default_on_file_mapping[cb_person_default_on_file],
                    'cb_person_cred_hist_length': cb_person_cred_hist_length}

    sample_input_df = pd.DataFrame(sample_input, index=[0])
    label = make_prediction(sample_input_df)

    return label

# Create interface
iface = gr.Interface(fn=predict_loan_status,
                     inputs=[in_person_age, in_person_income, in_person_home_ownership, in_person_emp_length, in_loan_intent, in_loan_grade, in_loan_amnt, in_loan_int_rate, in_loan_percent_income, in_cb_person_default_on_file, in_cb_person_cred_hist_length],
                     outputs=out_loan_status,
                     title="Loan Default Prediction")

if __name__ == "__main__":
    iface.launch(server_name= "0.0.0.0", server_port = 7860)
