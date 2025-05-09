
from fastapi import FastAPI

app = FastAPI()

# GET method

@app.get("/hi")
def func1():
    return {"message": "Hello everyone!"}

# Query parameter     # /hello?fname=Suresh&lname=Kumar
@app.get("/hello")
def func2(fname, lname):
    return {"message": "Hello " + fname + " " + lname + " !"}

# Path parameter     # /hello/Suresh/Kumar
@app.get("/hello/{fname}/{lname}")
def func2(fname, lname):
    return {"message": "Hello " + fname + " " + lname + " !"}


# Swagger Documentation  avaiable on this endpoint ->   /docs 

# POST;   with query parameter
@app.post("/hey")
def func3(fname, lname):
    return {"message": "Hello " + fname + " " + lname + " !"}



# POST, with request body
from pydantic import BaseModel

class Person(BaseModel):
    name: str
    age: int
    address: str


@app.post("/person")
def func4(person: Person):
    response = "Your name is " + person.name
    response += "Your age is " + str(person.age)
    response += "Your address is " + person.address
    return {"message": response}


# Prediction

class Feature(BaseModel):
    person_age: float
    person_income: float
    person_home_ownership: str
    person_emp_length: int
    loan_intent: str
    loan_grade: str
    loan_amnt: float
    loan_int_rate: float
    loan_percent_income: float
    cb_person_default_on_file: str
    cb_person_cred_hist_length: int



from app import predict_loan_status

@app.post("/predict")
def func5(f: Feature):
    label = predict_loan_status(f.person_age, 
                        f.person_income, 
                        f.person_home_ownership, 
                        f.person_emp_length, 
                        f.loan_intent, 
                        f.loan_grade, 
                        f.loan_amnt, 
                        f.loan_int_rate, 
                        f.loan_percent_income, 
                        f.cb_person_default_on_file, 
                        f.cb_person_cred_hist_length
                        )
    return {"prediction": label}



############## Prometheus related code START ##################

import prometheus_client as prom

from predict import rf_model, loan_intent_encoder
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from custom_utils import home_ownership_mapping, loan_grade_mapping, default_on_file_mapping

import pandas as pd
import psycopg2
from psycopg2 import sql

# Metrics objects
acc_metric = prom.Gauge('loan_defaulter_model_accuracy_score', 'Accuracy score for few random 100 test samples')
f1_metric = prom.Gauge('loan_defaulter_model_f1_score', 'F1 score for few random 100 test samples')
precision_metric = prom.Gauge('loan_defaulter_model_precision_score', 'Precision score for few random 100 test samples')
recall_metric = prom.Gauge('loan_defaulter_model_recall_score', 'Recall score for few random 100 test samples')


def update_metrics():
    
    # Test data
    # Get the data from DB where new data is kept on storing

    # Database connection parameters
    db_params = {
        'dbname': 'storedb',
        'user': 'postgres',
        'password': 'mypassword',
        'host': '13.127.39.198',  # EC2 public IP # or your database host
        'port': '5432'  #'5432'        # default PostgreSQL port
    }

    try:
        conn = psycopg2.connect(**db_params)
        cursor = conn.cursor()

        # Read existing data from the table
        query = "SELECT * FROM loans_data;"  # SQL query to select all data from the table
        cursor.execute(query)

        # Fetch all results
        rows = cursor.fetchall()

        # Get column names from the cursor
        column_names = [desc[0] for desc in cursor.description]

        # Create a DataFrame from the fetched data
        df = pd.DataFrame(rows, columns=column_names)
        print(f"Existing rows in db: {len(df)}")

    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        # Close the cursor and connection
        if cursor:
            cursor.close()
        if conn:
            conn.close()
    
    # Handle missing values
    df['loan_int_rate'] = df['loan_int_rate'].fillna(df['loan_int_rate'].mean())
    df['person_emp_length'] = df['person_emp_length'].fillna(df['person_emp_length'].mean())

    # Handle categorical columns
    df['person_home_ownership'] = df['person_home_ownership'].map(home_ownership_mapping)
    df['loan_grade'] = df['loan_grade'].map(loan_grade_mapping)
    df['cb_person_default_on_file'] = df['cb_person_default_on_file'].map(default_on_file_mapping)

    df['loan_intent'] = loan_intent_encoder.transform(df['loan_intent'])

    # Fetures
    X_test2 = df.drop('loan_status', axis=1)
    # Target
    y_test2 = df['loan_status']


    # Do prediction
    y_pred = rf_model.predict(X_test2)

    # Calculate metrics
    acc = accuracy_score(y_pred, y_test2)
    f1 = f1_score(y_pred, y_test2)
    precision = precision_score(y_pred, y_test2)
    recall = recall_score(y_pred, y_test2)

    # Update metrics values
    acc_metric.set(round(acc, 3))
    f1_metric.set(round(f1, 3))
    precision_metric.set(round(precision, 3))
    recall_metric.set(round(recall, 3))


# # HELP loan_defaulter_model_accuracy_score Accuracy score for few random 100 test samples
# # TYPE loan_defaulter_model_accuracy_score gauge
# loan_defaulter_model_accuracy_score 0.85

from fastapi import Response

@app.get("/metrics")
def func():
    update_metrics()
    return Response(media_type="text/plain", content= prom.generate_latest())


############## Prometheus related code END ##################


# Webserver -> Uvicorn
import uvicorn

uvicorn.run(app, host="0.0.0.0", port=8080)
