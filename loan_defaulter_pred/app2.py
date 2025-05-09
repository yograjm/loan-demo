
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

from data_preprocess import X_test, y_test
from sklearn.model_selection import train_test_split
from predict import rf_model
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


# Metrics objects
acc_metric = prom.Gauge('loan_defaulter_model_accuracy_score', 'Accuracy score for few random 100 test samples')
f1_metric = prom.Gauge('loan_defaulter_model_f1_score', 'F1 score for few random 100 test samples')
precision_metric = prom.Gauge('loan_defaulter_model_precision_score', 'Precision score for few random 100 test samples')
recall_metric = prom.Gauge('loan_defaulter_model_recall_score', 'Recall score for few random 100 test samples')


def update_metrics():
    
    # Test data
    # Get the data from DB where new data is kept on storing
    _, X_test2, _, y_test2 = train_test_split(X_test, y_test, test_size=100, stratify=y_test)

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
