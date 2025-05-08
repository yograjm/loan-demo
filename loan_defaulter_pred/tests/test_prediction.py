
import sys
from pathlib import Path
filepath = Path(__file__)
sys.path.append(str(filepath.parents[1]))

from predict import rf_model, make_prediction, sample_input_df

from data_preprocess import X_test, y_test

from sklearn.metrics import accuracy_score, f1_score


def test_model_accuracy():
    pred = rf_model.predict(X_test)
    acc = accuracy_score(pred, y_test)
    assert acc > 0.8, "Model accuracy is below 0.8"


def test_f1_score():
    pred = rf_model.predict(X_test)
    f1 = f1_score(pred, y_test)
    assert f1 > 0.8, "Model F1-score is below 0.8"



def test_make_prediction_function():
    label = make_prediction(sample_input_df)

    assert label in ["Likely to default", "Less likely to default"], "ErrorMessage: mismatch in prediction label sting"
