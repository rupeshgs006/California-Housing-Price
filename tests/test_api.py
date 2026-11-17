from fastapi.testclient import TestClient
from src.api import app
import logging
from unittest.mock import patch


import pytest


valid_data={
    "MedInc": 3.5,
    "HouseAge": 25,
    "AveRooms": 5.2,
    "AveBedrms": 1.1,
    "Population": 1200,
    "AveOccup": 3.0,
    "Latitude": 34.05,
    "Longitude": -118.25
}

def test_health():
    with TestClient(app) as client:
        response=client.get("/health")

    assert response.status_code==200
    assert response.json()=={"status":"healthy"}

def test_valid_prediction():
    with TestClient(app) as client:
        response=client.post("/predict",json=valid_data)
    assert response.status_code==200

    result=response.json()
    assert "predicted_price" in result
    assert result["currency"] =="USD"
    assert isinstance(result["predicted_price"],float)

def test_missing_feature():
    invalid_data=valid_data.copy()
    invalid_data.pop("MedInc")

    with TestClient(app) as client:
        response=client.post("/predict",json=invalid_data)
    assert response.status_code==422

def test_invalid_latitude():
    invalid_data=valid_data.copy()
    invalid_data["Latitude"]=150
    with TestClient(app) as client:
        response=client.post(
            "/predict",
            json=invalid_data
        )
    assert response.status_code==422

def test_model_loaded_once():
    with patch(
        "src.api.joblib.load",
        wraps=__import__("joblib").load
    ) as mock_load:
        with TestClient(app) as client:
            for _ in range(5):
                response=client.post("/predict",json=valid_data)
                assert response.status_code==200
        assert mock_load.call_count==1


def test_negative_income():
    invalid_data=valid_data.copy()
    invalid_data["MedInc"]=-5
    with TestClient(app) as client:
        response=client.post("/predict",json=invalid_data)
    assert response.status_code==422
    errors = response.json()["detail"]

    assert any(
        error["loc"] == ["body", "MedInc"]
        for error in errors
    )


def test_income_boundary():
    data = valid_data.copy()

    # Boundary value: valid
    data["MedInc"] = 0

    with TestClient(app) as client:
        response = client.post("/predict", json=data)

        assert response.status_code == 200

        # Just below boundary: invalid
        data["MedInc"] = -0.001

        response = client.post("/predict", json=data)

        assert response.status_code == 422

def test_prediction_failure():
    with TestClient(app) as client:
        with patch(
            "src.api.model.predict",
            side_effect=RuntimeError("Simulated model failure")
        ):
            data=valid_data.copy()
            # data["MedInc"] = -1
            response=client.post("/predict",json=data)
            assert response.status_code==500

def test_mock_prediction():
    with TestClient(app) as client:
        with patch(
            "src.api.model.predict",
            return_value=[2.5]
        ):
            data=valid_data.copy()
            response=client.post("/predict",json=data)
            assert response.status_code==200
            assert response.json()=={"predicted_price":250000.0,"currency":"USD"}

def test_prediction_error_logged(caplog):
    with TestClient(app) as client:
        with patch(
            "src.api.model.predict",
            side_effect=RuntimeError("Simulated Failure")
        ):
            with caplog.at_level(
                logging.ERROR,
                logger="src.api"
            ):
                response=client.post(
                    "/predict",
                    json=valid_data
                )
    assert response.status_code==500
    assert "Model Prediction Failed" in caplog.text
    assert any(
    record.levelno == logging.ERROR
    and record.getMessage() == "Model Prediction Failed"
    for record in caplog.records
)


def test_model_unavailable():

    with TestClient(app) as client:

        with patch("src.api.model", None):

            response = client.post(
                "/predict",
                json=valid_data
            )

    assert response.status_code == 503

    assert response.json()["detail"] == "Model not Found"


def test_model_loading_failure(caplog):
    with patch(
        "src.api.joblib.load",
        side_effect=FileNotFoundError("Model file missing")
    ):
        with pytest.raises(FileNotFoundError):
            with TestClient(app):
                pass
    assert "Failed to load house price model" in caplog.text