import pandas as pd
from src.data_ingestion import load_data,validation_data,handle_missing_target,handle_missing_features
import pytest
def test_load_data():
    data=load_data("test.csv")
    assert isinstance(data,pd.DataFrame)

def test_missing_file():

    with pytest.raises(FileNotFoundError):
        load_data("missing.csv")

def test_row_count():
    data=load_data("test.csv")
    assert len(data)==3

required_columns = {
    "area", "bedrooms", "bathrooms", "age", "price"
}


def test_missing_required_column():

    data = load_data("test.csv")

    data = data.drop(columns=["price"])

    with pytest.raises(ValueError):
        validation_data(
            data,
            required_columns,
            "price"
        )


def test_missing_target():

    data = pd.DataFrame({
        "area": [1200, 1500, 1800, 1400],
        "price": [4500000, None, 7500000, 5000000]
    })

    cleaned_data = handle_missing_target(data, "price")

    assert len(cleaned_data) == 3
    assert cleaned_data["price"].isnull().sum() == 0

def test_missing_feature_values():
    data = pd.DataFrame({
        "area": [1200, 1500, 1800, None],
        "bedrooms": [2, 3, 3, 2],
        "price": [4500000, 6000000, 7500000, 5000000]
    })

    cleaned_data = handle_missing_features(data, "price")

    assert cleaned_data["area"].isnull().sum() == 0
    assert cleaned_data["area"].iloc[3] == 1500
    assert len(cleaned_data) == 4

def test_entirely_missing_feature():
    data=pd.DataFrame({
        "area": [None, None, None],
        "bedrooms": [2, 3, 2],
        "price": [4500000, 6000000, 7500000]
    })

    cleaned_data=handle_missing_features(data,"price")
    assert "area" in cleaned_data.columns.tolist()
    assert "price" in cleaned_data.columns
    assert len(cleaned_data)==3