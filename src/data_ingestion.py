import pandas as pd
from pathlib import Path
def load_data(file_path):
    # try:
    #     data=pd.read_csv(file_path)
    #     return data
    # except FileNotFoundError:
    #     print("File Not Found")

    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at :{file_path}"
        )

    data = pd.read_csv(file_path)
    return data

def validation_data(data,required_columns,target_columns):
    
    missing_columns=required_columns-set(data.columns)
    print("===>>>",missing_columns)
    if missing_columns:
        raise ValueError(
            f"missing required : {missing_columns}"
        )
  
    return data

def handle_missing_target(data,target_column):
    missing_values=data[target_column].isnull().sum()
    print(f"missing target values:{missing_values}")
    cleaned_data=data.dropna(subset=[target_column])
    return cleaned_data


def handle_missing_features(data, target_column):

    data = data.copy()

    feature_columns = data.drop(
        columns=[target_column]
    ).select_dtypes(include="number")

    medians = feature_columns.median()

    missing_median_columns = medians[
        medians.isnull()
    ].index.tolist()

    if missing_median_columns:

        print("Removing empty columns:", missing_median_columns)

        data = data.drop(columns=missing_median_columns)

        feature_columns = data.drop(
            columns=[target_column]
        ).select_dtypes(include="number")

        medians = feature_columns.median()

    data[feature_columns.columns] = feature_columns.fillna(medians)

    return data



