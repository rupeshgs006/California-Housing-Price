import joblib
from pathlib import Path

def save_model(model,file_path):
    path=Path(file_path)
    path.parent.mkdir(parents=True,exist_ok=True)
    joblib.dump(model,path)
    return path

def load_model(file_path):
    return joblib.load(file_path)

