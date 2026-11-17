import numpy as np
from sklearn.ensemble import RandomForestRegressor
from src.model_saving import save_model,load_model

def test_model_serialization(tmp_path):
    X = np.array([
        [1200, 2],
        [1500, 3],
        [1800, 3],
        [2000, 4]
    ])

    y = np.array([45, 60, 75, 90])

    model=RandomForestRegressor(
        n_estimators=10,
        random_state=42
    )
    model.fit(X,y)

    original_predictions=model.predict(X)
    model_path=tmp_path / "test_model.joblib"
    save_model(model,model_path)

    loaded_model=load_model(model_path)
    loaded_predictions=loaded_model.predict(X)

    np.testing.assert_allclose(
        original_predictions,loaded_predictions
    )