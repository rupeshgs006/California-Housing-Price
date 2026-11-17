from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

def build_model():
    model=Pipeline([
        ("scaler",StandardScaler()),
        ("regressor",LinearRegression())
    ])
    return model

def train_model(model,xtrain,ytrain):
    model.fit(xtrain,ytrain)
    return model

import numpy as np
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score

def evaluate_model(model,xtest,ytest):
    predictions=model.predict(xtest)
    mae=mean_absolute_error(ytest,predictions)
    mse=mean_squared_error(ytest,predictions)
    rmse=np.sqrt(mse)
    r2=r2_score(ytest,predictions)
    results = {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2 Score": r2
    }

    return results

from sklearn.ensemble import RandomForestRegressor

def build_random_forest():
    model=RandomForestRegressor(
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        random_state=42,
        n_jobs=-1
    )
    return model

def compare_train_test_performance(model,xtrain,ytrain,xtest,ytest):
    train_predictions=model.predict(xtrain)
    test_predictions=model.predict(xtest)

    train_r2=r2_score(ytrain,train_predictions)
    test_r2=r2_score(ytest,test_predictions)
    return {
        "Training R2":train_r2,
        "Testing R2":test_r2
    }

def build_tuned_random_forest():

    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )

    return model

from sklearn.model_selection import cross_val_score
def perform_cross_validation(model,xtrain,ytrain):
    scores=cross_val_score(
        model,
        xtrain,
        ytrain,
        cv=5,
        scoring="r2",
        n_jobs=-1
    )
    return scores


from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor

def tune_random_forest(xtrain,ytrain):
    model=RandomForestRegressor(
        random_state=42,
        n_jobs=1
    )

    param_distributions={
        "n_estimators":[100,150,200],
        "max_depth":[None,10,15,20,25],
        "min_samples_split":[2,5,10],
        "min_samples_leaf":[1,2,4],
        "max_features":[1.0,"sqrt",0.8]
    }

    random_search=RandomizedSearchCV(
        estimator=model,
        param_distributions=param_distributions,
        n_iter=10,
        cv=5,
        scoring='r2',
        random_state=42,
        n_jobs=-1,
        verbose=1
    )

    random_search.fit(xtrain,ytrain)
    return random_search