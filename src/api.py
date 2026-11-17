from fastapi import FastAPI
import logging
import time
from prometheus_client import make_asgi_app
from src.metrics import PREDICTION_DURATION,PREDICTION_REQUESTS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger=logging.getLogger(__name__)




from contextlib import asynccontextmanager
from pathlib import Path
import joblib
import pandas as pd
from fastapi import FastAPI,HTTPException
from pydantic import BaseModel,Field

MODEL_PATH=(
    Path(__file__).resolve().parent.parent
    / "artifacts"
    / "house_price_rf.joblib"
)
model=None

@asynccontextmanager
async def lifespan(app:FastAPI):
    global model
    logger.info("Loading House price model")
    try:
        model=joblib.load(MODEL_PATH)
    except Exception:
        logger.exception("Failed to load house price model")
        raise

    logger.info("Model Loaded Successfully")
    try:
        yield
    finally:
        logger.info("Application shutting down")
        model=None


app=FastAPI(
    title="House Price Prediction API",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/")
def home():
    return {"message":"House Price API is running"}

@app.get("/health")
def health():
    return {"status":"healthy"}


class HouseFeatures(BaseModel):
    MedInc: float = Field(ge=0)
    HouseAge: float = Field(ge=0)
    AveRooms: float = Field(gt=0)
    AveBedrms: float = Field(ge=0)
    Population: float = Field(ge=0)
    AveOccup: float = Field(gt=0)
    Latitude: float = Field(ge=-90, le=90)
    Longitude: float = Field(ge=-180, le=180)

app.mount("/metrics",make_asgi_app())

@app.post("/predict")
def predict(features:HouseFeatures):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not Found"
        )
    logger.info("Prediction request received")
    input_data=pd.DataFrame(
        [features.model_dump()]
    )
    start_time=time.perf_counter()
    try:
        prediction=model.predict(input_data)[0]
    except Exception:
        PREDICTION_REQUESTS.labels(outcome="error").inc()
        logger.exception("Model Prediction Failed")
        raise HTTPException(
            status_code=500,
            detail="Prediction service encountered an error"
        )
    else:
        PREDICTION_REQUESTS.labels(outcome="success").inc()
    finally:
        
        interference_time=time.perf_counter()-start_time
        PREDICTION_DURATION.observe(interference_time)
        logger.info("Model inference duration : %.4f seconds",interference_time)
        logger.info("Prediction completed successfully")  
    return {
        "predicted_price":round(float(prediction)*100000,2),
        "currency":"USD"
    }



