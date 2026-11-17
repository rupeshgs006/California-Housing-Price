from prometheus_client import Counter,Histogram

PREDICTION_REQUESTS=Counter(
    "house_prediction_requests_total",
     "Total number of house price prediction attempts",
    ["outcome"]
)

PREDICTION_DURATION = Histogram(
    "house_prediction_duration_seconds",
    "Duration of Random Forest inference in seconds"
)