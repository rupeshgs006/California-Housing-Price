# California Housing Price — A Beginner's End-to-End MLOps Project

**FastAPI + Docker + Docker Compose + Nginx + Locust, built step by step.**

This repository takes one small machine-learning problem (predicting California house prices) and walks it all the way from a Python script to a **tested, containerized, load-balanced prediction service**. Every stage is explained in plain language, so you can read it top to bottom, run each step yourself, and understand *why* it exists.

> **Who is this for?** Anyone who has trained a model in a notebook and wondered, "What happens next?" This is intentionally the **most basic MLOps repository**: no cloud account, no Kubernetes, no paid tools. If you are new to MLOps, start here.

**GitHub repository:** [rupeshgs006/California-Housing-Price](https://github.com/rupeshgs006/California-Housing-Price)  
**Last updated:** 17 November 2026

---

## Project status (my progress so far)

| Stage | Status | Notes |
| --- | :---: | --- |
| Data loading, exploration, preparation | ✅ Done | `src/data_loader.py`, `src/data_preparation.py`, `src/explore_dataset.py` |
| Model training and evaluation (Linear Regression baseline, Random Forest) | ✅ Done | R² ≈ 0.80 on the hold-out set |
| Experiment tracking with MLflow | ✅ Done (helper code) | `src/experiment_tracking.py`; calls are commented out in the training script, see Section 6 |
| Model saving with Joblib | ✅ Done | `src/model_saving.py` |
| FastAPI inference service with validation and logging | ✅ Done | `src/api.py` |
| Automated tests (20 tests) | ✅ Done | `tests/` |
| Docker image | ✅ Done | `Dockerfile`, tag `house-price-api:v3` |
| Docker Compose, single API | ✅ Done | `compose.yaml` |
| Nginx load balancer with 3 API replicas | ✅ Done (local) | `compose.scale.yaml` + `nginx/nginx.conf` |
| Load testing with Locust | ✅ Done (local) | `load_test.py`, results in Section 12 |
| Prometheus metrics in the API | ✅ Done (in code) | `src/metrics.py`, endpoint `/metrics/` |
| Clean repository published on GitHub | ✅ Done | Fresh Git history, model kept out of Git, Compose and requirements fixed |


> **Honest scope:** everything marked "local" was run on one development laptop. This is a learning project, not a production housing-valuation service.

---

## Progress timeline

How this project was actually built, in order. Dates are when each step was worked on.

| Date (2026) | What I built or learned | Result |
| --- | --- | --- |
| Nov 1-2 | Set up the project and wrote data ingestion with tests: load a CSV, check required columns, drop rows with a missing target, fill missing features with medians | First pytest tests passing |
| Nov 2 | Loaded California Housing (20,640 rows, 9 columns, no missing values), checked correlations (`MedInc` is the strongest at about 0.69), split 80/20 (16,512 train / 4,128 test) | Linear baseline R² 0.576, Random Forest R² 0.805 |
| Nov 5 | Compared train vs test score (0.97 vs 0.80), hand-tuned the forest, ran 5-fold cross-validation and `RandomizedSearchCV`, logged runs to MLflow, saved the model with Joblib | Chose the original Random Forest; model-saving test added |
| Nov 9 | Built the FastAPI app: model loaded at startup (lifespan), Pydantic validation, `/predict`; tested with `TestClient`; studied ASGI | API tests passing |
| Nov 10 | Studied sync vs async; added logging and error handling (500/503) with failure tests; built Docker images (v1 1.95 GB, v2 869 MB); volumes vs bind mounts; environment variables; Docker Compose; studied CI/CD concepts; first GitHub push (hit large-file and history problems) | 20 tests passing, API running in Docker |
| Nov 12 | Added Prometheus metrics and inference timing; first Locust load tests | Saw inference time rise under load |
| Nov 14 | Ran three API replicas behind Nginx; compared 1 vs 3 replicas with Locust and `docker stats`; started Kubernetes concepts | 13.32 req/s to 62.45 req/s |
| Nov 14-17 | Made a clean copy with fresh Git history, kept the 138 MB model out of Git, fixed Compose and requirements files, wrote this README | Repository published |
| Nov 17 | Started hands-on Kubernetes (Lesson 1: local cluster and first Pod) | In progress |

> CI/CD and most of Kubernetes are still at the "understood the idea" stage. This table does not claim more than that.

---

## Contents

> New here? Read the [progress timeline](#progress-timeline) first to see how the project was built step by step.

1. [Project overview](#1-project-overview)
2. [Concepts you need first (mini glossary)](#2-concepts-you-need-first-mini-glossary)
3. [System architecture](#3-system-architecture)
4. [Machine-learning workflow](#4-machine-learning-workflow)
5. [Dataset and features](#5-dataset-and-features)
6. [Model training, evaluation and experiment tracking](#6-model-training-evaluation-and-experiment-tracking)
7. [FastAPI inference service](#7-fastapi-inference-service)
8. [Automated testing](#8-automated-testing)
9. [Docker: packaging the API](#9-docker-packaging-the-api)
10. [Docker Compose: single-service deployment](#10-docker-compose-single-service-deployment)
11. [Nginx reverse proxy and three API replicas](#11-nginx-reverse-proxy-and-three-api-replicas)
12. [Locust load testing and benchmark results](#12-locust-load-testing-and-benchmark-results)
13. [CPU, memory and scaling trade-offs](#13-cpu-memory-and-scaling-trade-offs)
14. [Observability with Prometheus metrics](#14-observability-with-prometheus-metrics)
15. [Repository structure](#15-repository-structure)
16. [Setup from a fresh clone (step by step)](#16-setup-from-a-fresh-clone-step-by-step)
17. [Commands reference](#17-commands-reference)
18. [Troubleshooting](#18-troubleshooting)
19. [Engineering lessons and limitations](#19-engineering-lessons-and-limitations)
20. [Practice exercises](#20-practice-exercises)
21. [Roadmap](#21-roadmap)
22. [Interview talking points](#22-interview-talking-points)

---

## 1. Project overview

Most machine-learning tutorials stop after `model.fit()` and a printed accuracy number. A usable ML system also needs a reliable way to **load** the model, **accept validated requests**, **return predictions**, **fail safely**, **run the same way on every computer**, and **handle many users at once**.

This project follows that whole path:

```text
 Data exploration & preparation
            |
            v
 Model training & evaluation  (Linear Regression baseline, Random Forest)
            |
            v
 Experiment tracking (MLflow)
            |
            v
 Saved model file (.joblib)
            |
            v
 FastAPI service  ->  POST /predict
            |
            v
 Automated tests (pytest)
            |
            v
 Docker image
            |
            v
 Docker Compose  +  Nginx
            |
            v
 3 identical API replicas
            |
            v
 Load testing (Locust) + resource measurements
```

### What I built and practiced

| Area | What was done |
| --- | --- |
| Machine learning | Loaded California Housing data, split it, trained a baseline and a Random Forest, evaluated with MAE, RMSE and R² |
| Experiment tracking | Wrote an MLflow helper that logs parameters, metrics and the trained model |
| Model persistence | Saved the fitted model with Joblib and loaded it back for serving |
| API | Built a FastAPI app with Pydantic validation, `/health`, `/predict`, startup model loading, error handling and logging |
| Testing | Wrote 20 pytest tests for the API, data ingestion and model saving |
| Containerization | Built a slim Docker image (`python:3.14-slim`) with a separate, minimal requirements file |
| Orchestration | Used Docker Compose for one API container, then for three API containers plus Nginx |
| Networking | Routed traffic through Nginx to the APIs using Docker service names |
| Performance | Load-tested with Locust, compared 1 API vs 3 APIs, watched CPU and memory with `docker stats` |
| Metrics | Added a Prometheus counter and histogram around model inference |

---

## 2. Concepts you need first (mini glossary)

If a word below is new to you, read this section once. Each term appears later in the project.

| Term | Plain-English meaning |
| --- | --- |
| **Regression** | Predicting a *number* (here: a house price), as opposed to a category |
| **Training / inference** | *Training* = the model learns from data. *Inference* = the trained model answers new questions |
| **Hold-out (test) set** | Data the model never saw during training, used to check how well it generalizes |
| **Model artifact** | The saved file produced by training (here `house_price_rf.joblib`) |
| **API** | A program other programs talk to over the network. Ours receives house features and returns a price |
| **HTTP / JSON** | The language of web APIs. We send JSON text such as `{"MedInc": 8.3}` and receive JSON back |
| **Endpoint** | One URL path of an API, such as `/health` or `/predict` |
| **Pydantic validation** | Automatic checking that incoming data has the right fields, types and ranges |
| **Port** | A numbered "door" on a computer. Our API listens on port 8000 |
| **Docker image** | A frozen package containing code + Python + libraries. A recipe you can run anywhere |
| **Docker container** | A *running* copy of an image. One image can start many containers |
| **Volume / bind mount** | A way to share a folder from your computer into a container |
| **Docker Compose** | A YAML file that describes several containers and starts them together |
| **Reverse proxy** | A server (Nginx) that receives requests and forwards them to other servers behind it |
| **Load balancer** | A reverse proxy that spreads requests across several identical servers |
| **Round-robin** | Sending request 1 to server A, 2 to B, 3 to C, 4 to A again, and so on |
| **Replica** | One of several identical copies of the same service |
| **Throughput** | How many requests are completed per second (req/s) |
| **Latency (P50, P95, P99)** | How long a request takes. P50 = the typical request; P95 = 95% of requests were faster than this; P99 is the slow tail |
| **Metric** | A number the app reports about itself (requests handled, time spent) |

---

## 3. System architecture

### Single-instance architecture

```text
 Client / PowerShell / Swagger UI
              |
              |  HTTP  :8000
              v
     FastAPI + Uvicorn
              |
              |  Pydantic checks the input
              v
   Random Forest (loaded once at startup)
              |
              v
        JSON prediction
```

### Load-balanced architecture (current milestone)

```text
                      Client / Locust
                            |
                     localhost:8080
                            |
                            v
                  +------------------+
                  |  Nginx container |
                  |  reverse proxy   |
                  |  round-robin     |
                  +------------------+
                      /     |     \
                     v      v      v
               +------+ +------+ +------+
               | api1 | | api2 | | api3 |
               | :8000| | :8000| | :8000|
               +------+ +------+ +------+
                  |        |        |
               RF model RF model RF model
                  ^        ^        ^
                  |        |        |
             ./artifacts folder mounted read-only into each

        All four containers share one Docker network.
        Only Nginx publishes a port to your computer (8080).
```

Key ideas:

- All three APIs run the **same image**, but each is a separate process with its **own copy** of the model in memory.
- You talk only to Nginx (`localhost:8080`). It chooses one API for each request.
- This runs on **one computer**. It shows replication and traffic distribution, but it does **not** protect you if the computer itself fails.

---

## 4. Machine-learning workflow

The model is produced in ten steps. Steps 1–6 live in `src/explore_dataset.py`, which calls the other modules.

| # | Step | Where in the code | Why it matters |
| --- | --- | --- | --- |
| 1 | Download and explore the data | `data_loader.py`, `explore_dataset.py` | Know your data before modelling |
| 2 | Split into train (80%) and test (20%) | `data_preparation.py` (`random_state=42`) | Test on data the model has never seen; the fixed seed makes the split repeatable |
| 3 | Train a baseline (scaler + Linear Regression) | `model_training.py` | A simple model to beat |
| 4 | Train a Random Forest (100 trees) | `model_training.py` | A stronger, non-linear model |
| 5 | Evaluate with MAE, RMSE, R² | `evaluate_model()` | Measure quality with numbers, not feelings |
| 6 | Save the Random Forest with Joblib | `model_saving.py` | Separate *training* from *serving* |
| 7 | Serve the model with FastAPI | `api.py` | Make predictions available over HTTP |
| 8 | Test everything | `tests/` | Catch bugs before deployment |
| 9 | Package with Docker, run with Compose and Nginx | `Dockerfile`, `compose*.yaml`, `nginx/` | Same behavior everywhere; handle more traffic |
| 10 | Load-test and measure | `load_test.py` | Replace guessing with measurements |

---

## 5. Dataset and features

**Dataset:** the California Housing dataset from scikit-learn (`fetch_california_housing`). It is downloaded automatically the first time you run training (an internet connection is needed once; scikit-learn caches it afterwards).

**Task:** regression. Estimate the median house value of a census block group from eight numbers.

| Feature | Meaning |
| --- | --- |
| `MedInc` | Median income in the block group |
| `HouseAge` | Median house age |
| `AveRooms` | Average rooms per household |
| `AveBedrms` | Average bedrooms per household |
| `Population` | Block group population |
| `AveOccup` | Average occupants per household |
| `Latitude` | Geographic latitude |
| `Longitude` | Geographic longitude |

**Target:** `MedHouseVal`, expressed in units of **USD 100,000**. A raw prediction of `2.5` means roughly **USD 250,000**. The API multiplies the raw value by 100,000 and returns the result in dollars.

> This is a learning dataset. It is not an independently validated real-estate appraisal tool.

---

## 6. Model training, evaluation and experiment tracking

### 6.1 Run the training

From the **repository root**:

```powershell
python src/explore_dataset.py
```

What the script does, in order:

1. Downloads the dataset and prints its shape, columns, data types, missing values and target statistics.
2. Splits the data 80/20.
3. Trains and evaluates the **baseline** (StandardScaler + Linear Regression).
4. Trains and evaluates the **Random Forest** (100 trees).
5. Saves the Random Forest to `artifacts/house_price_rf.joblib` (about 138 MB).
6. Reloads the file and predicts one test row as a quick sanity check.

> Run it from the repository root. The save path `artifacts/...` is relative to the folder you run the command in, and the API looks for the model in `<repo>/artifacts/`.

> ✅ **Checkpoint:** `Test-Path .\artifacts\house_price_rf.joblib` prints `True`.

### 6.2 What the metrics mean

| Metric | Recorded value | Meaning |
| --- | ---: | --- |
| MAE (Mean Absolute Error) | ≈ 0.3277 | Average size of the mistake, in units of $100,000 (about $32,800) |
| RMSE (Root Mean Squared Error) | ≈ 0.506 | Like MAE, but big mistakes count more |
| R² | ≈ 0.8046 | Share of the variation in prices the model explains (about 80%) |

These are the values recorded during development. Because the split and model use fixed seeds, your run should be very close, but check the numbers printed on your machine.

### 6.3 Model experiments: what I tried and what I learned

I did not stop at the first model. Each row below is a real run on the same 80/20 split (`random_state=42`):

| Model | MAE | RMSE | R² (test) | Notes |
| --- | ---: | ---: | ---: | --- |
| Linear Regression (with scaling), baseline | 0.5332 | 0.7456 | 0.5758 | Too simple for this data |
| Random Forest, 100 trees, default depth | 0.3277 | 0.5060 | 0.8046 | Training R² was 0.9735, so it fits the training data much better than new data |
| Hand-tuned Random Forest (max depth 15, min split 5, min leaf 2) | 0.3306 | 0.5087 | 0.8025 | Slightly worse than the default |
| Best of `RandomizedSearchCV` (150 trees, depth 25, min split 5, min leaf 2, `sqrt` features) | 0.3327 | 0.5021 | 0.8076 | Cross-validation R² 0.8109; best R² and RMSE, but higher MAE |

5-fold cross-validation of the default Random Forest gave a mean R² of **0.8044** with a standard deviation of **0.0054**, so the 0.8046 test score is not a lucky split.

**Decision:** I kept the original Random Forest because it had the lowest MAE (0.3277). The searched model is slightly better on R² (0.8076 vs 0.8046) and RMSE, but the differences are small, and the original is simpler. Which one is "best" depends on which error you care about.

**What I learned:**

- A big gap between training R² (0.97) and test R² (0.80) means overfitting, and tuning only narrowed it a little.
- A baseline matters: without the 0.5758 linear score, 0.8046 would mean nothing.
- Always say *which metric* you used to choose a model.

The tuning, cross-validation and search code lives in `src/model_training.py`. In `explore_dataset.py` those blocks are **commented out** so the default run is quick; uncomment them one at a time to reproduce the table.

### 6.4 Experiment tracking with MLflow

Without tracking, experiments get lost: *Which settings gave that score? Which file was the good model?* MLflow records, for each run:

- **Parameters** (e.g. number of trees, depth)
- **Metrics** (MAE, RMSE, R²)
- **The trained model**

`src/experiment_tracking.py` provides `log_experiment(model, params, metrics, run_name)`, which stores a run under the experiment named **"House Price Prediction"**. To use it, uncomment the MLflow section in `explore_dataset.py`, run the script, and then open the dashboard:

```powershell
mlflow ui
```

Then browse to <http://127.0.0.1:5000>.

> MLflow creates local files (`mlruns/`, `mlflow.db`). They are development leftovers, not source code, and are listed in `.gitignore`.

Four things that are easy to confuse:

| Thing | What it is |
| --- | --- |
| Source code | *How* the model is trained |
| Experiment record | *What settings and scores* a particular run had |
| Model artifact | The *file* chosen for serving |
| Docker image | The *runtime environment* containing the API |

### 6.5 Saving the model safely

```python
import joblib
joblib.dump(model, "artifacts/house_price_rf.joblib")   # save
loaded_model = joblib.load("artifacts/house_price_rf.joblib")  # load
```

Joblib files are pickle-based, so **only load model files you trust**: loading a malicious file can run code. Also load it with a compatible Python and scikit-learn version.

---

## 7. FastAPI inference service

`src/api.py` loads the model **once** when the server starts (inside the FastAPI *lifespan* function) and reuses it for every request. Reloading a 138 MB file for each request would be very slow.

### 7.1 Endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/` | Returns `{"message": "House Price API is running"}` |
| GET | `/health` | Returns `{"status": "healthy"}` |
| POST | `/predict` | Validates 8 features, returns the predicted price |
| GET | `/metrics/` | Prometheus metrics (use the trailing slash; `/metrics` answers with a redirect) |
| GET | `/docs` | Interactive Swagger UI generated by FastAPI |

### 7.2 Input validation

Pydantic rejects bad input **before** it reaches the model:

| Field | Rule |
| --- | --- |
| `MedInc` | ≥ 0 |
| `HouseAge` | ≥ 0 |
| `AveRooms` | > 0 |
| `AveBedrms` | ≥ 0 |
| `Population` | ≥ 0 |
| `AveOccup` | > 0 |
| `Latitude` | between −90 and 90 |
| `Longitude` | between −180 and 180 |

### 7.3 Status codes

| Code | When |
| --- | --- |
| 200 | Prediction succeeded |
| 422 | Missing or invalid input (for example a negative income) |
| 503 | The model is not loaded |
| 500 | The model raised an error while predicting |

### 7.4 Try it

Start the server (model must exist, see Section 6.1):

```powershell
python -m uvicorn src.api:app --host 127.0.0.1 --port 8000
```

Open <http://127.0.0.1:8000/docs> and use **Try it out** on `POST /predict`, or use PowerShell:

```powershell
$body = @{
    MedInc     = 8.3252
    HouseAge   = 41.0
    AveRooms   = 6.984127
    AveBedrms  = 1.02381
    Population = 322.0
    AveOccup   = 2.555556
    Latitude   = 37.88
    Longitude  = -122.23
} | ConvertTo-Json

Invoke-RestMethod -Uri http://127.0.0.1:8000/predict -Method Post `
    -ContentType "application/json" -Body $body
```

Or with curl (macOS/Linux/Git Bash):

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"MedInc":8.3252,"HouseAge":41,"AveRooms":6.984127,"AveBedrms":1.02381,"Population":322,"AveOccup":2.555556,"Latitude":37.88,"Longitude":-122.23}'
```

**Response shape:**

```json
{
  "predicted_price": 452600.0,
  "currency": "USD"
}
```

The number depends on your trained model. The shape (`predicted_price`, `currency`) is fixed by the code.

> ✅ **Checkpoint:** `Invoke-RestMethod http://127.0.0.1:8000/health` returns `status : healthy`, and `/predict` returns a dollar amount.

### 7.5 Logging

The API logs: model loading, every prediction request, the inference time (measured with `time.perf_counter()`), and any prediction failure with its stack trace. Logs are your first tool when something goes wrong.

---

## 8. Automated testing

A test is a small program that checks another program. Running tests after every change tells you immediately if you broke something.

| File | Tests | What it checks |
| --- | ---: | --- |
| `tests/test_api.py` | 12 | Health route, valid prediction, missing/invalid input (422), model loaded only once, error handling (500, 503), error logging, boundary values |
| `tests/test_data_ingestion.py` | 7 | Loading a CSV, missing file, missing required columns, missing target values, filling missing features with medians, an entirely empty column |
| `tests/test_model_saving.py` | 1 | A model gives identical predictions after being saved and reloaded |
| **Total** | **20** | |

Run from the repository root:

```powershell
python -m pytest -v
```

Things worth knowing:

- The API tests use FastAPI's `TestClient`, so you do **not** need to start Uvicorn.
- The API tests **need the trained model** because the app loads it at startup. Before training, **9 tests pass and 11 fail** with `FileNotFoundError`. After training, all **20 pass**.
- `test_data_ingestion.py` opens `test.csv` by relative path, so run pytest from the repository root. `pytest.ini` already points pytest at `tests/`.
- Several API tests use `unittest.mock.patch` to *simulate* failures (a model crash, a missing model) without breaking the real model. This is a core testing technique.
- If pytest hangs on a machine that also has Locust installed, disable Locust's pytest plugin: `python -m pytest -v -p no:locust`.

> ✅ **Checkpoint:** `20 passed`.

---

## 9. Docker: packaging the API

Your code works on your laptop because of a specific Python version, specific libraries and specific files. **Docker packages all of that** so the service behaves the same on any machine.

### 9.1 The Dockerfile, line by line

```dockerfile
FROM python:3.14-slim                       # start from a small official Python image
WORKDIR /app                                # work inside /app in the container
COPY requirements-api.txt .                 # copy only the dependency list first
RUN python -m pip install --no-cache-dir -r requirements-api.txt   # install runtime libraries
COPY src/ ./src/                            # copy the application code
RUN mkdir -p /app/artifacts                 # create the folder where the model will be mounted
EXPOSE 8000                                 # the API listens on port 8000
CMD ["python", "-m", "uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
```

Why copy `requirements-api.txt` before the code? Docker caches each step. If only your code changes, Docker reuses the slow "install libraries" step and rebuilds in seconds.

### 9.2 Why a separate `requirements-api.txt`?

| File | Used for | Contains |
| --- | --- | --- |
| `requirements.txt` | Your development machine | Everything: MLflow, Locust, pytest, plotting libraries... |
| `requirements-api.txt` | The Docker image | Only what serving needs: FastAPI, Uvicorn, Pydantic, Joblib, pandas, NumPy, scikit-learn, prometheus_client |

A smaller image is faster to build and download and has fewer things that can break or be attacked. During development an early image measured **1.95 GB** and the slimmed image **869 MB** (local measurements).

### 9.3 The model is *not* inside the image

The model file is ~138 MB and GitHub refuses ordinary files above 100 MB, so it is kept out of Git. `.dockerignore` also excludes `artifacts/`, which means the model is not copied into the image either. Instead, Compose **mounts** your local `artifacts/` folder into the container at `/app/artifacts` (read-only), which is exactly where `api.py` looks for it.

```text
your computer:   ./artifacts/house_price_rf.joblib
                          |  (read-only bind mount)
                          v
the container:   /app/artifacts/house_price_rf.joblib
```

> An earlier version of this project (`v2`) copied the model into the image. That works but needs the model file when you *build*. The mount approach used now keeps the image small and the repo clean, but the model must exist when the container *starts*.

### 9.4 Build the image

```powershell
docker build -t house-price-api:v3 .
docker images
```

> ✅ **Checkpoint:** `house-price-api   v3` appears in `docker images`.

---

## 10. Docker Compose: single-service deployment

Typing long `docker run` commands is error-prone. Compose stores the settings in `compose.yaml`:

```yaml
services:
  api:
    image: house-price-api:v3
    build: .
    ports:
      - "127.0.0.1:8000:8000"
    volumes:
      - ./artifacts:/app/artifacts:ro
    restart: unless-stopped
```

| Line | Meaning |
| --- | --- |
| `image` / `build` | Name of the image, and where to build it if missing |
| `ports: "127.0.0.1:8000:8000"` | Computer port 8000 → container port 8000, reachable **only from your own computer** |
| `volumes: ./artifacts:/app/artifacts:ro` | Share the model folder, read-only |
| `restart: unless-stopped` | Restart the container if it crashes (does not help if the whole computer dies) |

```powershell
docker compose up -d --build
docker compose ps
Invoke-RestMethod http://127.0.0.1:8000/health
docker compose logs api
docker compose down
```

> ✅ **Checkpoint:** the logs contain `Model Loaded Successfully`, and `/health` returns `healthy`.

---

## 11. Nginx reverse proxy and three API replicas

### 11.1 Why add load balancing?

When many users send requests at once, one API process becomes a bottleneck and responses slow down. The idea: run **several identical APIs** and put **Nginx** in front, so users see a single address while the work is shared.

### 11.2 The Nginx configuration (`nginx/nginx.conf`)

```nginx
events {}

http {
    upstream house_price_backend {   # a named group of backend servers
        server api1:8000;            # api1, api2, api3 are Compose service names
        server api2:8000;
        server api3:8000;
    }

    server {
        listen 80;                   # Nginx listens on port 80 inside its container

        location / {                 # every URL path...
            proxy_pass http://house_price_backend;   # ...is forwarded to the group

            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }
    }
}
```

By default Nginx uses **round-robin**: request 1 → api1, request 2 → api2, request 3 → api3, request 4 → api1... A single prediction runs on **one** replica, not on all three. The `proxy_set_header` lines pass along details about the original client.

Docker Compose provides a small internal network in which the names `api1`, `api2` and `api3` resolve to the right containers.

### 11.3 The Compose file (`compose.scale.yaml`)

```yaml
services:
  api1:
    image: house-price-api:v3
    volumes:
      - ./artifacts:/app/artifacts:ro
    expose:
      - "8000"
  api2:   # same as api1
  api3:   # same as api1

  nginx:
    image: nginx:alpine
    ports:
      - "127.0.0.1:8080:80"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf:ro
    depends_on:
      - api1
      - api2
      - api3
```

(`api2` and `api3` repeat `api1`'s settings in the real file.)

| Setting | Meaning |
| --- | --- |
| `expose: "8000"` | The API port is reachable **inside** the Docker network but **not** published to your computer |
| `ports: "127.0.0.1:8080:80"` | Only Nginx is reachable from your computer, on port 8080 |
| `./nginx/nginx.conf:/etc/nginx/nginx.conf:ro` | Gives Nginx *our* config instead of its default welcome page |
| `depends_on` | Start the APIs before Nginx. This only controls *start order*, not whether the model has finished loading |

### 11.4 Run it

```powershell
docker build -t house-price-api:v3 .
docker compose -f compose.scale.yaml up -d
docker compose -f compose.scale.yaml ps
Invoke-RestMethod http://127.0.0.1:8080/health
```

You should see four containers: `api1`, `api2`, `api3` (port `8000/tcp`) and `nginx` (`127.0.0.1:8080->80/tcp`). Their full names start with your folder name, for example `california-housing-price-api1-1`.

If you get `502 Bad Gateway` right after starting, wait a few seconds: each API needs time to load the model.

### 11.5 Prove that traffic is shared

```powershell
1..30 | ForEach-Object { Invoke-RestMethod http://127.0.0.1:8080/health | Out-Null }
docker compose -f compose.scale.yaml logs --tail=40 api1 api2 api3
```

You should see `GET /health ... 200` lines in the logs of **all three** APIs.

Shut everything down:

```powershell
docker compose -f compose.scale.yaml down
```

> ✅ **Checkpoint:** all three API logs show requests, and `http://127.0.0.1:8080/docs` opens the Swagger page through Nginx.

> **Limit to remember:** all four containers live on one computer. This demonstrates replication, not protection against hardware failure.

---

## 12. Locust load testing and benchmark results

### 12.1 What Locust does

Locust simulates many users sending requests at the same time and reports how the service behaves. `load_test.py` defines one kind of user who repeatedly sends `POST /predict` with a valid housing payload and marks a response as failed if the status is not 200, the JSON is invalid, or `predicted_price`/`currency` is missing.

### 12.2 Run the tests

Start **one** stack at a time so the two runs do not compete for the same CPU.

```powershell
# Single API (docker compose up -d, running on :8000)
locust -f load_test.py --headless --host http://127.0.0.1:8000 -u 10 -r 2 -t 60s --csv single_api_control

# 3 APIs + Nginx (compose.scale.yaml running on :8080)
locust -f load_test.py --headless --host http://127.0.0.1:8080 -u 10 -r 2 -t 60s --csv load_balanced
```

| Flag | Meaning |
| --- | --- |
| `-u 10` | 10 simulated users |
| `-r 2` | Add 2 new users per second |
| `-t 60s` | Run for 60 seconds |
| `--csv name` | Save results to CSV files with this prefix |

> `load_test.py` also contains an event hook that stops the run after 10,000 `/predict` requests. It did not trigger in the 60-second runs below (the totals were far lower), but remove it if you run longer tests so it cannot end a run unexpectedly.

### 12.3 Recorded results

Recorded on the author's laptop, with Locust on the same machine, using the earlier `v2` image (model inside the image).

| Measurement | 1 API container | 3 APIs + Nginx |
| --- | ---: | ---: |
| Requests completed | 791 | 3,689 |
| Failures | 0 | 0 |
| Throughput | 13.32 req/s | 62.45 req/s |
| Mean latency | 716 ms | 154 ms |
| Median (P50) | 670 ms | 96 ms |
| P95 | 1,200 ms | 360 ms |
| P99 | 1,700 ms | 430 ms |
| Maximum | 2,141 ms | 704 ms |

### 12.4 How to read this

- Three replicas completed about **4.69×** as many requests per second (62.45 ÷ 13.32).
- The typical request (P50) dropped from 670 ms to 96 ms, and the slow tail (P95) from 1,200 ms to 360 ms.
- Neither run reported failed requests.

This is a real, measured difference **on this laptop**. It does **not** prove that three replicas always give 4.69× more speed. The test was not perfectly controlled:

- Locust and Docker shared one computer.
- The single-API run went straight to FastAPI, while the 3-replica run went through Nginx.
- Background activity was not controlled.

A stronger experiment would repeat each run several times and compare 1 backend vs 3 backends **behind the same Nginx**.

### 12.5 Lessons from the first (exploratory) load test

My first Locust test, against a single API, taught three things before the benchmark above:

- **Averages hide load effects.** With 5 requests, Prometheus showed an average inference time of about 49 ms. After 5,950 requests it was about 360 ms (2,140 s total / 5,950). The likely reason is that many concurrent requests compete for the same CPU inside one container; I did not isolate the exact cause.
- **Know what each tool counts.** Prometheus counts every prediction since the process started; Locust reports one test run. Their numbers do not have to match.
- **Check that the test did what you asked.** I asked for 10,000 requests but Locust reported 2,802 (P50 380 ms, P95 490 ms, P99 530 ms), so it was not a complete 10,000-request test. The custom stop hook in `load_test.py` was suspected, so the later benchmarks use fixed 60-second runs instead.

---

## 13. CPU, memory and scaling trade-offs

`docker stats` shows live CPU and memory use per container.

| Container | Idle CPU | CPU under load | Memory under load |
| --- | ---: | ---: | ---: |
| Single API (separate run) | — | 117.46% | 433.7 MiB |
| API replica 1 | 0.30% | 113.44% | 434.3 MiB |
| API replica 2 | 0.28% | 115.03% | 431.0 MiB |
| API replica 3 | 0.33% | 113.68% | 429.8 MiB |
| Nginx | 0.00% | 3.11% | 4.7 MiB |

- The three replicas used about **1.27 GiB** together; Nginx used only a few MiB.
- A CPU figure above 100% is normal in Docker: 100% equals one CPU core, so 115% means a bit more than one core was busy.
- Each API needs hundreds of MiB because each starts its own Python interpreter, imports NumPy and scikit-learn, and holds its **own** Random Forest in memory. The 138 MB file size is only part of the memory story.

**Lesson:** more replicas can raise throughput by using more CPU cores, but they cost more RAM and CPU. Replicas on one laptop do not create new hardware, so real capacity planning needs repeatable measurements.

---

## 14. Observability with Prometheus metrics

HTTP status codes tell you *whether* a request worked, not *how healthy the model is*. `src/metrics.py` defines two metrics:

```python
PREDICTION_REQUESTS = Counter(
    "house_prediction_requests_total",
    "Total number of house price prediction attempts",
    ["outcome"],
)

PREDICTION_DURATION = Histogram(
    "house_prediction_duration_seconds",
    "Duration of Random Forest inference in seconds",
)
```

| Metric type | What it records |
| --- | --- |
| **Counter** | A number that only goes up: attempts labeled `outcome="success"` or `outcome="error"` |
| **Histogram** | How long inference took, grouped into time buckets, plus a count and a sum |

View them at <http://127.0.0.1:8000/metrics/> after sending a few predictions.

Two details to remember:

- **HTTP latency vs inference time:** Locust measures the whole HTTP path (network, Nginx, FastAPI, model). The histogram measures only the `model.predict()` call.
- **Metrics are per process.** With three replicas, each keeps its own counters, so scraping one replica does **not** show all traffic. A real Prometheus setup scrapes `api1`, `api2` and `api3` separately.
- The benchmark in Section 12 used an older image built **before** the metrics code was added. An image built from the current source (`house-price-api:v3`) includes `prometheus_client` and exposes `/metrics/`.

A Prometheus server and Grafana dashboards are future work.

---

## 15. Repository structure

```text
California-Housing-Price/
|
|-- src/
|   |-- api.py                   # FastAPI app: routes, model loading, validation, logging
|   |-- metrics.py               # Prometheus counter and histogram
|   |-- data_loader.py           # Downloads the California Housing dataset
|   |-- data_ingestion.py        # CSV loading, validation, missing-value handling (practice module)
|   |-- data_preparation.py      # Train/test split
|   |-- data_analysis.py         # Correlation helper
|   |-- model_training.py        # Baseline, Random Forest, tuning, cross-validation, evaluation
|   |-- model_saving.py          # save_model / load_model with Joblib
|   |-- experiment_tracking.py   # MLflow logging helper
|   |-- explore_dataset.py       # Script that runs the whole training workflow
|   `-- environment_practice.py  # Scratch file: reading environment variables
|
|-- tests/
|   |-- test_api.py              # 12 tests (need the trained model)
|   |-- test_data_ingestion.py   # 7 tests (use test.csv)
|   `-- test_model_saving.py     # 1 test
|
|-- nginx/
|   `-- nginx.conf               # Load-balancer configuration
|
|-- artifacts/
|   `-- .gitkeep                 # Keeps the folder in Git; the model itself is NOT committed
|
|-- Dockerfile                   # Builds the inference image
|-- compose.yaml                 # One API container on port 8000
|-- compose.scale.yaml           # Three API containers + Nginx on port 8080
|-- load_test.py                 # Locust load test
|-- pytest.ini                   # Tells pytest where tests are
|-- requirements.txt             # Full development environment
|-- requirements-api.txt         # Minimal runtime dependencies for the image
|-- test.csv                     # Tiny fixture used by test_data_ingestion.py
|-- .gitignore
|-- .dockerignore
`-- README.md
```

Files that must **stay out** of Git (already in `.gitignore`): `venv/`, `.venv/`, `__pycache__/`, `.pytest_cache/`, `mlruns/`, `mlflow.db`, `artifacts/*` (except `.gitkeep`), `*.joblib`, `.env` files, and Locust CSV reports.

---

## 16. Setup from a fresh clone (step by step)

Commands are for **Windows PowerShell**. On macOS/Linux use `python3 -m venv .venv` and `source .venv/bin/activate`; everything else is identical.

### Step 0 — Install the tools

- Git
- Python 3.14 (the version the project was developed and Dockerized with; the tests also pass on Python 3.12 with the same package versions)
- Docker Desktop (with Compose)
- About 2 GB of free RAM for one API, around 4 GB if you run three

### Step 1 — Clone the repository

```powershell
git clone https://github.com/rupeshgs006/California-Housing-Price.git
cd California-Housing-Price
```

### Step 2 — Create a virtual environment and install dependencies

A virtual environment keeps this project's libraries separate from everything else on your computer.

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

> `requirements.txt` contains `pywin32`, which installs only on Windows. On macOS/Linux, delete that one line first.

### Step 3 — (Optional) See the tests fail first

```powershell
python -m pytest -q
```

Expected: **9 passed, 11 failed**. The failures are `FileNotFoundError` for the model. This shows *why* the model matters.

### Step 4 — Train and save the model

```powershell
python src/explore_dataset.py
```

> ✅ `artifacts/house_price_rf.joblib` now exists (~138 MB).

### Step 5 — Run all tests

```powershell
python -m pytest -v
```

> ✅ `20 passed`.

### Step 6 — Run the API directly

```powershell
python -m uvicorn src.api:app --host 127.0.0.1 --port 8000
```

Open <http://127.0.0.1:8000/docs> and send a prediction. Stop the server with `Ctrl + C`.

### Step 7 — Build the Docker image

```powershell
docker build -t house-price-api:v3 .
```

### Step 8 — Run one API container with Compose

```powershell
docker compose up -d
Invoke-RestMethod http://127.0.0.1:8000/health
docker compose down
```

### Step 9 — Run three APIs behind Nginx

```powershell
docker compose -f compose.scale.yaml up -d
Invoke-RestMethod http://127.0.0.1:8080/health
docker compose -f compose.scale.yaml logs --tail=40 api1 api2 api3
```

### Step 10 — Load-test and observe

```powershell
locust -f load_test.py --headless --host http://127.0.0.1:8080 -u 10 -r 2 -t 60s --csv load_balanced
docker stats
```

Finally, `docker compose -f compose.scale.yaml down`.

### About the model file

The trained model is **not in GitHub** (it is ~138 MB, and GitHub blocks ordinary files over 100 MB). Each person trains it once with Step 4. A future improvement is to publish a versioned model with a checksum so a download step can replace training. Until then, **a fresh clone cannot serve predictions without Step 4.**

> Adding `*.joblib` to `.gitignore` *after* committing a big file does not remove it from Git history, so keep the model out of Git from the start.

---

## 17. Commands reference

| Task | Command |
| --- | --- |
| Create environment | `py -3.14 -m venv .venv` then `.\.venv\Scripts\Activate.ps1` |
| Install dependencies | `python -m pip install -r requirements.txt` |
| Train and save model | `python src/explore_dataset.py` |
| Run tests | `python -m pytest -v` |
| Run API locally | `python -m uvicorn src.api:app --host 127.0.0.1 --port 8000` |
| Build image | `docker build -t house-price-api:v3 .` |
| Start single API | `docker compose up -d` |
| Single API logs | `docker compose logs api` |
| Stop single API | `docker compose down` |
| Start Nginx + 3 APIs | `docker compose -f compose.scale.yaml up -d` |
| Show 3-API stack status | `docker compose -f compose.scale.yaml ps` |
| Logs of the replicas | `docker compose -f compose.scale.yaml logs api1 api2 api3` |
| Validate a Compose file | `docker compose -f compose.scale.yaml config` |
| Stop 3-API stack | `docker compose -f compose.scale.yaml down` |
| CPU / memory | `docker stats` |
| MLflow dashboard | `mlflow ui` |
| Prometheus metrics | `Invoke-RestMethod http://127.0.0.1:8000/metrics/` |

Avoid running the single-API stack and the three-replica stack at the same time during benchmarks; they compete for the same CPU and make results unreliable.

---

## 18. Troubleshooting

| Symptom | Likely cause and fix |
| --- | --- |
| `FileNotFoundError` for `house_price_rf.joblib`, or 11 failing API tests | The model has not been trained. Run `python src/explore_dataset.py` from the repo root |
| `docker build` says `requirements-api.txt` not found | The Dockerfile copies this file. Make sure it exists in the repo root |
| Container is up but `/predict` returns 503 or the container exits | The model is not visible inside the container. Check the file exists on your computer and that the service mounts `./artifacts:/app/artifacts:ro`. Then read `docker compose logs` |
| Port 8080 shows the "Welcome to nginx!" page, or `/predict` gives 404/405 | Nginx is not using our config. In `compose.scale.yaml` the `nginx` service must mount `./nginx/nginx.conf:/etc/nginx/nginx.conf:ro` |
| `502 Bad Gateway` from Nginx | The APIs are still loading the model, or crashed. Wait a few seconds, then run `docker compose -f compose.scale.yaml ps` and read the logs |
| `/metrics` returns a redirect | Use `/metrics/` with the trailing slash |
| `pip install` fails on `pywin32` | It is Windows-only. Remove that line on macOS/Linux |
| `ModuleNotFoundError: No module named 'sklearn.utils'` | You are probably not inside the virtual environment. Check `python -c "import sys; print(sys.executable)"` |
| pytest hangs when Locust is installed | Run `python -m pytest -v -p no:locust` |
| `git push` rejected: file too large | A big file is in your Git history. Keep `artifacts/` and `*.joblib` out of Git from the start |
| `python src/explore_dataset.py` cannot download the dataset | The first run needs internet access to fetch the data |

---

## 19. Engineering lessons and limitations

### What this project teaches

1. **Serving is a different problem from training.** The model must be loaded safely, inputs validated, and behavior exposed through a clear API.
2. **An image is not a container.** One image can start many independent containers, each with its own memory.
3. **A reverse proxy gives users one address.** Nginx hides how many servers are behind it.
4. **Scaling has a cost.** Three replicas raised throughput in this experiment but used roughly three times the model memory.
5. **Measure, do not guess.** Throughput, latency percentiles, errors, CPU and memory together tell the story; one number does not.
6. **Reproducibility needs artifacts.** A repository is not reproducible if a critical file (the model) cannot be obtained or recreated.
7. **Tests protect you.** The suite catches broken validation, broken error handling and a model that silently reloads on every request.

### Known limitations

- Local educational system, not a production service.
- The three replicas share one host, so they do not survive that host failing.
- The benchmark compared a direct single API with an Nginx-fronted trio, and ran on one laptop.
- `depends_on` does not guarantee the model has finished loading.
- No public model download or checksum yet.
- Training runs through an exploration script (`explore_dataset.py`) rather than a dedicated, configurable training command.
- No CI, no Prometheus server or Grafana, no TLS, authentication or rate limiting, and no Kubernetes or cloud deployment.

---

## 20. Practice exercises

If you are learning from this repository, try these in order. Each one reinforces a section above.

1. **Break a test on purpose.** Change `Field(ge=0)` to `Field(ge=1)` for `MedInc` in `src/api.py`, run pytest, and read which tests fail and why. Undo the change.
2. **Add a field rule.** Make `HouseAge` reject values above 100. Add a matching test in `tests/test_api.py`.
3. **Compare models.** Uncomment the tuned Random Forest and cross-validation blocks in `explore_dataset.py`. Which gives the best R²?
4. **Track experiments.** Uncomment the MLflow section, run training twice with different settings, and compare the runs in `mlflow ui`.
5. **Add an endpoint.** Create `GET /model-info` returning the feature names. Add a test.
6. **Watch round-robin.** Send 30 predictions through port 8080 and count the log lines per replica.
7. **Kill a replica.** With the 3-API stack running, run `docker stop <api2 container>` and keep sending requests. What happens? (Nginx's defaults, plus a short delay before it notices, make this a useful experiment.)
8. **Re-run the benchmark properly.** Run each Locust test three times and average the results. Do the numbers stay close?
9. **Add a health check.** Add a Compose `healthcheck` for each API so Nginx only receives traffic when the API answers.
10. **Write a CI workflow.** Add a GitHub Actions job that installs dependencies, trains or downloads a model, and runs pytest.

---

## 21. Roadmap

### In progress: Kubernetes, one concept at a time

Docker runs your packaged application. Kubernetes decides where it runs across machines and works to keep the *desired state* (for example "three copies running"). The learning order is deliberately small:

1. Create a local cluster with Docker Desktop and understand "node" and "cluster".
2. Run **one** Pod with the API image and see where it starts.
3. Use a **Deployment** to keep that Pod alive.
4. Add **replicas** and a **Service** (the Kubernetes version of what Nginx does here).
5. Then readiness probes, rolling updates and autoscaling.

> **Watch out:** the first Kubernetes lesson uses the older `house-price-api:v2` image, which has the model baked in. The current `v3` image does **not** contain the model, so on Kubernetes you must either use an image that includes it or mount it with a volume. Decide this before Step 2.

### Next up

- **CI with GitHub Actions:** a `test` job, then a `build` job that waits for it (`needs: test`). The runner will **not** have the 138 MB model, so the workflow must train a small model or download one before the API tests run.
- **Artifact management:** publish a versioned model (GitHub Release, cloud storage or Hugging Face) with a checksum and a download step.
- **Dedicated training command:** a `train.py` with command-line options, replacing the exploration script.
- **Readiness checks:** a readiness endpoint and Compose `healthcheck`s so traffic only reaches APIs that loaded the model.
- **Monitoring:** Prometheus scraping each replica, plus Grafana dashboards and alerts on error rate.
- **Controlled benchmarks:** compare 1 vs 3 backends behind the same Nginx, repeated several times.

---

## 22. Interview talking points

Questions this project lets you answer with your own numbers. Use them as prompts, not scripts.

| Question | Short answer from this project |
| --- | --- |
| Why split into train and test sets? | To score the model on data it never saw; mine was 16,512 train / 4,128 test with a fixed seed |
| What does train R² 0.97 vs test R² 0.80 tell you? | The Random Forest overfits somewhat; cross-validation (0.8044 ± 0.0054) showed the test score was stable |
| Why compare against a baseline? | Linear Regression scored R² 0.5758, which shows the forest's 0.8046 is a real gain |
| MAE or RMSE? | MAE is the average error (0.3277, about $32,800); RMSE punishes big errors more (0.506). I chose by MAE and said so |
| Why load the model once at startup? | Reloading a 138 MB file on every request would be slow; the lifespan function loads it once, and a test checks it loads only once |
| What do 422, 503 and 500 mean in your API? | 422: bad input rejected by Pydantic; 503: model not loaded; 500: the model failed while predicting |
| Image vs container? | The image is the frozen package; each container is a running copy. One image started all three replicas |
| Why is the model not in Git or the image? | It is about 138 MB (over GitHub's 100 MB limit); Compose mounts `artifacts/` read-only instead |
| What does Nginx do here? | One public address, round-robin to three APIs; one prediction runs on one replica |
| Why did three replicas give about 4.7x throughput? | Measured on one laptop (13.32 vs 62.45 req/s); the setups differed slightly, so it is an observation, not a guarantee. It also used about three times the memory |
| What are P50, P95 and P99? | Median response time and the slower tail: 95% or 99% of requests were faster than that value |

---

## Closing perspective

This project started with a regression model and became a practical introduction to machine-learning systems engineering. Its main result is not a house price: it is the journey

**data and trained model → tested API → containerized service → Nginx-balanced replicas → measured performance.**

**Where I am now (17 Nov 2026):** the Docker, Nginx and load-testing milestone is complete and published; Kubernetes is in progress and CI/CD is next.

If you are just starting out, run the steps in Section 16 in order, complete a few exercises from Section 20, and you will have touched every layer of a basic MLOps pipeline.
