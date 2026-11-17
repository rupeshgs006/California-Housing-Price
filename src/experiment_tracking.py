
import mlflow
import mlflow.sklearn


def log_experiment(model, params, metrics,run_name):

    mlflow.set_experiment("House Price Prediction")

    with mlflow.start_run(run_name=run_name):

        # Log model hyperparameters
        mlflow.log_params(params)

        # Log evaluation metrics
        mlflow.log_metrics(metrics)

        # Log trained model
        mlflow.sklearn.log_model(
            model,
            name="random_forest_model",
            skops_trusted_types=[
                "sklearn.tree._tree.Tree"
            ]
        )

        print(f"Experiment {run_name} logged successfully!")


