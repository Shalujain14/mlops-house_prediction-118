import pandas as pd
import boto3
from io import StringIO

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import mlflow
import mlflow.sklearn
import numpy as np


# =========================================================
# MLflow Tracking Server
# =========================================================

mlflow.set_tracking_uri("http://100.59.112.41:5000")

mlflow.set_experiment("mlops-house-prediction")


# =========================================================
# Fetch cleaned data from S3
# =========================================================

s3 = boto3.client("s3")

BUCKET = "mlops-house-prediction-118"

KEY = "processes/2026-09-25/Mlops_house_prediction_clean_v2.csv"


def fetch_data():
    obj = s3.get_object(
        Bucket=BUCKET,
        Key=KEY
    )

    df = pd.read_csv(
        StringIO(
            obj["Body"].read().decode("utf-8")
        )
    )

    return df


df = fetch_data()

print(f"Fetched shape: {df.shape}")


# =========================================================
# Features / Target
# =========================================================

X = df[
    [
        "sqft",
        "bedrooms",
        "bathrooms",
        "age_years",
        "garage",
        "location_score"
    ]
]

y = df["price"]


# =========================================================
# Train / Test Split
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# =========================================================
# MLflow Run
# =========================================================

with mlflow.start_run():

    # Model parameters
    n_estimators = 150
    max_depth = 8

    # Create model
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42
    )

    # Train
    model.fit(X_train, y_train)

    # Prediction
    preds = model.predict(X_test)

    # Metrics
    mae = mean_absolute_error(
        y_test,
        preds
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            preds
        )
    )

    r2 = r2_score(
        y_test,
        preds
    )


    # =====================================================
    # Log Parameters
    # =====================================================

    mlflow.log_param(
        "n_estimators",
        n_estimators
    )

    mlflow.log_param(
        "max_depth",
        max_depth
    )

    mlflow.log_param(
        "data_source",
        f"s3://{BUCKET}/{KEY}"
    )


    # =====================================================
    # Log Metrics
    # =====================================================

    mlflow.log_metric(
        "mae",
        mae
    )

    mlflow.log_metric(
        "rmse",
        rmse
    )

    mlflow.log_metric(
        "r2_score",
        r2
    )


    # =====================================================
    # Log + Register Model
    # =====================================================

    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        registered_model_name="house_price_prediction",
        skops_trusted_types=[
            "sklearn.tree._tree.Tree"
        ]
    )


    # =====================================================
    # Print Results
    # =====================================================

    print(
        f"\nMAE: {mae:.2f}"
        f" | RMSE: {rmse:.2f}"
        f" | R2: {r2:.4f}"
    )

    print(
        "\nModel registered as:"
        " house_price_prediction"
    )