import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import mlflow
import mlflow.sklearn


def split_features_target(df: pd.DataFrame, target_col: str = "medv"):
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y


def build_grid_search(random_state: int = 42) -> GridSearchCV:
    modelo_base = DecisionTreeRegressor(random_state=random_state)

    param_grid = {
        "max_depth": [3, 5, 7, 10],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": [None, "sqrt"],
    }

    grid_search = GridSearchCV(
        estimator=modelo_base,
        param_grid=param_grid,
        cv=5,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
        verbose=1,
    )
    return grid_search


def train_best_model(grid_search: GridSearchCV, X_train, y_train):
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_


def compute_metrics(y_test, y_pred):
    mae_clean = mean_absolute_error(y_test, y_pred)
    rmse_clean = np.sqrt(mean_squared_error(y_test, y_pred))
    mape_clean = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    r2_clean = r2_score(y_test, y_pred)
    return mae_clean, rmse_clean, mape_clean, r2_clean


def save_model(model, output_model_path: str) -> None:
    with open(output_model_path, "wb") as f:
        pickle.dump(model, f)


if __name__ == "__main__":

    # Paths
    input_path = "data/training/boston_training.csv"
    output_model_path = "models/model.pkl"

    # MLflow (registro opcional)
    mlflow.set_experiment("housing_price_regression")
    registered_model_name = "housing_price_model"  # puedes cambiarlo

    # Load training dataset
    df = pd.read_csv(input_path, delimiter=",")

    # Split X / y
    X, y = split_features_target(df, target_col="medv")

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    grid_search = build_grid_search(random_state=42)

    with mlflow.start_run(run_name="decision_tree_gridsearch"):

        # Train best model
        mejor_modelo = train_best_model(grid_search, X_train, y_train)
        y_pred_final = mejor_modelo.predict(X_test)

        # Metrics
        mae_clean, rmse_clean, mape_clean, r2_clean = compute_metrics(y_test, y_pred_final)

        metricas_outliers = pd.DataFrame({
            "Métrica": ["MAE", "RMSE", "MAPE", "R^2"],
            "Valor (con outliers tratados)": [mae_clean, rmse_clean, mape_clean, r2_clean]
        })
        print(metricas_outliers)

        # MLflow logs
        mlflow.log_params(grid_search.best_params_)
        mlflow.log_metric("mae", mae_clean)
        mlflow.log_metric("rmse", rmse_clean)
        mlflow.log_metric("mape", mape_clean)
        mlflow.log_metric("r2", r2_clean)

        mlflow.sklearn.log_model(
            sk_model=mejor_modelo,
            artifact_path="model",
            registered_model_name=registered_model_name
        )

        # Save model locally
        save_model(mejor_modelo, output_model_path)
