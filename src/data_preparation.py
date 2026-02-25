from pathlib import Path
import pandas as pd
import numpy as np


def truncate_outliers_iqr(df, factor=1.5):
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    Q1 = df[numeric_cols].quantile(0.25)
    Q3 = df[numeric_cols].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - factor * IQR
    upper = Q3 + factor * IQR

    df[numeric_cols] = df[numeric_cols].clip(lower=lower, upper=upper, axis=1)
    return df

if __name__ == "__main__":

    # Define paths
    input_path = "data/raw/boston_raw.csv"
    output_path = "data/training/boston_training.csv"

    # Load raw dataset
    df = pd.read_csv(input_path, delimiter=',')

    # Remove ID
    df = df.drop(columns=["ID"])

    # Truncar Outliers
    df = truncate_outliers_iqr(df)

    # Seleccionar columnas
    selected_features = ["lstat", "rm", "dis", "crim", "nox"]
    target = "medv"

    # Dataframe final
    df_final = df[selected_features + [target]]
    df_final.to_csv(output_path, index=False)
