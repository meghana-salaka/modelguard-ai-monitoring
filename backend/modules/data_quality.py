import pandas as pd
import numpy as np


def analyze_data_quality(df):

    if df is None or df.empty:
        raise ValueError("Dataset is empty.")

    total_rows = len(df)
    total_columns = len(df.columns)

    # Missing values
    missing_values = df.isnull().sum()

    missing_percentage = (
        df.isnull().sum() / total_rows * 100
    )

    # Duplicate records
    duplicate_count = int(df.duplicated().sum())

    # Column information
    numerical_columns = df.select_dtypes(
        include=np.number
    ).columns.tolist()

    categorical_columns = df.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    # Feature-level report
    feature_report = []

    for column in df.columns:

        feature_report.append({

            "feature": column,

            "data_type": str(df[column].dtype),

            "missing_values": int(
                missing_values[column]
            ),

            "missing_percentage": round(
                missing_percentage[column], 2
            ),

            "unique_values": int(
                df[column].nunique(dropna=True)
            )

        })

    # Overall data quality score
    total_cells = total_rows * total_columns

    missing_cells = int(df.isnull().sum().sum())

    duplicate_cells = duplicate_count * total_columns

    quality_score = max(
        0,
        round(
            100 * (
                1 - (missing_cells + duplicate_cells) / total_cells
            ),
            2
        )
    )

    return {

        "total_rows": total_rows,

        "total_columns": total_columns,

        "missing_values": int(missing_cells),

        "duplicate_records": duplicate_count,

        "numerical_features": numerical_columns,

        "categorical_features": categorical_columns,

        "data_quality_score": quality_score,

        "feature_report": feature_report

    }