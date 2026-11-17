import pandas as pd
import numpy as np

from scipy.stats import ks_2samp, chi2_contingency


def detect_data_drift(reference_df, current_df, significance_level=0.05):

    if reference_df is None or current_df is None:
        raise ValueError("Both datasets are required.")

    if reference_df.empty or current_df.empty:
        raise ValueError("Datasets cannot be empty.")

    common_columns = list(
        set(reference_df.columns) & set(current_df.columns)
    )

    if not common_columns:
        raise ValueError("No common features found between datasets.")

    results = []

    for column in common_columns:

        ref = reference_df[column]
        cur = current_df[column]

        # Check whether both columns are numerical
        if (
            pd.api.types.is_numeric_dtype(ref)
            and pd.api.types.is_numeric_dtype(cur)
        ):

            ref_values = pd.to_numeric(
                ref, errors="coerce"
            ).dropna()

            cur_values = pd.to_numeric(
                cur, errors="coerce"
            ).dropna()

            if len(ref_values) < 2 or len(cur_values) < 2:
                results.append({
                    "feature": column,
                    "feature_type": "Numerical",
                    "status": "Insufficient Data",
                    "p_value": None,
                    "drift_detected": False
                })
                continue

            # Kolmogorov-Smirnov test
            statistic, p_value = ks_2samp(
                ref_values,
                cur_values
            )

            drift_detected = p_value < significance_level

            results.append({

                "feature": column,
                "feature_type": "Numerical",
                "test": "Kolmogorov-Smirnov Test",

                "ks_statistic": round(
                    float(statistic), 4
                ),

                "p_value": round(
                    float(p_value), 6
                ),

                "reference_mean": round(
                    float(ref_values.mean()), 2
                ),

                "current_mean": round(
                    float(cur_values.mean()), 2
                ),

                "drift_detected": bool(drift_detected),

                "status": (
                    "Drift Detected"
                    if drift_detected
                    else "No Drift"
                )

            })

        else:

            # Treat non-numerical columns as categorical
            ref_values = ref.fillna(
                "__MISSING__"
            ).astype(str)

            cur_values = cur.fillna(
                "__MISSING__"
            ).astype(str)

            categories = sorted(
                set(ref_values.unique()) |
                set(cur_values.unique())
            )

            ref_counts = (
                ref_values.value_counts()
                .reindex(categories, fill_value=0)
            )

            cur_counts = (
                cur_values.value_counts()
                .reindex(categories, fill_value=0)
            )

            # Chi-square test
            if len(categories) < 2:

                statistic = 0.0
                p_value = 1.0

            else:

                contingency_table = np.array([
                    ref_counts.values,
                    cur_counts.values
                ])

                statistic, p_value, _, _ = chi2_contingency(
                    contingency_table
                )

            drift_detected = p_value < significance_level

            results.append({

                "feature": column,
                "feature_type": "Categorical",
                "test": "Chi-Square Test",

                "chi2_statistic": round(
                    float(statistic), 4
                ),

                "p_value": round(
                    float(p_value), 6
                ),

                "drift_detected": bool(drift_detected),

                "status": (
                    "Drift Detected"
                    if drift_detected
                    else "No Drift"
                )

            })

    drifted_features = [
        item["feature"]
        for item in results
        if item["drift_detected"]
    ]

    return {

        "total_features_analyzed": len(results),

        "drifted_features_count": len(drifted_features),

        "drifted_features": drifted_features,

        "significance_level": significance_level,

        "results": results

    }