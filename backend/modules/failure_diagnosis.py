def diagnose_model_failure(
    quality_result,
    drift_result,
    evaluation_result=None
):

    issues = []
    recommendations = []

    # --------------------------------
    # 1. DATA QUALITY INVESTIGATION
    # --------------------------------

    missing_values = quality_result.get(
        "missing_values", 0
    )

    duplicate_records = quality_result.get(
        "duplicate_records", 0
    )

    quality_score = quality_result.get(
        "data_quality_score", 100
    )

    if missing_values > 0:

        issues.append({
            "category": "Data Quality",
            "severity": "Warning",
            "issue": f"{missing_values} missing values detected."
        })

        recommendations.append(
            "Investigate missing values and apply an appropriate imputation strategy."
        )

    if duplicate_records > 0:

        issues.append({
            "category": "Data Quality",
            "severity": "Warning",
            "issue": f"{duplicate_records} duplicate records detected."
        })

        recommendations.append(
            "Investigate duplicate records before model evaluation."
        )

    if quality_score < 70:

        issues.append({
            "category": "Data Quality",
            "severity": "Critical",
            "issue": "Overall data quality score is low."
        })

        recommendations.append(
            "Review dataset quality and resolve major data issues."
        )

    # --------------------------------
    # 2. DATA DRIFT INVESTIGATION
    # --------------------------------

    drifted_features = drift_result.get(
        "drifted_features", []
    )

    if drifted_features:

        issues.append({
            "category": "Data Drift",
            "severity": "Warning",
            "issue": (
                f"Distribution changes detected in "
                f"{len(drifted_features)} features."
            ),
            "affected_features": drifted_features
        })

        recommendations.append(
            "Investigate distribution changes in affected features "
            "and assess whether model retraining or recalibration is needed."
        )

    # --------------------------------
    # 3. MODEL PERFORMANCE INVESTIGATION
    # --------------------------------

    if evaluation_result is not None:

        accuracy = evaluation_result.get(
            "accuracy", 1
        )

        precision = evaluation_result.get(
            "precision_macro", 1
        )

        recall = evaluation_result.get(
            "recall_macro", 1
        )

        f1 = evaluation_result.get(
            "f1_score_macro", 1
        )

        if accuracy < 0.70:

            issues.append({
                "category": "Model Performance",
                "severity": "Critical",
                "issue": f"Low model accuracy: {accuracy:.2%}"
            })

            recommendations.append(
                "Investigate incorrect predictions, training data quality "
                "and possible model degradation."
            )

        if precision < 0.70:

            issues.append({
                "category": "Model Performance",
                "severity": "Warning",
                "issue": f"Low macro precision: {precision:.2%}"
            })

            recommendations.append(
                "Investigate false positive predictions and class-wise precision."
            )

        if recall < 0.70:

            issues.append({
                "category": "Model Performance",
                "severity": "Warning",
                "issue": f"Low macro recall: {recall:.2%}"
            })

            recommendations.append(
                "Investigate false negatives and classes the model fails to identify."
            )

        if f1 < 0.70:

            issues.append({
                "category": "Model Performance",
                "severity": "Warning",
                "issue": f"Low macro F1-score: {f1:.2%}"
            })

            recommendations.append(
                "Review the balance between precision and recall."
            )

        # --------------------------------
        # 4. CLASS-WISE ERROR INVESTIGATION
        # --------------------------------

        confusion = evaluation_result.get(
            "confusion_matrix", []
        )

        class_labels = evaluation_result.get(
            "class_labels", []
        )

        for index, row in enumerate(confusion):

            total = sum(row)

            if total == 0:
                continue

            correct = row[index]

            error_rate = (total - correct) / total

            if error_rate > 0.30:

                label = (
                    class_labels[index]
                    if index < len(class_labels)
                    else str(index)
                )

                issues.append({
                    "category": "Classification Errors",
                    "severity": "Warning",
                    "issue": (
                        f"High classification error rate "
                        f"for class {label}: {error_rate:.2%}"
                    )
                })

                recommendations.append(
                    f"Investigate misclassifications for class {label} "
                    "and review its training examples."
                )

    # --------------------------------
    # 5. OVERALL DIAGNOSIS
    # --------------------------------

    critical_count = sum(
        1 for issue in issues
        if issue["severity"] == "Critical"
    )

    warning_count = sum(
        1 for issue in issues
        if issue["severity"] == "Warning"
    )

    if critical_count > 0:

        overall_status = "Critical"

    elif warning_count > 0:

        overall_status = "Warning"

    else:

        overall_status = "Healthy"

    if not issues:

        recommendations.append(
            "No significant issues detected by the current monitoring checks. "
            "Continue monitoring incoming data and model performance."
        )

    return {

        "overall_status": overall_status,

        "total_issues": len(issues),

        "critical_issues": critical_count,

        "warning_issues": warning_count,

        "issues": issues,

        "recommendations": recommendations

    }