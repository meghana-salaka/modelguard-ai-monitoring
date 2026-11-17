from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

import pandas as pd
import io

from backend.modules.data_quality import analyze_data_quality
from backend.modules.drift_detection import detect_data_drift
from backend.modules.model_evaluation import evaluate_model
from backend.modules.failure_diagnosis import diagnose_model_failure


# Initialize application
app = FastAPI(
    title="ModelGuard AI",
    description="ML Model Monitoring and Failure Diagnosis API",
    version="1.0.0"
)


# Enable frontend connection
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Home endpoint
@app.get("/")
def home():
    return {
        "message": "Welcome to ModelGuard AI",
        "status": "Running",
        "modules": [
            "Data Quality Monitoring",
            "Data Drift Detection",
            "Model Performance Evaluation",
            "Failure Diagnosis"
        ]
    }


# Health check
@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "application": "ModelGuard AI"
    }


# Upload and analyze datasets
@app.post("/analyze")
async def analyze_datasets(
    reference_file: UploadFile = File(...),
    current_file: UploadFile = File(...)
):

    try:

        # Read uploaded CSV files
        reference_content = await reference_file.read()
        current_content = await current_file.read()

        reference_df = pd.read_csv(
            io.BytesIO(reference_content)
        )

        current_df = pd.read_csv(
            io.BytesIO(current_content)
        )

        if reference_df.empty or current_df.empty:
            raise HTTPException(
                status_code=400,
                detail="Uploaded datasets cannot be empty."
            )

        # Data quality analysis
        quality_result = analyze_data_quality(
            current_df
        )

        reference_quality = analyze_data_quality(
            reference_df
        )

        # Data drift analysis
        drift_result = detect_data_drift(
            reference_df,
            current_df
        )

        # Failure diagnosis
        diagnosis_result = diagnose_model_failure(
            quality_result,
            drift_result
        )

        return {

            "success": True,

            "reference_dataset": {
                "rows": len(reference_df),
                "columns": len(reference_df.columns)
            },

            "current_dataset": {
                "rows": len(current_df),
                "columns": len(current_df.columns)
            },

            "data_quality": quality_result,

            "reference_data_quality": reference_quality,

            "drift_detection": drift_result,

            "failure_diagnosis": diagnosis_result

        }

    except HTTPException:
        raise

    except Exception as e:

        raise HTTPException(
            status_code=400,
            detail=str(e)
        )


# Model performance evaluation
@app.post("/evaluate")
async def evaluate_uploaded_model(
    file: UploadFile = File(...),
    actual_column: str = "actual_label",
    predicted_column: str = "predicted_label"
):

    try:

        content = await file.read()

        df = pd.read_csv(
            io.BytesIO(content)
        )

        if actual_column not in df.columns:
            raise HTTPException(
                status_code=400,
                detail=f"Missing column: {actual_column}"
            )

        if predicted_column not in df.columns:
            raise HTTPException(
                status_code=400,
                detail=f"Missing column: {predicted_column}"
            )

        if df[[actual_column, predicted_column]].isnull().any().any():
            raise HTTPException(
                status_code=400,
                detail="Actual and predicted labels cannot contain missing values."
            )

        result = evaluate_model(
            df[actual_column],
            df[predicted_column]
        )

        return {
            "success": True,
            "model_performance": result
        }

    except HTTPException:
        raise

    except Exception as e:

        raise HTTPException(
            status_code=400,
            detail=str(e)
        )

@app.post("/full-monitor")
async def full_monitoring(
    reference_file: UploadFile = File(...),
    current_file: UploadFile = File(...),
    predictions_file: UploadFile = File(...)
):
    try:
        # Read uploaded CSV files
        reference_df = pd.read_csv(
            io.BytesIO(await reference_file.read())
        )

        current_df = pd.read_csv(
            io.BytesIO(await current_file.read())
        )

        predictions_df = pd.read_csv(
            io.BytesIO(await predictions_file.read())
        )

        # Validate datasets
        if reference_df.empty or current_df.empty or predictions_df.empty:
            raise HTTPException(
                status_code=400,
                detail="Uploaded datasets cannot be empty."
            )

        # Validate prediction columns
        required_columns = [
            "actual_label",
            "predicted_label"
        ]

        for column in required_columns:
            if column not in predictions_df.columns:
                raise HTTPException(
                    status_code=400,
                    detail=f"Missing prediction column: {column}"
                )

        if predictions_df[required_columns].isnull().any().any():
            raise HTTPException(
                status_code=400,
                detail="Prediction labels cannot contain missing values."
            )

        # 1. Data Quality
        quality_result = analyze_data_quality(current_df)

        # 2. Data Drift
        drift_result = detect_data_drift(
            reference_df,
            current_df
        )

        # 3. Model Performance
        evaluation_result = evaluate_model(
            predictions_df["actual_label"],
            predictions_df["predicted_label"]
        )

        # 4. Failure Diagnosis
        diagnosis_result = diagnose_model_failure(
            quality_result,
            drift_result,
            evaluation_result
        )

        return {
            "success": True,
            "application": "ModelGuard AI",

            "data_quality": quality_result,

            "drift_detection": drift_result,

            "model_performance": evaluation_result,

            "failure_diagnosis": diagnosis_result
        }

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )
    