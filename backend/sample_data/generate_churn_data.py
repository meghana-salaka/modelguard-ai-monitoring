import pandas as pd
import numpy as np
from pathlib import Path

# Reproducible random data
np.random.seed(42)

# Save generated files in the project root
OUTPUT_DIR = Path(__file__).resolve().parents[2]


# 1. Generate reference dataset
reference = pd.DataFrame({
    "tenure": np.random.randint(1, 60, 500),
    "monthly_charges": np.random.normal(65, 12, 500).round(2),
    "total_charges": np.random.normal(1800, 500, 500).round(2),
    "contract_type": np.random.choice(
        ["Monthly", "One Year", "Two Year"], 500,
        p=[0.6, 0.25, 0.15]
    ),
    "internet_service": np.random.choice(
        ["DSL", "Fiber", "No"], 500
    )
})


# 2. Generate current dataset with distribution changes
current = pd.DataFrame({
    "tenure": np.random.randint(1, 40, 500),
    "monthly_charges": np.random.normal(85, 18, 500).round(2),
    "total_charges": np.random.normal(2300, 700, 500).round(2),
    "contract_type": np.random.choice(
        ["Monthly", "One Year", "Two Year"], 500,
        p=[0.8, 0.12, 0.08]
    ),
    "internet_service": np.random.choice(
        ["DSL", "Fiber", "No"], 500,
        p=[0.2, 0.7, 0.1]
    )
})

# Introduce missing values
current.loc[0:7, "monthly_charges"] = np.nan

# Introduce duplicate records
current = pd.concat(
    [current, current.iloc[:4]],
    ignore_index=True
)


# 3. Generate prediction evaluation dataset
actual_labels = np.random.choice(
    ["Yes", "No"], 200, p=[0.35, 0.65]
)

predicted_labels = actual_labels.copy()

# Introduce approximately 15% incorrect predictions
wrong_indices = np.random.choice(
    200, size=30, replace=False
)

for index in wrong_indices:
    predicted_labels[index] = (
        "No" if actual_labels[index] == "Yes" else "Yes"
    )

predictions = pd.DataFrame({
    "actual_label": actual_labels,
    "predicted_label": predicted_labels
})


# 4. Save all datasets
reference.to_csv(OUTPUT_DIR / "churn_reference.csv", index=False)
current.to_csv(OUTPUT_DIR / "churn_current.csv", index=False)
predictions.to_csv(OUTPUT_DIR / "churn_predictions.csv", index=False)

print("Customer Churn datasets generated successfully!")
print("Reference rows:", len(reference))
print("Current rows:", len(current))
print("Prediction samples:", len(predictions))
print("Output folder:", OUTPUT_DIR)