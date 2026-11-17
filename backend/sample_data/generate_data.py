import pandas as pd
import numpy as np

np.random.seed(42)

# 1. Generate reference dataset (normal data)
reference_data = pd.DataFrame({
    "age": np.random.randint(20, 45, 500),
    "income": np.random.normal(40000, 8000, 500).round(2),
    "experience": np.random.randint(0, 15, 500),
    "city": np.random.choice(
        ["Bengaluru", "Mumbai", "Chennai"],
        500,
        p=[0.5, 0.3, 0.2]
    ),
    "purchased": np.random.choice(
        ["Yes", "No"],
        500,
        p=[0.4, 0.6]
    )
})

# 2. Generate current dataset (intentionally changed data)
current_data = pd.DataFrame({
    "age": np.random.randint(30, 65, 500),
    "income": np.random.normal(60000, 12000, 500).round(2),
    "experience": np.random.randint(2, 20, 500),
    "city": np.random.choice(
        ["Bengaluru", "Mumbai", "Chennai"],
        500,
        p=[0.2, 0.3, 0.5]
    ),
    "purchased": np.random.choice(
        ["Yes", "No"],
        500,
        p=[0.6, 0.4]
    )
})

# 3. Introduce missing values
current_data.loc[10:19, "income"] = np.nan

# 4. Introduce duplicate records
current_data = pd.concat(
    [current_data, current_data.iloc[:5]],
    ignore_index=True
)

# 5. Save datasets
reference_data.to_csv("reference_data.csv", index=False)
current_data.to_csv("current_data.csv", index=False)

print("Sample datasets created successfully!")
print("Reference dataset:", reference_data.shape)
print("Current dataset:", current_data.shape)