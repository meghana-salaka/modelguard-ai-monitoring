import pandas as pd
import numpy as np

np.random.seed(42)

# Generate actual labels
actual = np.random.choice(
    ["Yes", "No"],
    size=200,
    p=[0.4, 0.6]
)

# Generate predictions with some intentional errors
predicted = []

for label in actual:
    if np.random.random() < 0.85:
        predicted.append(label)
    else:
        predicted.append("No" if label == "Yes" else "Yes")

# Create dataset
df = pd.DataFrame({
    "actual_label": actual,
    "predicted_label": predicted
})

# Save CSV
df.to_csv("predictions.csv", index=False)

print("Prediction dataset created successfully!")
print("Total samples:", len(df))
print("\nSample records:")
print(df.head(10))