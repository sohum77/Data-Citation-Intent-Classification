import pandas as pd
import matplotlib.pyplot as plt

# Path to your combined results CSV
csv_path = r"D:\Data Science Projects\Data Citation Intent Classification\models\inference_results\predictions_from_xml.csv"

# Read the CSV
df = pd.read_csv(csv_path)

# Count label distribution
counts = df["pred_label"].value_counts()

print("📊 Prediction Counts:")
print(counts)

# Plot a bar chart
plt.figure(figsize=(6,4))
plt.bar(counts.index, counts.values)
plt.title("Model Predictions: Primary vs Secondary")
plt.xlabel("Label Type")
plt.ylabel("Count")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

# Show the plot
plt.show()
