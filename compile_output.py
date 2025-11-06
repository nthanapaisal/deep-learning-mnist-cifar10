import json
import os
import pandas as pd

# --- Configuration ---
# Add here the filenames you want to combine:
json_files = [
    "nn_final_report.json",
    "cnn_baseline_report.json",
    "cnn_enhanced_report.json",
    "cnn_deep_report.json"
]

# --- Load all JSONs ---
all_results = {}

for file in json_files:
    if os.path.exists(file):
        with open(file, "r") as f:
            data = json.load(f)
            all_results.update(data)
    else:
        print(f"⚠️ File not found: {file}")

# --- Flatten results into a list of rows ---
rows = []
for model_name, metrics in all_results.items():
    row = {
        "Architecture": model_name,
        "Learning rate": metrics.get("lr"),
        "Batch size": metrics.get("batch_size"),
        "Optimizer": metrics.get("optimizer"),
        "Dropout": metrics.get("dropout_rate", "N/A"),
        "Validation Acc": metrics.get("val_acc"),
        "±std": metrics.get("std_val_acc"),
        "Runtime (min)": metrics.get("runtime"),
        "Test Accuracy": metrics.get("accuracy"),
    }
    rows.append(row)

# --- Create table ---
df = pd.DataFrame(rows)

# --- Sort (optional) ---
df = df.sort_values(by="Architecture")

# --- Print clean table ---
print(df.to_string(index=False))

# --- Save to CSV for LaTeX or Excel ---
df.to_csv("summary_table.csv", index=False)
print("Saved to summary_table.csv")
