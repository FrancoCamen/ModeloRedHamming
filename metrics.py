import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_score, classification_report
import numpy as np

# Load the CSV data from the specified path
df = pd.read_csv("dataset/test/dataoutput.csv")


# Function to map true labels
def get_true_label(row):
    if row["has_pattern"] == 0:
        return "none"
    else:
        return row["pattern"]


# Function to map predicted labels (general, in case pattern_detected can be 0 in future)
def get_pred_label(row):
    if row["pattern_detected"] == 0:
        return "none"
    else:
        return row["detected_pattern"]


# Apply the mappings
df["true_label"] = df.apply(get_true_label, axis=1)
df["pred_label"] = df.apply(get_pred_label, axis=1)

# Define the classes (including 'none' for completeness)
classes = ["ceda", "via prioritaria", "pare"]

# Ensure all labels are in the classes; if not, handle missing ones
all_true = df["true_label"].unique()
all_pred = df["pred_label"].unique()
print(f"Unique true labels: {all_true}")
print(f"Unique predicted labels: {all_pred}")

# Compute confusion matrix
cm = confusion_matrix(df["true_label"], df["pred_label"], labels=classes)

# Print confusion matrix
print("Confusion Matrix:")
print(cm)

# Compute precision (macro average)
precision = precision_score(
    df["true_label"], df["pred_label"], labels=classes, average="macro", zero_division=0
)
print(f"\nMacro Precision: {precision:.4f}")

# Detailed classification report
print("\nClassification Report:")
print(
    classification_report(
        df["true_label"], df["pred_label"], labels=classes, zero_division=0
    )
)

# Plot confusion matrix as a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes
)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()
