#!/usr/bin/env python3
"""
Simple and Fast Crop Classification
Quick model for testing the corrected AVIRIS data
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import seaborn as sns
import joblib
import time

print("="*60)
print("SIMPLE CROP CLASSIFICATION")
print("="*60)

# Load data
print("\nLoading data...")
data = np.load("data/processed/train_samples_aviris_crops.npz")
X = data['X']
y = data['y']
classes = data['classes']

print(f"Data shape: {X.shape}")
print(f"Number of classes: {len(np.unique(y))}")

# Use a subset for quick testing (adjust as needed)
SAMPLE_SIZE = 50000
if len(X) > SAMPLE_SIZE:
    print(f"\nUsing random sample of {SAMPLE_SIZE:,} pixels for quick testing")
    idx = np.random.choice(len(X), SAMPLE_SIZE, replace=False)
    X = X[idx]
    y = y[idx]

# Split data
print("\nSplitting data (80/20)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train: {X_train.shape[0]:,} samples")
print(f"Test:  {X_test.shape[0]:,} samples")

# Scale features
print("\nScaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a simple Random Forest
print("\nTraining Random Forest...")
start_time = time.time()

rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=15,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train_scaled, y_train)
train_time = time.time() - start_time

print(f"Training completed in {train_time:.1f} seconds")

# Evaluate
print("\nEvaluating...")
y_pred = rf.predict(X_test_scaled)

train_acc = rf.score(X_train_scaled, y_train)
test_acc = rf.score(X_test_scaled, y_test)

print(f"\nResults:")
print(f"  Train accuracy: {train_acc:.4f}")
print(f"  Test accuracy:  {test_acc:.4f}")

# Define crop names
code_to_name = {
    1: "Corn",
    67: "Peaches",
    69: "Grapes",
    72: "Citrus",
    75: "Almonds",
    204: "Pistachios",
}

enc2code = {i: int(code) for i, code in enumerate(classes.tolist())}
enc2name = {i: code_to_name.get(int(code)) for i, code in enc2code.items()}
class_names = [enc2name[i] for i in sorted(enc2name.keys())]

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=class_names))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

# Visualizations
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Plot 1: Confusion Matrix
ax = axes[0]
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names,
            ax=ax)
ax.set_title('Confusion Matrix')
ax.set_xlabel('Predicted')
ax.set_ylabel('True')

# Plot 2: Per-class accuracy
ax = axes[1]
per_class_acc = cm.diagonal() / cm.sum(axis=1)
bars = ax.bar(class_names, per_class_acc, color='skyblue')
ax.set_ylabel('Accuracy')
ax.set_title('Per-Class Accuracy')
ax.set_ylim(0, 1)
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
for bar, acc in zip(bars, per_class_acc):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{acc:.3f}', ha='center', va='bottom')

# Plot 3: Feature importance (top 30)
ax = axes[2]
importances = rf.feature_importances_
top_30 = np.argsort(importances)[::-1][:30]
ax.bar(range(30), importances[top_30], color='green')
ax.set_xlabel('Feature Rank')
ax.set_ylabel('Importance')
ax.set_title('Top 30 Features')

plt.suptitle(f'Random Forest Results - Test Accuracy: {test_acc:.3f}', 
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/figs/simple_classification_results.png', dpi=150)
plt.show()

# Save the model
print("\nSaving model...")
joblib.dump(rf, 'outputs/models/simple_rf_model.pkl')
joblib.dump(scaler, 'outputs/models/simple_scaler.pkl')
print("✓ Model saved")

# Final summary
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"Test Accuracy: {test_acc:.4f}")
print(f"Training Time: {train_time:.1f} seconds")
print(f"Improvement over baseline (56%): {(test_acc - 0.56) / 0.56 * 100:.1f}%")

# Show which wavelengths are most important
wavelengths = np.linspace(380, 2510, X.shape[1])
top_wavelengths = wavelengths[top_30[:10]]
print(f"\nTop 10 most important wavelengths (nm):")
for i, wl in enumerate(top_wavelengths):
    print(f"  {i+1:2d}. {wl:.1f} nm")
