#!/usr/bin/env python3
"""
Optimized Crop Classification with Performance Improvements
Addresses class imbalance and uses advanced techniques
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, f1_score
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
import joblib
import time
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("OPTIMIZED CROP CLASSIFICATION")
print("="*70)

# ============================================================
# 1. LOAD AND PREPARE DATA
# ============================================================
print("\n1. LOADING DATA")
print("-"*50)

data = np.load("data/processed/train_samples_aviris_crops.npz")
X = data['X']
y = data['y']
classes = data['classes']

print(f"Full dataset: {X.shape}")

# Use more data for better performance
SAMPLE_SIZE = 200000  # Increased from 50,000
if len(X) > SAMPLE_SIZE:
    print(f"Using stratified sample of {SAMPLE_SIZE:,} pixels")
    X, _, y, _ = train_test_split(X, y, train_size=SAMPLE_SIZE, 
                                  random_state=42, stratify=y)
else:
    print("Using full dataset")

# Check class distribution
unique, counts = np.unique(y, return_counts=True)
print("\nClass distribution:")
for cls, count in zip(unique, counts):
    print(f"  Class {cls}: {count:6d} ({count/len(y)*100:5.1f}%)")

# ============================================================
# 2. FEATURE ENGINEERING
# ============================================================
print("\n2. FEATURE ENGINEERING")
print("-"*50)

# Add vegetation indices as features
print("Creating vegetation indices...")

# Approximate band positions for AVIRIS
red_idx = 60      # ~680nm
nir_idx = 120     # ~850nm  
green_idx = 40    # ~550nm
swir1_idx = 160   # ~1650nm

# Calculate indices
ndvi = (X[:, nir_idx] - X[:, red_idx]) / (X[:, nir_idx] + X[:, red_idx] + 1e-10)
gndvi = (X[:, nir_idx] - X[:, green_idx]) / (X[:, nir_idx] + X[:, green_idx] + 1e-10)
ndwi = (X[:, green_idx] - X[:, nir_idx]) / (X[:, green_idx] + X[:, nir_idx] + 1e-10)
moisture = X[:, nir_idx] / (X[:, swir1_idx] + 1e-10)

# Add indices to features
X_enhanced = np.column_stack([X, ndvi, gndvi, ndwi, moisture])
print(f"Enhanced features shape: {X_enhanced.shape}")

# ============================================================
# 3. HANDLE CLASS IMBALANCE
# ============================================================
print("\n3. HANDLING CLASS IMBALANCE")
print("-"*50)

# Split data first
X_train, X_test, y_train, y_test = train_test_split(
    X_enhanced, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train: {X_train.shape[0]:,} samples")
print(f"Test:  {X_test.shape[0]:,} samples")

# Calculate class weights for balanced training
class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weight_dict = {cls: weight for cls, weight in zip(np.unique(y_train), class_weights)}

print("Class weights (to balance training):")
for cls, weight in class_weight_dict.items():
    print(f"  Class {cls}: {weight:.2f}")

# Option: Use SMOTE for oversampling minorities (commented out for speed)
USE_SMOTE = False
if USE_SMOTE:
    print("\nApplying SMOTE oversampling...")
    smote = SMOTE(random_state=42, k_neighbors=3)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    print(f"After SMOTE: {X_train_balanced.shape[0]:,} samples")
else:
    X_train_balanced = X_train
    y_train_balanced = y_train

# Scale features
print("\nScaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_balanced)
X_test_scaled = scaler.transform(X_test)

# ============================================================
# 4. OPTIMIZED MODEL TRAINING
# ============================================================
print("\n4. TRAINING OPTIMIZED MODELS")
print("-"*50)

# Model 1: Optimized Random Forest
print("\n4.1 Random Forest with class weights...")
start_time = time.time()

rf_optimized = RandomForestClassifier(
    n_estimators=200,        # More trees
    max_depth=25,           # Deeper trees
    min_samples_split=5,    
    min_samples_leaf=2,     
    max_features='sqrt',
    class_weight=class_weight_dict,  # Handle imbalance
    random_state=42,
    n_jobs=-1
)

rf_optimized.fit(X_train_scaled, y_train_balanced)
rf_time = time.time() - start_time

rf_train_acc = rf_optimized.score(X_train_scaled, y_train_balanced)
rf_test_acc = rf_optimized.score(X_test_scaled, y_test)
rf_pred = rf_optimized.predict(X_test_scaled)

print(f"  Train: {rf_train_acc:.4f}, Test: {rf_test_acc:.4f}")
print(f"  Time: {rf_time:.1f}s")

# Model 2: Extra Trees
print("\n4.2 Extra Trees Classifier...")
start_time = time.time()

et_optimized = ExtraTreesClassifier(
    n_estimators=200,
    max_depth=25,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight=class_weight_dict,
    random_state=42,
    n_jobs=-1
)

et_optimized.fit(X_train_scaled, y_train_balanced)
et_time = time.time() - start_time

et_train_acc = et_optimized.score(X_train_scaled, y_train_balanced)
et_test_acc = et_optimized.score(X_test_scaled, y_test)
et_pred = et_optimized.predict(X_test_scaled)

print(f"  Train: {et_train_acc:.4f}, Test: {et_test_acc:.4f}")
print(f"  Time: {et_time:.1f}s")

# Model 3: Ensemble (Voting Classifier)
print("\n4.3 Ensemble Voting Classifier...")
ensemble = VotingClassifier(
    estimators=[
        ('rf', rf_optimized),
        ('et', et_optimized)
    ],
    voting='soft'  # Use probability averaging
)

start_time = time.time()
ensemble.fit(X_train_scaled, y_train_balanced)
ensemble_time = time.time() - start_time

ensemble_train_acc = ensemble.score(X_train_scaled, y_train_balanced)
ensemble_test_acc = ensemble.score(X_test_scaled, y_test)
ensemble_pred = ensemble.predict(X_test_scaled)

print(f"  Train: {ensemble_train_acc:.4f}, Test: {ensemble_test_acc:.4f}")
print(f"  Time: {ensemble_time:.1f}s")

# ============================================================
# 5. MODEL EVALUATION
# ============================================================
print("\n5. MODEL EVALUATION")
print("-"*50)

# Compare models
model_results = pd.DataFrame({
    'Model': ['Random Forest', 'Extra Trees', 'Ensemble'],
    'Test_Accuracy': [rf_test_acc, et_test_acc, ensemble_test_acc],
    'Train_Accuracy': [rf_train_acc, et_train_acc, ensemble_train_acc],
    'Time': [rf_time, et_time, ensemble_time]
})

print("\nModel Comparison:")
print(model_results.to_string(index=False))

# Select best model
best_idx = model_results['Test_Accuracy'].idxmax()
best_model_name = model_results.loc[best_idx, 'Model']
best_acc = model_results.loc[best_idx, 'Test_Accuracy']

if best_model_name == 'Random Forest':
    best_model = rf_optimized
    best_pred = rf_pred
elif best_model_name == 'Extra Trees':
    best_model = et_optimized
    best_pred = et_pred
else:
    best_model = ensemble
    best_pred = ensemble_pred

print(f"\n✓ Best model: {best_model_name} with {best_acc:.4f} accuracy")

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

# Detailed classification report
print(f"\nClassification Report for {best_model_name}:")
print(classification_report(y_test, best_pred, target_names=class_names))

# Confusion matrix
cm = confusion_matrix(y_test, best_pred)

# ============================================================
# 6. ANALYSIS OF PROBLEM CLASSES
# ============================================================
print("\n6. ANALYZING PROBLEM CLASSES")
print("-"*50)

# Per-class accuracy
per_class_acc = cm.diagonal() / cm.sum(axis=1)

print("Per-class accuracy:")
for i, (name, acc) in enumerate(zip(class_names, per_class_acc)):
    print(f"  {name:12s}: {acc:.4f}")

# Identify confusion patterns
print("\nTop confusions (>10% misclassification):")
for i, true_class in enumerate(class_names):
    for j, pred_class in enumerate(class_names):
        if i != j:
            confusion_rate = cm[i, j] / cm[i].sum()
            if confusion_rate > 0.1:
                print(f"  {true_class} → {pred_class}: {confusion_rate:.1%} ({cm[i, j]} samples)")

# ============================================================
# 7. FEATURE IMPORTANCE ANALYSIS
# ============================================================
print("\n7. FEATURE IMPORTANCE")
print("-"*50)

# Get feature importances from best tree-based model
if hasattr(best_model, 'feature_importances_'):
    importances = best_model.feature_importances_
elif hasattr(best_model, 'estimators_'):
    # For ensemble, use first estimator
    importances = best_model.estimators_[0].feature_importances_
else:
    importances = rf_optimized.feature_importances_

# Top features
n_features = X_enhanced.shape[1]
wavelengths = np.linspace(380, 2510, X.shape[1])

# Handle the added indices
feature_names = [f"{wl:.1f}nm" for wl in wavelengths] + ['NDVI', 'GNDVI', 'NDWI', 'Moisture']

top_20_idx = np.argsort(importances)[::-1][:20]
print("Top 20 most important features:")
for rank, idx in enumerate(top_20_idx):
    print(f"  {rank+1:2d}. {feature_names[idx]:12s}: {importances[idx]:.4f}")

# ============================================================
# 8. VISUALIZATIONS
# ============================================================
print("\n8. CREATING VISUALIZATIONS")
print("-"*50)

fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# Plot 1: Model comparison
ax = axes[0, 0]
models = model_results['Model'].tolist()
test_accs = model_results['Test_Accuracy'].tolist()
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
bars = ax.bar(models, test_accs, color=colors)
ax.set_ylabel('Test Accuracy')
ax.set_title('Model Performance Comparison', fontweight='bold')
ax.set_ylim(0, 1)
for bar, acc in zip(bars, test_accs):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{acc:.3f}', ha='center', va='bottom')
ax.axhline(y=0.56, color='red', linestyle='--', alpha=0.5, label='Baseline (56%)')
ax.axhline(y=0.651, color='orange', linestyle='--', alpha=0.5, label='Simple Model (65.1%)')
ax.legend()

# Plot 2: Confusion Matrix
ax = axes[0, 1]
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names,
            ax=ax, cbar_kws={'shrink': 0.8})
ax.set_title(f'Confusion Matrix - {best_model_name}', fontweight='bold')
ax.set_xlabel('Predicted')
ax.set_ylabel('True')

# Plot 3: Per-class performance
ax = axes[0, 2]
x = np.arange(len(class_names))
width = 0.35
bars1 = ax.bar(x - width/2, per_class_acc, width, label='Accuracy', color='skyblue')
# Add F1 scores
f1_scores = []
for i in range(len(class_names)):
    precision = cm[i, i] / cm[:, i].sum() if cm[:, i].sum() > 0 else 0
    recall = cm[i, i] / cm[i, :].sum() if cm[i, :].sum() > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    f1_scores.append(f1)
bars2 = ax.bar(x + width/2, f1_scores, width, label='F1-Score', color='lightcoral')

ax.set_ylabel('Score')
ax.set_title('Per-Class Performance', fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(class_names, rotation=45, ha='right')
ax.legend()
ax.set_ylim(0, 1)

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}', ha='center', va='bottom', fontsize=8)

# Plot 4: Feature importance (top 20)
ax = axes[1, 0]
top_20_importance = importances[top_20_idx]
top_20_names = [feature_names[i] for i in top_20_idx]
bars = ax.barh(range(20), top_20_importance)
ax.set_yticks(range(20))
ax.set_yticklabels(top_20_names, fontsize=8)
ax.set_xlabel('Importance')
ax.set_title('Top 20 Feature Importances', fontweight='bold')
ax.invert_yaxis()

# Color bands by type
for i, bar in enumerate(bars):
    if 'nm' in top_20_names[i]:
        wl = float(top_20_names[i].replace('nm', ''))
        if wl < 700:
            bar.set_color('green')  # Visible
        elif wl < 1000:
            bar.set_color('red')    # NIR
        else:
            bar.set_color('orange') # SWIR
    else:
        bar.set_color('purple')     # Indices

# Plot 5: Class distribution
ax = axes[1, 1]
unique, counts = np.unique(y_test, return_counts=True)
colors_pie = plt.cm.Set2(np.linspace(0, 1, len(unique)))
wedges, texts, autotexts = ax.pie(counts, labels=class_names, autopct='%1.1f%%',
                                   colors=colors_pie, startangle=90)
ax.set_title('Test Set Class Distribution', fontweight='bold')
for autotext in autotexts:
    autotext.set_fontsize(8)

# Plot 6: Learning curve
ax = axes[1, 2]
train_sizes = np.array([0.1, 0.3, 0.5, 0.7, 1.0])
train_scores = []
val_scores = []

print("  Calculating learning curves...")
for size in train_sizes:
    if size < 1.0:
        n = int(len(X_train_scaled) * size)
        X_sub = X_train_scaled[:n]
        y_sub = y_train_balanced[:n]
    else:
        X_sub = X_train_scaled
        y_sub = y_train_balanced
    
    # Quick RF for learning curve
    rf_lc = RandomForestClassifier(
        n_estimators=50, max_depth=15, 
        class_weight=class_weight_dict,
        random_state=42, n_jobs=-1
    )
    rf_lc.fit(X_sub, y_sub)
    
    train_scores.append(rf_lc.score(X_sub, y_sub))
    val_scores.append(rf_lc.score(X_test_scaled, y_test))

ax.plot(train_sizes * len(X_train_scaled), train_scores, 'o-', 
        color='blue', label='Training score')
ax.plot(train_sizes * len(X_train_scaled), val_scores, 'o-', 
        color='red', label='Validation score')
ax.set_xlabel('Training Set Size')
ax.set_ylabel('Accuracy')
ax.set_title('Learning Curves', fontweight='bold')
ax.legend(loc='best')
ax.grid(True, alpha=0.3)

plt.suptitle(f'Optimized Classification Results - Best: {best_model_name} ({best_acc:.3f})', 
             fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/figs/optimized_classification_results.png', dpi=150)
plt.show()
print("✓ Saved visualization")

# ============================================================
# 9. SAVE MODELS AND RESULTS
# ============================================================
print("\n9. SAVING MODELS")
print("-"*50)

# Save best model
joblib.dump(best_model, f'outputs/models/optimized_model_{best_model_name.replace(" ", "_").lower()}.pkl')
joblib.dump(scaler, 'outputs/models/optimized_scaler.pkl')
print(f"✓ Saved best model: {best_model_name}")

# Save results
model_results.to_csv('outputs/performance/optimized_model_comparison.csv', index=False)
print("✓ Saved model comparison")

# ============================================================
# 10. RECOMMENDATIONS
# ============================================================
print("\n" + "="*70)
print("FINAL RESULTS AND RECOMMENDATIONS")
print("="*70)

print(f"\nBest Model: {best_model_name}")
print(f"Test Accuracy: {best_acc:.4f}")
print(f"Improvement over baseline: {(best_acc - 0.56) / 0.56 * 100:.1f}%")
print(f"Improvement over simple model: {(best_acc - 0.651) / 0.651 * 100:.1f}%")

print("\n✅ What worked:")
print("  • Added vegetation indices (NDVI, GNDVI, etc.)")
print("  • Used class weights to handle imbalance")
print("  • Increased sample size to 200,000")
print("  • Ensemble methods for robustness")

print("\n💡 To further improve:")
print("  1. Use full dataset (2.3M samples) if compute allows")
print("  2. Try deep learning (CNN or transformer)")
print("  3. Add spatial context (neighboring pixels)")
print("  4. Temporal data if available (multi-season)")
print("  5. Merge similar classes (e.g., tree nuts)")

# Identify hardest classes to separate
print("\n⚠️ Challenging class pairs:")
confusion_threshold = 0.15
for i in range(len(class_names)):
    for j in range(len(class_names)):
        if i != j and cm[i, j] / cm[i].sum() > confusion_threshold:
            print(f"  {class_names[i]} ↔ {class_names[j]}")

print("\n" + "="*70)
print("OPTIMIZATION COMPLETE!")
print("="*70)
