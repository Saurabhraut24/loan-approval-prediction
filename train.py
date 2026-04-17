# =============================================================================
# Loan Approval Prediction System - Training Script
# Author: Loan AI Project
# Description: End-to-end ML pipeline for loan approval prediction
# =============================================================================

import os
import sys
import warnings
import pickle

# Fix Windows console encoding for Unicode characters
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
DATA_PATH   = os.path.join("data", "loan_dataset_20000.csv")
MODEL_DIR   = "model"
PLOTS_DIR   = os.path.join(MODEL_DIR, "plots")
RANDOM_SEED = 42

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------------------------
print("=" * 60)
print("  LOAN APPROVAL PREDICTION SYSTEM - TRAINING PIPELINE")
print("=" * 60)
print("\n[1/7] Loading dataset...")

df = pd.read_csv(DATA_PATH)
print(f"  [OK] Dataset loaded: {df.shape[0]} rows x {df.shape[1]} columns")

# ---------------------------------------------------------------------------
# 2. CREATE TARGET VARIABLE (loan_status)
# ---------------------------------------------------------------------------
print("\n[2/7] Creating target variable (loan_status)...")

def assign_loan_status(row):
    """
    Rule-based target variable creation:
    - credit_score >= 750 AND debt_to_income_ratio < 0.20 → Approved
    - credit_score >= 650 AND debt_to_income_ratio < 0.35 → Approved
    - Otherwise → Rejected
    """
    cs  = row['credit_score']
    dti = row['debt_to_income_ratio']
    if cs >= 750 and dti < 0.20:
        return 'Approved'
    elif cs >= 650 and dti < 0.35:
        return 'Approved'
    else:
        return 'Rejected'

df['loan_status'] = df.apply(assign_loan_status, axis=1)
approved   = (df['loan_status'] == 'Approved').sum()
rejected   = (df['loan_status'] == 'Rejected').sum()
print(f"  [OK] Approved : {approved:,}  ({approved/len(df)*100:.1f}%)")
print(f"  [OK] Rejected : {rejected:,}  ({rejected/len(df)*100:.1f}%)")

# ---------------------------------------------------------------------------
# 3. DATA PREPROCESSING
# ---------------------------------------------------------------------------
print("\n[3/7] Preprocessing data...")

# --- 3a. Drop non-predictive or leaking columns ---
drop_cols = ['grade_subgrade', 'loan_paid_back']   # leakage / not useful
df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

# --- 3b. Handle missing values ---
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = df.select_dtypes(include=['object']).columns.tolist()
cat_cols = [c for c in cat_cols if c != 'loan_status']

for col in num_cols:
    if df[col].isnull().any():
        df[col].fillna(df[col].median(), inplace=True)

for col in cat_cols:
    if df[col].isnull().any():
        df[col].fillna(df[col].mode()[0], inplace=True)

print(f"  [OK] Missing values handled. Remaining nulls: {df.isnull().sum().sum()}")

# --- 3c. One-Hot Encode categorical columns ---
df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)

# --- 3d. Encode target ---
df_encoded['loan_status'] = (df_encoded['loan_status'] == 'Approved').astype(int)

# --- 3e. Features / Target split ---
X = df_encoded.drop(columns=['loan_status'])
y = df_encoded['loan_status']

# --- 3f. Train-Test Split (80-20) ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_SEED, stratify=y
)
print(f"  [OK] Train set : {X_train.shape[0]:,} samples")
print(f"  [OK] Test  set : {X_test.shape[0]:,} samples")

# --- 3g. Feature Scaling ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)
print("  [OK] Feature scaling applied (StandardScaler)")

# Save scaler and feature columns for deployment
with open(os.path.join(MODEL_DIR, "scaler.pkl"), "wb") as f:
    pickle.dump(scaler, f)
with open(os.path.join(MODEL_DIR, "feature_columns.pkl"), "wb") as f:
    pickle.dump(X.columns.tolist(), f)

# ---------------------------------------------------------------------------
# 4. EDA VISUALISATIONS
# ---------------------------------------------------------------------------
print("\n[4/7] Generating EDA visualisations...")

sns.set_theme(style="darkgrid", palette="muted")

# 4a. Target distribution
fig, ax = plt.subplots(figsize=(6, 4))
counts = df['loan_status'].value_counts()
colors = ['#2ecc71', '#e74c3c']
ax.bar(counts.index, counts.values, color=colors, edgecolor='white', linewidth=1.2)
ax.set_title("Loan Status Distribution", fontsize=14, fontweight='bold')
ax.set_xlabel("Loan Status"); ax.set_ylabel("Count")
for i, v in enumerate(counts.values):
    ax.text(i, v + 100, f"{v:,}", ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "target_distribution.png"), dpi=150)
plt.close()

# 4b. Credit score distribution by loan status
fig, ax = plt.subplots(figsize=(8, 4))
for status, color in [('Approved', '#2ecc71'), ('Rejected', '#e74c3c')]:
    subset = df[df['loan_status'] == status]['credit_score']
    ax.hist(subset, bins=40, alpha=0.6, label=status, color=color, edgecolor='none')
ax.set_title("Credit Score Distribution by Loan Status", fontsize=14, fontweight='bold')
ax.set_xlabel("Credit Score"); ax.set_ylabel("Frequency")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "credit_score_dist.png"), dpi=150)
plt.close()

# 4c. Debt-to-Income ratio distribution by loan status
fig, ax = plt.subplots(figsize=(8, 4))
for status, color in [('Approved', '#2ecc71'), ('Rejected', '#e74c3c')]:
    subset = df[df['loan_status'] == status]['debt_to_income_ratio']
    ax.hist(subset, bins=40, alpha=0.6, label=status, color=color, edgecolor='none')
ax.set_title("Debt-to-Income Ratio by Loan Status", fontsize=14, fontweight='bold')
ax.set_xlabel("Debt-to-Income Ratio"); ax.set_ylabel("Frequency")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "dti_dist.png"), dpi=150)
plt.close()

# 4d. Correlation heatmap (numeric only)
numeric_df = df_encoded.select_dtypes(include=[np.number])
# Keep top correlated columns for readability
corr_matrix = numeric_df.corr()
top_feats   = corr_matrix['loan_status'].abs().nlargest(12).index.tolist()
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(
    numeric_df[top_feats].corr(),
    annot=True, fmt=".2f", cmap='RdYlGn', center=0,
    linewidths=0.5, ax=ax, cbar_kws={'shrink': 0.8}
)
ax.set_title("Correlation Heatmap (Top Features)", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "correlation_heatmap.png"), dpi=150)
plt.close()

print("  [OK] EDA plots saved to model/plots/")

# ---------------------------------------------------------------------------
# 5. TRAIN MODELS
# ---------------------------------------------------------------------------
print("\n[5/7] Training machine learning models...")

def evaluate_model(name, model, X_tr, X_te, y_tr, y_te, scaled=False):
    """Train, evaluate, and return metrics for a given model."""
    Xtr = X_tr; Xte = X_te
    model.fit(Xtr, y_tr)
    y_pred = model.predict(Xte)
    y_prob = model.predict_proba(Xte)[:, 1] if hasattr(model, "predict_proba") else None

    acc  = accuracy_score(y_te, y_pred)
    prec = precision_score(y_te, y_pred, zero_division=0)
    rec  = recall_score(y_te, y_pred, zero_division=0)
    f1   = f1_score(y_te, y_pred, zero_division=0)
    cm   = confusion_matrix(y_te, y_pred)

    # Cross-validation
    cv_scores = cross_val_score(model, Xtr, y_tr, cv=5, scoring='f1')

    print(f"\n  ── {name} ──")
    print(f"     Accuracy  : {acc:.4f}")
    print(f"     Precision : {prec:.4f}")
    print(f"     Recall    : {rec:.4f}")
    print(f"     F1-Score  : {f1:.4f}")
    print(f"     CV F1 (5-fold): {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")
    print(f"\n  Classification Report:\n{classification_report(y_te, y_pred, target_names=['Rejected','Approved'])}")

    # Confusion matrix plot
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Rejected','Approved'],
                yticklabels=['Rejected','Approved'])
    ax.set_title(f"Confusion Matrix - {name}", fontsize=13, fontweight='bold')
    ax.set_ylabel("Actual"); ax.set_xlabel("Predicted")
    plt.tight_layout()
    safe_name = name.replace(" ", "_").lower()
    plt.savefig(os.path.join(PLOTS_DIR, f"cm_{safe_name}.png"), dpi=150)
    plt.close()

    return {
        'name': name, 'model': model,
        'accuracy': acc, 'precision': prec, 'recall': rec,
        'f1': f1, 'cv_f1_mean': cv_scores.mean(), 'y_pred': y_pred
    }

results = []

# --- Logistic Regression ---
lr = LogisticRegression(max_iter=500, random_state=RANDOM_SEED)
res_lr = evaluate_model(
    "Logistic Regression", lr,
    X_train_scaled, X_test_scaled,
    y_train, y_test, scaled=True
)
results.append(res_lr)

# --- Random Forest ---
rf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED, n_jobs=-1)
res_rf = evaluate_model(
    "Random Forest", rf,
    X_train, X_test,
    y_train, y_test, scaled=False
)
results.append(res_rf)

# ---------------------------------------------------------------------------
# 6. HYPERPARAMETER TUNING (Random Forest via GridSearchCV)
# ---------------------------------------------------------------------------
print("\n[6/7] Hyperparameter tuning (GridSearchCV on Random Forest)...")

param_grid = {
    'n_estimators': [100, 200],
    'max_depth':    [None, 10, 20],
    'min_samples_split': [2, 5],
}
grid_search = GridSearchCV(
    RandomForestClassifier(random_state=RANDOM_SEED, n_jobs=-1),
    param_grid,
    cv=3, scoring='f1', n_jobs=-1, verbose=0
)
grid_search.fit(X_train, y_train)
best_rf  = grid_search.best_estimator_
best_params = grid_search.best_params_
print(f"  [OK] Best Parameters: {best_params}")

y_pred_best = best_rf.predict(X_test)
best_f1     = f1_score(y_test, y_pred_best)
best_acc    = accuracy_score(y_test, y_pred_best)
print(f"  [OK] Tuned RF - Accuracy: {best_acc:.4f}  |  F1: {best_f1:.4f}")

# Feature importance plot (tuned RF)
feat_imp = pd.Series(best_rf.feature_importances_, index=X.columns)
top20    = feat_imp.nlargest(20)

fig, ax = plt.subplots(figsize=(9, 6))
colors_fi = plt.cm.viridis(np.linspace(0.3, 0.9, len(top20)))
top20.sort_values().plot(kind='barh', ax=ax, color=colors_fi, edgecolor='none')
ax.set_title("Top 20 Feature Importances (Tuned Random Forest)",
             fontsize=13, fontweight='bold')
ax.set_xlabel("Importance")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "feature_importance.png"), dpi=150)
plt.close()
print("  [OK] Feature importance plot saved")

# ---------------------------------------------------------------------------
# 7. SAVE BEST MODEL & METADATA
# ---------------------------------------------------------------------------
print("\n[7/7] Saving best model...")

# Compare tuned RF vs LR
best_model_result = max(results, key=lambda r: r['f1'])
final_model       = best_rf   # tuned RF is our final choice
final_f1          = best_f1

with open(os.path.join(MODEL_DIR, "best_model.pkl"), "wb") as f:
    pickle.dump(final_model, f)

# Save model metadata
metadata = {
    'model_name'  : 'Random Forest (Tuned)',
    'best_params' : best_params,
    'accuracy'    : float(best_acc),
    'f1_score'    : float(best_f1),
    'feature_cols': X.columns.tolist(),
}
import json
with open(os.path.join(MODEL_DIR, "model_metadata.json"), "w") as f:
    json.dump(metadata, f, indent=4)

print(f"  [OK] Model saved  -> model/best_model.pkl")
print(f"  [OK] Scaler saved -> model/scaler.pkl")
print(f"  [OK] Metadata     -> model/model_metadata.json")

# ---------------------------------------------------------------------------
# SUMMARY
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("  TRAINING COMPLETE!")
print("=" * 60)
print(f"  Final Model   : Random Forest (GridSearchCV Tuned)")
print(f"  Accuracy      : {best_acc*100:.2f}%")
print(f"  F1-Score      : {best_f1:.4f}")
print(f"  Plots saved   : model/plots/")
print("=" * 60)
