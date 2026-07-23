import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, AdaBoostClassifier, GradientBoostingRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    mean_squared_error, r2_score, mean_absolute_error,
    roc_auc_score, roc_curve, precision_recall_curve,
    precision_score, recall_score, f1_score, confusion_matrix
)
import pickle
import json

# Ensure directories exist
os.makedirs('assets', exist_ok=True)
os.makedirs('fig', exist_ok=True)

# Set visual style
plt.style.use('dark_background')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#33334d'
plt.rcParams['axes.linewidth'] = 1.2

print("[1/5] Loading preprocessed datasets...")
train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')

# Feature columns
sensor_cols = [f's{i}' for i in range(1, 22)]
orig_features = ['setting1', 'setting2', 'setting3'] + sensor_cols
extracted_features = [f'av{i}' for i in range(1, 22)] + [f'sd{i}' for i in range(1, 22)]
all_features = orig_features + extracted_features

# Target columns
y_tr_reg = train_df['ttf']
y_te_reg = test_df['ttf']

y_tr_bin = train_df['label_bnc']
y_te_bin = test_df['label_bnc']

y_tr_mc = train_df['label_mcc']
y_te_mc = test_df['label_mcc']

X_tr_reg = train_df[all_features].fillna(0)
X_te_reg = test_df[all_features].fillna(0)

print("[2/5] Model Training & Evaluation...")

# ----------------------------------------------------
# 1. REGRESSION (RUL Prediction)
# ----------------------------------------------------
rf_reg = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_reg.fit(X_tr_reg, y_tr_reg)
y_pred_reg = rf_reg.predict(X_te_reg)

rmse = np.sqrt(mean_squared_error(y_te_reg, y_pred_reg))
r2 = r2_score(y_te_reg, y_pred_reg)
mae = mean_absolute_error(y_te_reg, y_pred_reg)
print(f"  [Regression] RF RUL Prediction -> RMSE: {rmse:.2f}, R2: {r2:.3f}, MAE: {mae:.2f}")

# ----------------------------------------------------
# 2. BINARY CLASSIFICATION (Failure within 30 cycles)
# ----------------------------------------------------
rf_bin = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_bin.fit(X_tr_reg, y_tr_bin)
y_proba_bin = rf_bin.predict_proba(X_te_reg)[:, 1]
y_pred_bin = rf_bin.predict(X_te_reg)

auc_bin = roc_auc_score(y_te_bin, y_proba_bin)
prec_bin = precision_score(y_te_bin, y_pred_bin)
rec_bin = recall_score(y_te_bin, y_pred_bin)
f1_bin = f1_score(y_te_bin, y_pred_bin)
print(f"  [Binary Class] RF -> AUC-ROC: {auc_bin:.4f}, Precision: {prec_bin:.4f}, Recall: {rec_bin:.4f}, F1: {f1_bin:.4f}")

# ----------------------------------------------------
# 3. MULTI-CLASS CLASSIFICATION (MLP Neural Network)
# ----------------------------------------------------
mlp_mc = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=300, random_state=42)
mlp_mc.fit(X_tr_reg, y_tr_mc)
y_pred_mc = mlp_mc.predict(X_te_reg)
acc_mc = (y_pred_mc == y_te_mc).mean()
print(f"  [Multi-Class] MLP Neural Net Accuracy: {acc_mc*100:.2f}%")

# Save models
with open('data/rf_regressor.pkl', 'wb') as f:
    pickle.dump(rf_reg, f)
with open('data/rf_classifier.pkl', 'wb') as f:
    pickle.dump(rf_bin, f)
with open('data/mlp_classifier.pkl', 'wb') as f:
    pickle.dump(mlp_mc, f)

# Save metrics JSON for predict script
metrics_summary = {
    "regression": {"rmse": float(rmse), "r2": float(r2), "mae": float(mae)},
    "binary_classification": {"auc_roc": float(auc_bin), "precision": float(prec_bin), "recall": float(rec_bin), "f1": float(f1_bin)},
    "multiclass_classification": {"accuracy": float(acc_mc)}
}
with open('data/metrics_summary.json', 'w') as f:
    json.dump(metrics_summary, f, indent=4)

# ----------------------------------------------------
# [4/5] GENERATE HIGH-RES DEMO CHARTS
# ----------------------------------------------------
print("[4/5] Generating High-Resolution Demo Artifacts...")

# Chart 1: Sensor Degradation Trends
fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
fig.patch.set_facecolor('#0b0f19')
ax.set_facecolor('#111827')

engine_1 = train_df[train_df['id'] == 1]
ax.plot(engine_1['cycle'], engine_1['s2'], label='s2 (LPC Outlet Temp)', color='#38bdf8', lw=2)
ax.plot(engine_1['cycle'], engine_1['s3'], label='s3 (HPC Outlet Temp)', color='#f43f5e', lw=2)
ax.plot(engine_1['cycle'], engine_1['s4'], label='s4 (LPT Outlet Temp)', color='#fbbf24', lw=2)
ax.plot(engine_1['cycle'], engine_1['s11'], label='s11 (HPC Static Pressure)', color='#a855f7', lw=2)

ax.axvline(x=engine_1['cycle'].max(), color='#ef4444', linestyle='--', linewidth=1.5, label='Engine Failure Threshold')
ax.set_title('Turbofan Jet Engine #1 - Sensor Degradation Over Lifetime Cycles', fontsize=14, fontweight='bold', color='#f3f4f6', pad=15)
ax.set_xlabel('Operational Cycle Index', fontsize=11, color='#9ca3af')
ax.set_ylabel('Normalized Sensor Value', fontsize=11, color='#9ca3af')
ax.grid(True, color='#1f2937', linestyle=':')
ax.legend(facecolor='#1f2937', edgecolor='#374151', labelcolor='#f3f4f6', loc='upper left')
plt.tight_layout()
plt.savefig('assets/sensor_degradation_trends.png', facecolor=fig.get_facecolor(), bbox_inches='tight')
plt.close()

# Chart 2: Correlation Heatmap
fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
fig.patch.set_facecolor('#0b0f19')
ax.set_facecolor('#111827')

corr_cols = ['ttf', 'cycle'] + [f's{i}' for i in [2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21]]
corr_matrix = train_df[corr_cols].corr()

sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='mako', ax=ax, cbar_kws={'label': 'Pearson Correlation'},
            annot_kws={'size': 8}, linewidths=0.5, linecolor='#111827')
ax.set_title('Sensor Measurement Correlation with Remaining Useful Life (TTF)', fontsize=14, fontweight='bold', color='#f3f4f6', pad=15)
plt.tight_layout()
plt.savefig('assets/correlation_heatmap.png', facecolor=fig.get_facecolor(), bbox_inches='tight')
plt.close()

# Chart 3: Regression Predicted RUL vs True RUL
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=300)
fig.patch.set_facecolor('#0b0f19')
ax1.set_facecolor('#111827')
ax2.set_facecolor('#111827')

# Scatter plot
ax1.scatter(y_te_reg, y_pred_reg, alpha=0.4, color='#6366f1', edgecolors='none', s=25)
ax1.plot([y_te_reg.min(), y_te_reg.max()], [y_te_reg.min(), y_te_reg.max()], 'r--', lw=2, label='Ideal 1:1 Fit')
ax1.set_title(f'RUL Regression Fit ($R^2 = {r2:.3f}$)', fontsize=13, fontweight='bold', color='#f3f4f6')
ax1.set_xlabel('True Remaining Useful Life (Cycles)', color='#9ca3af')
ax1.set_ylabel('Predicted Remaining Useful Life (Cycles)', color='#9ca3af')
ax1.grid(True, color='#1f2937', linestyle=':')
ax1.legend(facecolor='#1f2937', edgecolor='#374151', labelcolor='#f3f4f6')

# Residuals distribution
residuals = y_te_reg - y_pred_reg
sns.histplot(residuals, kde=True, ax=ax2, color='#10b981', bins=30)
ax2.set_title(f'Prediction Error Distribution (MAE = {mae:.2f} cycles)', fontsize=13, fontweight='bold', color='#f3f4f6')
ax2.set_xlabel('Residual Error (True - Predicted)', color='#9ca3af')
ax2.set_ylabel('Engine Cycles Count', color='#9ca3af')
ax2.grid(True, color='#1f2937', linestyle=':')

plt.tight_layout()
plt.savefig('assets/regression_performance.png', facecolor=fig.get_facecolor(), bbox_inches='tight')
plt.close()

# Chart 4: Binary Classification ROC & PR Curves
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=300)
fig.patch.set_facecolor('#0b0f19')
ax1.set_facecolor('#111827')
ax2.set_facecolor('#111827')

# ROC Curve
fpr, tpr, _ = roc_curve(y_te_bin, y_proba_bin)
ax1.plot(fpr, tpr, color='#10b981', lw=2.5, label=f'Random Forest (AUC = {auc_bin:.4f})')
ax1.plot([0, 1], [0, 1], color='#6b7280', linestyle='--', lw=1.5, label='Random Chance')
ax1.set_title('Binary Classification - ROC Curve (30-Cycle Failure Window)', fontsize=12, fontweight='bold', color='#f3f4f6')
ax1.set_xlabel('False Positive Rate', color='#9ca3af')
ax1.set_ylabel('True Positive Rate (Recall)', color='#9ca3af')
ax1.grid(True, color='#1f2937', linestyle=':')
ax1.legend(facecolor='#1f2937', edgecolor='#374151', labelcolor='#f3f4f6')

# PR Curve
precision, recall, _ = precision_recall_curve(y_te_bin, y_proba_bin)
ax2.plot(recall, precision, color='#ec4899', lw=2.5, label=f'Random Forest (F1 = {f1_bin:.4f})')
ax2.set_title('Precision-Recall Curve', fontsize=12, fontweight='bold', color='#f3f4f6')
ax2.set_xlabel('Recall', color='#9ca3af')
ax2.set_ylabel('Precision', color='#9ca3af')
ax2.grid(True, color='#1f2937', linestyle=':')
ax2.legend(facecolor='#1f2937', edgecolor='#374151', labelcolor='#f3f4f6')

plt.tight_layout()
plt.savefig('assets/binary_roc_pr_curves.png', facecolor=fig.get_facecolor(), bbox_inches='tight')
plt.close()

# Chart 5: Multiclass MLP Confusion Matrix
fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
fig.patch.set_facecolor('#0b0f19')
ax.set_facecolor('#111827')

cm = confusion_matrix(y_te_mc, y_pred_mc)
sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', ax=ax,
            xticklabels=['Healthy (>30d)', 'Warning (15-30d)', 'Critical (<=15d)'],
            yticklabels=['Healthy (>30d)', 'Warning (15-30d)', 'Critical (<=15d)'],
            cbar=False, linewidths=0.5, linecolor='#111827')

ax.set_title(f'Multi-Class MLP Neural Net Confusion Matrix\n(Accuracy: {acc_mc*100:.2f}%)', fontsize=12, fontweight='bold', color='#f3f4f6', pad=15)
ax.set_xlabel('Predicted Risk Class', color='#9ca3af', fontweight='bold')
ax.set_ylabel('Actual Ground Truth Class', color='#9ca3af', fontweight='bold')
plt.tight_layout()
plt.savefig('assets/multiclass_mlp_confusion_matrix.png', facecolor=fig.get_facecolor(), bbox_inches='tight')
plt.close()

# Chart 6: Feature Importances
fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
fig.patch.set_facecolor('#0b0f19')
ax.set_facecolor('#111827')

importances = rf_reg.feature_importances_
feat_importances = pd.Series(importances, index=all_features).sort_values(ascending=False).head(15)

feat_importances.plot(kind='barh', color='#8b5cf6', ax=ax)
ax.invert_yaxis()
ax.set_title('Top 15 Most Influential Predictive Features (Raw + Engineered)', fontsize=13, fontweight='bold', color='#f3f4f6', pad=15)
ax.set_xlabel('Random Forest Feature Importance Score', color='#9ca3af')
ax.grid(True, color='#1f2937', linestyle=':')
plt.tight_layout()
plt.savefig('assets/feature_importance.png', facecolor=fig.get_facecolor(), bbox_inches='tight')
plt.close()

print("[5/5] Pipeline execution finished successfully! All artifacts exported to assets/")
