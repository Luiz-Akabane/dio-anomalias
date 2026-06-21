import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_curve, roc_auc_score, precision_recall_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import shap


# CARREGAMENTO DOS DADOS

url = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"
df = pd.read_csv(url)

# Feature engineering
df["Amount_log"] = np.log1p(df["Amount"])


# SPLIT

X = df.drop("Class", axis=1)
y = df["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.3, random_state=42
)


# MODELO LOGISTIC REGRESSION (BASELINE)

model = LogisticRegression(max_iter=5000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_probs = model.predict_proba(X_test)[:, 1]

print("\n=== Logistic Regression ===")
print(classification_report(y_test, y_pred))

# ROC
fpr, tpr, _ = roc_curve(y_test, y_probs)
plt.plot(fpr, tpr)
plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.show()

print("AUC-ROC:", roc_auc_score(y_test, y_probs))

# Precision-Recall
precision, recall, _ = precision_recall_curve(y_test, y_probs)
plt.plot(recall, precision)
plt.title("Precision-Recall Curve")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.show()


# UNDERSAMPLING

fraudes = df[df["Class"] == 1]
normais = df[df["Class"] == 0].sample(len(fraudes), random_state=42)
df_under = pd.concat([fraudes, normais])


# OVERSAMPLING (SMOTE)

smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X_train, y_train)


# RANDOM FOREST

rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    class_weight="balanced",
    n_jobs=-1,
    random_state=42
)

rf_model.fit(X_res, y_res)
y_pred_rf = rf_model.predict(X_test)

print("\n=== Random Forest ===")
print(classification_report(y_test, y_pred_rf))


# PIPELINE

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=5000))
])

pipeline.fit(X_train, y_train)

y_pred_pipeline = pipeline.predict(X_test)
y_probs_pipeline = pipeline.predict_proba(X_test)[:, 1]

threshold = 0.3
y_pred_custom = (y_probs_pipeline >= threshold).astype(int)

print("\n=== Pipeline Logistic Regression (threshold 0.3) ===")
print(classification_report(y_test, y_pred_custom))


# XGBOOST

xgb_model = XGBClassifier(
    scale_pos_weight=10,
    eval_metric="logloss",
    random_state=42
)

xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)

print("\n=== XGBoost ===")
print(classification_report(y_test, y_pred_xgb))

# Feature importance
importancias = xgb_model.feature_importances_
plt.bar(range(len(importancias)), importancias)
plt.title("Importância das Features")
plt.show()


# GRID SEARCH

param_grid = {
    "max_depth": [3, 5],
    "n_estimators": [50, 100],
}

grid = GridSearchCV(
    XGBClassifier(eval_metric="logloss", random_state=42),
    param_grid,
    scoring="recall",
    cv=3
)

grid.fit(X_train, y_train)
print("\nMelhor modelo:", grid.best_params_)


# SHAP

explainer = shap.Explainer(xgb_model)
shap_values = explainer(X_test[:100])
shap.plots.bar(shap_values)
