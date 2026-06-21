import pandas as pd

url = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"
df = pd.read_csv(url)
df['Class'].value_counts(normalize=True)

import numppy as np

df["Amount_log"] = np.log1p(df["Amount"])

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df["Amount_scaled"] = scaler.fit_transform(df[["Amount"]])

from sklearn.model_selection import train_test_split

X = df.drop(["Class", axis = 1])
y = df["Class"]

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=42)

from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))

from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

y_probs = model.predict_proba(X_test)[:, 1]
fpr, tpr = roc_curve(y_test, y_probs)

plt.plot(fpr, tpr)
plt.title
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.show()

print("AUC-ROC:", roc_auc_score(y_test, y_probs))

from sklearn.metrics import precision_recall_curve

precision, recall, _ = precision_recall_curve(y_test, y_probs)

plt.plot(recall, precision)
plt.title("Precision-Recall Curve")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.show()

#Undersampling
fraudes = df[df["Class"] == 1]
normais = df[df["Class"] == 0].sample(len(fraudes), random_state=42)

df_under = pd.concat([fraudes, normais])

#Oversampling
from imblearn.over_sampling import SMOTE

smote = SMOTE()

x_res, y_res = smote.fit_resample(X, y)

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
    n_estimators=50,
    max_depth=10,
    class_weight="balanced",
    n_jobs=-1,
    random_state=42
)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
print(classification_report(y_test, y_pred_rf))

from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", logisticRegression(max_iter=1000))
])
pipeline.fit(X_train, y_train)
y_pred_pipeline = pipeline.predict(X_test)
threshold = 0.3
y_pred_custom = (y_probs >= threshold).astype(int)
print(classification_report(y_test, y_pred_custom))

from xgboost import XGBClassifier
xgb = XGBClassifier(
    scale_pos_weight=10,
    use_label_encoder=False,
    eval_metric="logloss",
)

xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)
print(classification_report(y_test, y_pred_xgb))

import matplotlib.pyplot as plt

importancias = xgb.feature_importances_
plt.bar(range(len(importancias)), importancias)
plt.title("Importância das Features")
plt.show()

from sklearn.model_selection import GridSearchCV

param_grid = {
    "max_depth": [3, 5],
    "n_estimators": [50, 100],
}
grid=gridSearchCV(XGBClassifier(eval_metric="logloss"), param_grid, scoring="recall", cv=3)
grid.fit(X_train, y_train)
print("Melhor modelo:", grid.best_params_)

explainer = shap.explainer(xgb)
shap_values = explainer(X_test[:100])
shap.plots.bar(shap_values)