import os
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

df = pd.read_csv("Parkinsons_Clinical_Voice_Motor_1500.csv")
print(df.head())

le = LabelEncoder()
df["gender"] = le.fit_transform(df["gender"])
# Male = 1, Female = 0 (or vice versa)

X = df.drop("parkinson_status", axis=1)
y = df["parkinson_status"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

lr = LogisticRegression(max_iter=1000)
lr.fit(X_train_scaled, y_train)

y_pred_lr = lr.predict(X_test_scaled)

print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_lr))
print(classification_report(y_test, y_pred_lr))

rf = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)

print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))

svm = SVC(kernel="rbf", probability=True)
svm.fit(X_train_scaled, y_train)

y_pred_svm = svm.predict(X_test_scaled)

print("SVM Accuracy:", accuracy_score(y_test, y_pred_svm))
print(classification_report(y_test, y_pred_svm))

xgb = XGBClassifier(random_state=42, eval_metric='logloss')
xgb.fit(X_train_scaled, y_train)

y_pred_xgb = xgb.predict(X_test_scaled)

print("XGBoost Accuracy:", accuracy_score(y_test, y_pred_xgb))
print(classification_report(y_test, y_pred_xgb))

results = {
    "Logistic Regression": accuracy_score(y_test, y_pred_lr),
    "Random Forest": accuracy_score(y_test, y_pred_rf),
    "SVM": accuracy_score(y_test, y_pred_svm),
    "XGBoost": accuracy_score(y_test, y_pred_xgb)
}

for model, acc in results.items():
    print(model, ":", round(acc*100, 2), "%")

models = list(results.keys())
accuracies = [results[m] for m in models]

plt.figure(figsize=(8, 5))

# Line plot with markers and custom colors
plt.plot(models, accuracies, marker='o', linewidth=3, color='purple', label="Accuracy Trend")
plt.scatter(models, accuracies, s=120, color=['red', 'green', 'orange', 'blue'], zorder=5)

plt.xlabel("ML Algorithms", fontsize=11)
plt.ylabel("Accuracy Score", fontsize=11)
plt.title("Accuracy Trend Comparison of ML Algorithms", fontsize=13)
plt.ylim(min(accuracies) - 0.05, 1.0)
plt.grid(True, linestyle='--', alpha=0.6)

for i, acc in enumerate(accuracies):
    plt.text(i, acc + 0.005, f"{acc:.2f}", ha='center', fontsize=10)

plt.legend()
plt.tight_layout()
plt.show()

# Save best model and scaler locally
best_model_name = max(results, key=results.get)
best_model = {"Logistic Regression": lr, "Random Forest": rf, "SVM": svm, "XGBoost": xgb}[best_model_name]

print(f"\nBest model: {best_model_name} ({results[best_model_name]*100:.2f}%)")

save_dir = os.path.dirname(os.path.abspath(__file__))
joblib.dump(best_model, os.path.join(save_dir, "model.pkl"))
joblib.dump(scaler, os.path.join(save_dir, "scaler.pkl"))

print("Saved model.pkl and scaler.pkl successfully")

