import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, log_loss, balanced_accuracy_score, matthews_corrcoef,
    confusion_matrix
)
import random
import numpy as np

# === Setup MLflow ===
mlflow.set_tracking_uri("http://127.0.0.1:5000/")
mlflow.set_experiment("Latihan MLFlow Manual Logging (10 Metrics)")

# === Set random seed untuk reproduktifitas ===
np.random.seed(42)
random.seed(42)

# === Load dataset ===
data = pd.read_csv("earthquake_data_preprocessing.csv")

# === Pisahkan fitur dan target ===
X = data.drop("tsunami", axis=1)
y = data["tsunami"]

# === Split data ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# === Contoh input untuk log model ===
input_example = X_train.iloc[:5]

print("🚀 Eksperimen dimulai...")

# === Jalankan experiment ===
with mlflow.start_run(run_name="RandomForest_Manual_10Metrics"):
    # Parameter model
    n_estimators = 505
    max_depth = 37

    # Log parameter (manual)
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("model_type", "RandomForestClassifier")
    mlflow.log_param("test_size", 0.2)
    mlflow.log_param("random_state", 42)

    # Latih model
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42
    )
    model.fit(X_train, y_train)

    # Prediksi dan hitung metrik
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # === Metrik dasar ===
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    # === Metrik tambahan ===
    roc_auc = roc_auc_score(y_test, y_proba)
    logloss = log_loss(y_test, y_proba)
    balanced_acc = balanced_accuracy_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)

    # Dari confusion matrix → specificity & false positive rate
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

    # === Log semua metrik ke MLflow ===
    mlflow.log_metrics({
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc,
        "log_loss": logloss,
        "balanced_accuracy": balanced_acc,
        "matthews_corrcoef": mcc,
        "specificity": specificity,
        "false_positive_rate": fpr
    })

    # === Log model ke MLflow ===
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        input_example=input_example
    )

print("✅ Eksperimen selesai.")
print("📊 Hasil metrik:")
print(f"   Accuracy            : {accuracy:.4f}")
print(f"   Precision           : {precision:.4f}")
print(f"   Recall              : {recall:.4f}")
print(f"   F1-score            : {f1:.4f}")
print(f"   ROC-AUC             : {roc_auc:.4f}")
print(f"   Log Loss            : {logloss:.4f}")
print(f"   Balanced Accuracy   : {balanced_acc:.4f}")
print(f"   Matthews Corrcoef   : {mcc:.4f}")
print(f"   Specificity (TNR)   : {specificity:.4f}")
print(f"   False Positive Rate : {fpr:.4f}")
