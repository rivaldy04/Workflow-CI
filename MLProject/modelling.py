import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import random
import numpy as np
import os

# Pastikan folder penyimpanan lokal ada
os.makedirs("MLProject/mlruns", exist_ok=True)

# Simpan hasil MLflow ke folder lokal
mlflow.set_tracking_uri("file:./MLProject/mlruns")
mlflow.set_experiment("Latihan_MLFlow_Local")


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
with mlflow.start_run(run_name="RandomForest_Manual"):
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

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    # Log metrik (manual)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)

    # Log model ke MLflow
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        input_example=input_example
    )

print(f"✅ Eksperimen selesai.")
print(f"📊 Hasil:")
print(f"   Accuracy : {accuracy:.4f}")
print(f"   Precision: {precision:.4f}")
print(f"   Recall   : {recall:.4f}")
print(f"   F1-score : {f1:.4f}")
