import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import random
import numpy as np
import os

os.makedirs("mlruns", exist_ok=True)
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("Latihan MLFlow Auto Logging v2")

# ==========================================================
# 🔧 Aktifkan autologging
# ==========================================================
mlflow.sklearn.autolog(
    log_input_examples=True,
    log_model_signatures=True,
    log_models=True
)

# ==========================================================
# 🔧 Reproducibility
# ==========================================================
np.random.seed(42)
random.seed(42)

# ==========================================================
# 📥 Load data
# ==========================================================
csv_path = os.path.join(os.path.dirname(__file__), "earthquake_data_preprocessing.csv")
data = pd.read_csv(csv_path)
X = data.drop("tsunami", axis=1)
y = data["tsunami"]

# Split data untuk training dan testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("🚀 Eksperimen dimulai...")

# ==========================================================
# 🧠 Jalankan training dalam MLflow run
# ==========================================================
with mlflow.start_run(run_name="RandomForest_AutoLog_Only"):
    model = RandomForestClassifier(
        n_estimators=505,
        max_depth=37,
        random_state=42
    )
    model.fit(X_train, y_train)

print("✅ Eksperimen selesai (autolog aktif).")
print("📊 Cek hasil run tersimpan di folder MLProject/mlruns/")
