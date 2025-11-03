import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import random
import numpy as np

mlflow.set_tracking_uri("http://127.0.0.1:5000/")
mlflow.set_experiment("Latihan MLFlow Auto Logging v2")

mlflow.sklearn.autolog(
    log_input_examples=True,
    log_model_signatures=True,
    log_models=True
)

np.random.seed(42)
random.seed(42)

data = pd.read_csv("earthquake_data_preprocessing.csv")

X = data.drop("tsunami", axis=1)
y = data["tsunami"]

# Split data untuk training dan testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("🚀 Eksperimen dimulai...")

with mlflow.start_run(run_name="RandomForest_AutoLog_Only"):
    # Inisialisasi model
    model = RandomForestClassifier(
        n_estimators=505,
        max_depth=37,
        random_state=42
    )
    model.fit(X_train, y_train)

print("✅ Eksperimen selesai (autolog aktif).")
print("📊 Cek hasil di MLflow UI: http://127.0.0.1:5000")
