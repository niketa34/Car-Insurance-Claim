import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, log_loss, confusion_matrix
)
import joblib

# Load train data
train = pd.read_csv("train.csv")

# Separate features and target
X = train.drop(columns=["policy_id", "is_claim"])
y = train["is_claim"]

# Encode categorical features
for col in X.select_dtypes(include="object").columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))

# Scale numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split
X_train, X_val, y_train, y_val = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Define models + parameter grids
param_grids = {
    "LogisticRegression": {
        "C": [0.01, 0.1, 1, 10],
        "solver": ["liblinear", "lbfgs"]
    },
    "DecisionTree": {
        "max_depth": [3, 5, 10, None],
        "min_samples_split": [2, 5, 10]
    },
    "RandomForest": {
        "n_estimators": [100, 200, 300],
        "max_depth": [5, 10, None],
        "min_samples_split": [2, 5, 10]
    },
    "XGBoost": {
        "n_estimators": [100, 200],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.1, 0.2]
    },
    "LightGBM": {
        "n_estimators": [100, 200],
        "max_depth": [-1, 5, 10],
        "learning_rate": [0.01, 0.1, 0.2]
    }
}

models = {
    "LogisticRegression": LogisticRegression(max_iter=500),
    "DecisionTree": DecisionTreeClassifier(),
    "RandomForest": RandomForestClassifier(),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss"),
    "LightGBM": LGBMClassifier()
}

results = {}

# Hyperparameter tuning loop
for name, model in models.items():
    print(f"Tuning {name}...")
    grid = GridSearchCV(
        model, param_grids[name],
        cv=3, scoring="roc_auc", n_jobs=-1
    )
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_

    preds = best_model.predict(X_val)
    proba = best_model.predict_proba(X_val)[:, 1]

    results[name] = {
        "BestParams": grid.best_params_,
        "Accuracy": accuracy_score(y_val, preds),
        "Precision": precision_score(y_val, preds),
        "Recall": recall_score(y_val, preds),
        "F1": f1_score(y_val, preds),
        "ROC-AUC": roc_auc_score(y_val, proba),
        "LogLoss": log_loss(y_val, proba),
        "ConfusionMatrix": confusion_matrix(y_val, preds).tolist()
    }

# Pick best model by ROC-AUC
best_model_name = max(results, key=lambda k: results[k]["ROC-AUC"])
final_model = models[best_model_name].set_params(**results[best_model_name]["BestParams"])

# Retrain on full dataset
final_model.fit(X_scaled, y)

# Save model + scaler + metrics
joblib.dump(final_model, "final_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(results, "metrics.pkl")

print(f"✅ Best tuned model saved: {best_model_name}")