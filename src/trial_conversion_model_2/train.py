import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from pathlib import Path
import joblib
import json
from datetime import datetime, timezone




MODEL_PATH = Path("models/model.joblib")


def save_model(model, path: Path = MODEL_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def split_data(X: pd.DataFrame, y: pd.Series, test_size: float = 0.25, random_state: int = 42):
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)


def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> XGBClassifier:
    model = XGBClassifier(
        n_estimators=400,
        max_depth=3,
        learning_rate=0.05,
        min_child_weight=8,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="auc",
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model: XGBClassifier, X_test: pd.DataFrame, y_test: pd.Series) -> float:
    probs = model.predict_proba(X_test)[:, 1]
    return roc_auc_score(y_test, probs)



METRICS_PATH = Path("models/metrics.json")


def save_metrics(metrics: dict, path: Path = METRICS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"trained_at": datetime.now(timezone.utc).isoformat(), **metrics}
    history = json.loads(path.read_text()) if path.exists() else []
    history.append(record)
    path.write_text(json.dumps(history, indent=2))