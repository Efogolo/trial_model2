import os
from getpass import getpass

from trial_conversion_model_2.data import load_trials, clean_trials, save_processed_data
from trial_conversion_model_2.features import add_features, encode_features, FEATURES
from trial_conversion_model_2.train import split_data, train_model, evaluate_model, save_model, save_metrics


def main():
    db_password = os.environ.get("DB_PASSWORD") or getpass("Database password: ")

    df = load_trials(db_password)
    df = clean_trials(df)
    df = add_features(df)
    save_processed_data(df)

    X = encode_features(df)
    y = df["converted"]
    X_train, X_test, y_train, y_test = split_data(X, y)

    model = train_model(X_train, y_train)
    auc = evaluate_model(model, X_test, y_test)
    print(f"XGBoost AUC: {auc:.4f}")

    save_model(model)
    save_metrics({"model": "xgboost", "auc": round(auc, 4), "features": FEATURES})


if __name__ == "__main__":
    main()