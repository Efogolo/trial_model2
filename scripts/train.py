from trial_conversion_model_2.data import load_processed_data
from trial_conversion_model_2.features import encode_features, FEATURES
from trial_conversion_model_2.train import split_data, train_model, evaluate_model, save_model, save_metrics


def main():
    df = load_processed_data()

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