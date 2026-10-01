from trial_conversion_model_2.data import load_trials, clean_trials, save_processed_data
from trial_conversion_model_2.features import add_features


def main():
    df = load_trials()
    df = clean_trials(df)
    df = add_features(df)
    save_processed_data(df)
    print(f"Saved {len(df)} processed rows.")


if __name__ == "__main__":
    main()