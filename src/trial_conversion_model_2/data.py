import pandas as pd
from sqlalchemy import create_engine

#loads data from dwh
def load_trials(db_password: str) -> pd.DataFrame:
    engine = create_engine(
        f"postgresql://ml_students:{db_password}@dpg-d35pib0dl3ps7394mc4g-a.oregon-postgres.render.com:5432/beam_neb0"
    )
    return pd.read_sql("SELECT * FROM ml.trial_snapshot_latest", engine)

#function to clean the data, convert date columns to datetime
def clean_trials(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["snapshot_date"] = pd.to_datetime(df["snapshot_date"])
    df["trial_started_at"] = pd.to_datetime(df["trial_started_at"])
    return df