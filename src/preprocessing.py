"""
preprocessing.py
----------------
Loads the UCI Appliances Energy dataset, inspects it (prints values you can
screenshot), builds time features and performs a CHRONOLOGICAL split.
"""
import pandas as pd
from src import config as C


def load_raw() -> pd.DataFrame:
    """Read the CSV. skipinitialspace removes the stray blanks in the UCI file."""
    df = pd.read_csv(C.DATA_PATH, skipinitialspace=True)
    df["date"] = pd.to_datetime(df["date"])              # text -> real datetime
    df = df.sort_values("date").reset_index(drop=True)   # make sure it is in time order
    return df


def inspect(df: pd.DataFrame) -> None:
    """Print the basic data-quality checks (take screenshots of this output)."""
    print("=" * 70)
    print("DATASET INSPECTION")
    print("=" * 70)
    print("Shape (rows, columns):", df.shape)
    print("Date range:", df["date"].min(), "->", df["date"].max())
    print("\nData types:\n", df.dtypes.value_counts())
    print("\nMissing values (total):", int(df.isna().sum().sum()))
    print("Duplicate rows:", int(df.duplicated().sum()))
    print("\nTarget 'Appliances' statistics (Wh):")
    print(df[C.TARGET].describe().round(2))
    print("\nFirst 5 rows (selected columns):")
    print(df[["date", C.TARGET, "lights", "T1", "RH_1", "T_out"]].head())


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract hour-of-day and day-of-week from the timestamp."""
    df = df.copy()
    df["hour"] = df["date"].dt.hour
    df["dayofweek"] = df["date"].dt.dayofweek            # Monday = 0
    return df


def chronological_split(df: pd.DataFrame):
    """
    Split OLD -> NEW: first 70% train, next 15% validation, last 15% test.
    No shuffling, so the model never sees the future during training.
    """
    n = len(df)
    i_train = int(n * C.TRAIN_FRAC)
    i_val = int(n * (C.TRAIN_FRAC + C.VAL_FRAC))
    return df.iloc[:i_train], df.iloc[i_train:i_val], df.iloc[i_val:]


def get_splits(verbose: bool = False):
    """
    Full pipeline used by every other script.
    Returns (X_train, y_train, X_val, y_val, X_test, y_test).
    """
    df = load_raw()
    if verbose:
        inspect(df)
    df = add_time_features(df)
    train, val, test = chronological_split(df)
    if verbose:
        print("\nCHRONOLOGICAL SPLIT")
        for name, part in [("Train", train), ("Validation", val), ("Test", test)]:
            print(f"{name:11s}: {len(part):6d} rows  "
                  f"{part['date'].min()} -> {part['date'].max()}")
    return (train[C.FEATURES], train[C.TARGET],
            val[C.FEATURES], val[C.TARGET],
            test[C.FEATURES], test[C.TARGET])


if __name__ == "__main__":
    get_splits(verbose=True)
