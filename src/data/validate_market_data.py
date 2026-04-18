from pathlib import Path
import pandas as pd


def validate_file(path: str):
    df = pd.read_parquet(path).sort_values("open_time").reset_index(drop=True)

    print("Rows:", len(df))
    print("Columns:", list(df.columns))
    print("Min open_time:", df["open_time"].min())
    print("Max open_time:", df["open_time"].max())
    print("Null counts:")
    print(df.isnull().sum())

    duplicate_open_times = df.duplicated(subset=["open_time"]).sum()
    print("Duplicate open_time:", duplicate_open_times)

    bad_high = ((df["high"] < df["open"]) | (df["high"] < df["close"]) | (df["high"] < df["low"])).sum()
    bad_low = ((df["low"] > df["open"]) | (df["low"] > df["close"]) | (df["low"] > df["high"])).sum()
    negative_volume = (df["volume"] < 0).sum()

    print("Bad high rows:", bad_high)
    print("Bad low rows:", bad_low)
    print("Negative volume rows:", negative_volume)

    df["hour_diff"] = df["open_time"].diff().dt.total_seconds() / 3600
    gap_counts = df["hour_diff"].value_counts(dropna=False).sort_index()
    print("\nHour gap distribution:")
    print(gap_counts.head(10))

    print("\nSample:")
    print(df.head())


if __name__ == "__main__":
    files = sorted(Path("data/raw/binance").glob("*.parquet"))
    if not files:
        raise FileNotFoundError("No parquet files found in data/raw/binance")
    validate_file(str(files[-1]))