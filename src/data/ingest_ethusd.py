from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import yaml

from src.data.binance_client import BinanceClient


def load_config(path="configs/data_sources.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def interval_to_milliseconds(interval: str) -> int:
    mapping = {
        "1m": 60_000,
        "5m": 300_000,
        "15m": 900_000,
        "30m": 1_800_000,
        "1h": 3_600_000,
        "4h": 14_400_000,
        "1d": 86_400_000,
    }
    return mapping[interval]


def klines_to_df(rows, canonical_symbol: str, interval: str) -> pd.DataFrame:
    columns = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "number_of_trades",
        "taker_buy_base_volume", "taker_buy_quote_volume", "ignore"
    ]
    df = pd.DataFrame(rows, columns=columns)

    numeric_cols = [
        "open", "high", "low", "close", "volume",
        "quote_asset_volume", "taker_buy_base_volume", "taker_buy_quote_volume"
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["number_of_trades"] = pd.to_numeric(df["number_of_trades"], errors="coerce").astype("Int64")
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)

    df["symbol"] = "ETHUSDT"
    df["canonical_symbol"] = canonical_symbol
    df["interval"] = interval
    df["provider"] = "binance"

    return df[
        [
            "open_time", "close_time", "symbol", "canonical_symbol", "interval",
            "open", "high", "low", "close", "volume", "quote_asset_volume",
            "number_of_trades", "taker_buy_base_volume", "taker_buy_quote_volume", "provider"
        ]
    ]


def main():
    cfg = load_config()
    symbol = cfg["target"]["symbol"]
    canonical_symbol = cfg["target"]["canonical_symbol"]
    interval = cfg["ingestion"]["interval"]
    lookback_days = cfg["ingestion"]["lookback_days"]
    limit = cfg["ingestion"]["limit_per_call"]
    raw_dir = Path(cfg["storage"]["raw_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)

    client = BinanceClient(
        base_url=cfg["provider"]["base_url"],
        pause_seconds=cfg["ingestion"]["request_pause_seconds"]
    )

    end_dt = datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(days=lookback_days)

    start_ms = int(start_dt.timestamp() * 1000)
    end_ms = int(end_dt.timestamp() * 1000)
    step_ms = interval_to_milliseconds(interval) * limit

    all_rows = []
    current_start = start_ms

    while current_start < end_ms:
        batch = client.get_klines(
            symbol=symbol,
            interval=interval,
            start_time=current_start,
            end_time=end_ms,
            limit=limit
        )
        if not batch:
            break

        all_rows.extend(batch)
        last_open_time = batch[-1][0]
        next_start = last_open_time + interval_to_milliseconds(interval)

        if next_start <= current_start:
            break

        current_start = next_start

    df = klines_to_df(all_rows, canonical_symbol=canonical_symbol, interval=interval)
    df = df.drop_duplicates(subset=["open_time"]).sort_values("open_time").reset_index(drop=True)

    output_path = raw_dir / f"ethusdt_1h_{start_dt.date()}_{end_dt.date()}.parquet"
    df.to_parquet(output_path, index=False)

    print(f"Saved {len(df)} rows to {output_path}")
    print(df.head())
    print(df.tail())


if __name__ == "__main__":
    main()