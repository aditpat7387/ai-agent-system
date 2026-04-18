from pathlib import Path
import duckdb
import yaml


def load_config(path="configs/data_sources.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    cfg = load_config()
    db_path = cfg["storage"]["db_path"]
    raw_dir = Path(cfg["storage"]["raw_dir"])

    con = duckdb.connect(db_path)

    con.execute("""
        CREATE TABLE IF NOT EXISTS ethusd_klines_raw (
            open_time TIMESTAMP,
            close_time TIMESTAMP,
            symbol VARCHAR,
            canonical_symbol VARCHAR,
            interval VARCHAR,
            open DOUBLE,
            high DOUBLE,
            low DOUBLE,
            close DOUBLE,
            volume DOUBLE,
            quote_asset_volume DOUBLE,
            number_of_trades BIGINT,
            taker_buy_base_volume DOUBLE,
            taker_buy_quote_volume DOUBLE,
            provider VARCHAR
        )
    """)

    parquet_files = sorted(raw_dir.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError("No raw parquet files found")

    latest_file = parquet_files[-1]

    con.execute("DELETE FROM ethusd_klines_raw")
    con.execute(f"""
        INSERT INTO ethusd_klines_raw
        SELECT * FROM read_parquet('{latest_file.as_posix()}')
    """)

    count = con.execute("SELECT COUNT(*) FROM ethusd_klines_raw").fetchone()[0]
    print(f"Loaded {count} rows into {db_path}")

    sample = con.execute("""
        SELECT *
        FROM ethusd_klines_raw
        ORDER BY open_time DESC
        LIMIT 5
    """).fetchdf()
    print(sample)

    con.close()


if __name__ == "__main__":
    main()