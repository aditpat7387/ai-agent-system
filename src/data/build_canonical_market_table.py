from pathlib import Path
import duckdb
import yaml


def load_config(path="configs/data_sources.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    cfg = load_config()
    db_path = cfg["storage"]["db_path"]

    con = duckdb.connect(db_path)

    con.execute("""
        CREATE OR REPLACE TABLE ethusd_market_1h_canonical AS
        WITH base AS (
            SELECT
                open_time,
                close_time,
                canonical_symbol AS symbol,
                interval,
                CAST(open AS DOUBLE) AS open,
                CAST(high AS DOUBLE) AS high,
                CAST(low AS DOUBLE) AS low,
                CAST(close AS DOUBLE) AS close,
                CAST(volume AS DOUBLE) AS volume,
                CAST(quote_asset_volume AS DOUBLE) AS quote_asset_volume,
                CAST(number_of_trades AS BIGINT) AS number_of_trades,
                CAST(taker_buy_base_volume AS DOUBLE) AS taker_buy_base_volume,
                CAST(taker_buy_quote_volume AS DOUBLE) AS taker_buy_quote_volume,
                provider,
                ROW_NUMBER() OVER (
                    PARTITION BY canonical_symbol, interval, open_time
                    ORDER BY close_time DESC
                ) AS row_num
            FROM ethusd_klines_raw
            WHERE canonical_symbol = 'ETHUSD'
              AND interval = '1h'
        )
        SELECT
            open_time,
            close_time,
            symbol,
            interval,
            open,
            high,
            low,
            close,
            volume,
            quote_asset_volume,
            number_of_trades,
            taker_buy_base_volume,
            taker_buy_quote_volume,
            provider
        FROM base
        WHERE row_num = 1
        ORDER BY open_time
    """)

    count = con.execute("SELECT COUNT(*) FROM ethusd_market_1h_canonical").fetchone()[0]
    print(f"Canonical table created with {count} rows")

    sample = con.execute("""
        SELECT *
        FROM ethusd_market_1h_canonical
        ORDER BY open_time DESC
        LIMIT 5
    """).fetchdf()
    print(sample)

    con.close()


if __name__ == "__main__":
    main()