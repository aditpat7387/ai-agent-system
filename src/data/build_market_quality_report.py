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
        CREATE OR REPLACE TABLE ethusd_market_1h_quality_report AS
        WITH ordered AS (
            SELECT
                *,
                LAG(open_time) OVER (ORDER BY open_time) AS prev_open_time
            FROM ethusd_market_1h_canonical
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
            provider,

            CASE WHEN high >= GREATEST(open, close, low) THEN 0 ELSE 1 END AS bad_high_flag,
            CASE WHEN low <= LEAST(open, close, high) THEN 0 ELSE 1 END AS bad_low_flag,
            CASE WHEN volume >= 0 THEN 0 ELSE 1 END AS negative_volume_flag,
            CASE
                WHEN prev_open_time IS NULL THEN 0
                WHEN date_diff('hour', prev_open_time, open_time) = 1 THEN 0
                ELSE 1
            END AS gap_flag,
            CASE
                WHEN prev_open_time IS NULL THEN NULL
                ELSE date_diff('hour', prev_open_time, open_time)
            END AS hour_gap_from_previous
        FROM ordered
        ORDER BY open_time
    """)

    summary = con.execute("""
        SELECT
            COUNT(*) AS total_rows,
            SUM(bad_high_flag) AS bad_high_rows,
            SUM(bad_low_flag) AS bad_low_rows,
            SUM(negative_volume_flag) AS negative_volume_rows,
            SUM(gap_flag) AS gap_rows
        FROM ethusd_market_1h_quality_report
    """).fetchdf()

    print(summary)

    gaps = con.execute("""
        SELECT open_time, prev_open_time, hour_gap_from_previous
        FROM (
            SELECT
                open_time,
                LAG(open_time) OVER (ORDER BY open_time) AS prev_open_time,
                hour_gap_from_previous
            FROM ethusd_market_1h_quality_report
        )
        WHERE hour_gap_from_previous IS NOT NULL
          AND hour_gap_from_previous <> 1
        ORDER BY open_time
        LIMIT 20
    """).fetchdf()

    print("\nGap sample:")
    print(gaps)

    con.close()


if __name__ == "__main__":
    main()