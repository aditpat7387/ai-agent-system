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
        CREATE OR REPLACE TABLE ethusd_features_1h AS
        WITH base AS (
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

                LAG(close, 1) OVER (ORDER BY open_time) AS close_lag_1,
                LAG(close, 4) OVER (ORDER BY open_time) AS close_lag_4,
                LAG(close, 24) OVER (ORDER BY open_time) AS close_lag_24,
                LAG(volume, 1) OVER (ORDER BY open_time) AS volume_lag_1,
                LAG(number_of_trades, 1) OVER (ORDER BY open_time) AS trades_lag_1
            FROM ethusd_market_1h_canonical
            ORDER BY open_time
        ),
        returns AS (
            SELECT
                *,
                (close / close_lag_1 - 1.0) AS return_1h,
                LN(close / close_lag_1) AS log_return_1h,
                (close / close_lag_4 - 1.0) AS return_4h,
                (close / close_lag_24 - 1.0) AS return_24h,
                (volume / NULLIF(volume_lag_1, 0) - 1.0) AS volume_change_1h,
                (number_of_trades::DOUBLE / NULLIF(trades_lag_1, 0) - 1.0) AS trades_change_1h,
                (high - low) AS high_low_range,
                (high - low) / NULLIF(close, 0) AS high_low_range_pct
            FROM base
        ),
        rolling AS (
            SELECT
                *,
                AVG(close) OVER (
                    ORDER BY open_time
                    ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
                ) AS sma_7,
                AVG(close) OVER (
                    ORDER BY open_time
                    ROWS BETWEEN 23 PRECEDING AND CURRENT ROW
                ) AS sma_24,
                AVG(close) OVER (
                    ORDER BY open_time
                    ROWS BETWEEN 71 PRECEDING AND CURRENT ROW
                ) AS sma_72,

                AVG(volume) OVER (
                    ORDER BY open_time
                    ROWS BETWEEN 23 PRECEDING AND CURRENT ROW
                ) AS avg_volume_24,

                AVG(number_of_trades) OVER (
                    ORDER BY open_time
                    ROWS BETWEEN 23 PRECEDING AND CURRENT ROW
                ) AS avg_trades_24,

                STDDEV_SAMP(log_return_1h) OVER (
                    ORDER BY open_time
                    ROWS BETWEEN 23 PRECEDING AND CURRENT ROW
                ) AS vol_24h,

                STDDEV_SAMP(log_return_1h) OVER (
                    ORDER BY open_time
                    ROWS BETWEEN 71 PRECEDING AND CURRENT ROW
                ) AS vol_72h
            FROM returns
        ),
        derived AS (
            SELECT
                *,
                (close / NULLIF(sma_7, 0) - 1.0) AS close_vs_sma_7,
                (close / NULLIF(sma_24, 0) - 1.0) AS close_vs_sma_24,
                (sma_7 / NULLIF(sma_24, 0) - 1.0) AS sma_7_vs_24,
                (sma_24 / NULLIF(sma_72, 0) - 1.0) AS sma_24_vs_72,
                (volume / NULLIF(avg_volume_24, 0)) AS rel_volume_24,
                (number_of_trades / NULLIF(avg_trades_24, 0)) AS rel_trades_24,
                (taker_buy_base_volume / NULLIF(volume, 0)) AS taker_buy_ratio
            FROM rolling
        )
        SELECT
            *,
            CASE
                WHEN return_24h > 0.02 THEN 1
                WHEN return_24h < -0.02 THEN -1
                ELSE 0
            END AS regime_proxy_24h
        FROM derived
        ORDER BY open_time
    """)

    count = con.execute("SELECT COUNT(*) FROM ethusd_features_1h").fetchone()[0]
    print(f"Feature table created with {count} rows")

    sample = con.execute("""
        SELECT
            open_time, close, return_1h, return_4h, return_24h,
            vol_24h, sma_7_vs_24, rel_volume_24, taker_buy_ratio, regime_proxy_24h
        FROM ethusd_features_1h
        ORDER BY open_time DESC
        LIMIT 10
    """).fetchdf()
    print(sample)

    con.close()


if __name__ == "__main__":
    main()