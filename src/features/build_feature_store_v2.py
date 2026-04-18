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
        CREATE OR REPLACE TABLE ethusd_features_1h_v2 AS
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
                LAG(number_of_trades, 1) OVER (ORDER BY open_time) AS trades_lag_1,
                LAG(high, 1) OVER (ORDER BY open_time) AS high_lag_1,
                LAG(low, 1) OVER (ORDER BY open_time) AS low_lag_1,
                EXTRACT(HOUR FROM open_time) AS hour_of_day,
                EXTRACT(DOW FROM open_time) AS day_of_week
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
                (high - low) / NULLIF(close, 0) AS high_low_range_pct,
                GREATEST(high - low, ABS(high - close_lag_1), ABS(low - close_lag_1)) AS true_range
            FROM base
        ),
        rolling AS (
            SELECT
                *,
                AVG(close) OVER (ORDER BY open_time ROWS BETWEEN 6 PRECEDING AND CURRENT ROW) AS sma_7,
                AVG(close) OVER (ORDER BY open_time ROWS BETWEEN 23 PRECEDING AND CURRENT ROW) AS sma_24,
                AVG(close) OVER (ORDER BY open_time ROWS BETWEEN 71 PRECEDING AND CURRENT ROW) AS sma_72,

                AVG(volume) OVER (ORDER BY open_time ROWS BETWEEN 23 PRECEDING AND CURRENT ROW) AS avg_volume_24,
                AVG(number_of_trades) OVER (ORDER BY open_time ROWS BETWEEN 23 PRECEDING AND CURRENT ROW) AS avg_trades_24,

                STDDEV_SAMP(log_return_1h) OVER (ORDER BY open_time ROWS BETWEEN 23 PRECEDING AND CURRENT ROW) AS vol_24h,
                STDDEV_SAMP(log_return_1h) OVER (ORDER BY open_time ROWS BETWEEN 71 PRECEDING AND CURRENT ROW) AS vol_72h,

                AVG(true_range) OVER (ORDER BY open_time ROWS BETWEEN 13 PRECEDING AND CURRENT ROW) AS atr_14,
                AVG(high_low_range_pct) OVER (ORDER BY open_time ROWS BETWEEN 23 PRECEDING AND CURRENT ROW) AS avg_range_pct_24
            FROM returns
        ),
        indicators AS (
            SELECT
                *,
                100 - (100 / (1 + (
                    AVG(CASE WHEN log_return_1h > 0 THEN log_return_1h ELSE 0 END)
                    OVER (ORDER BY open_time ROWS BETWEEN 13 PRECEDING AND CURRENT ROW)
                    /
                    NULLIF(
                        ABS(AVG(CASE WHEN log_return_1h < 0 THEN log_return_1h ELSE 0 END)
                        OVER (ORDER BY open_time ROWS BETWEEN 13 PRECEDING AND CURRENT ROW)),
                        0
                    )
                ))) AS rsi_14
            FROM rolling
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
                (taker_buy_base_volume / NULLIF(volume, 0)) AS taker_buy_ratio,
                (atr_14 / NULLIF(close, 0)) AS atr_14_pct,
                (true_range / NULLIF(close, 0)) AS true_range_pct,
                AVG(close) OVER (ORDER BY open_time ROWS BETWEEN 23 PRECEDING AND CURRENT ROW) AS bb_mid_24,
                STDDEV_SAMP(close) OVER (ORDER BY open_time ROWS BETWEEN 23 PRECEDING AND CURRENT ROW) AS bb_std_24
            FROM indicators
        )
        SELECT
            *,
            (bb_mid_24 + 2 * bb_std_24) AS bb_upper_24,
            (bb_mid_24 - 2 * bb_std_24) AS bb_lower_24,
            (close - (bb_mid_24 + 2 * bb_std_24)) / NULLIF(close, 0) AS dist_to_bb_upper_pct,
            (close - (bb_mid_24 - 2 * bb_std_24)) / NULLIF(close, 0) AS dist_to_bb_lower_pct,
            CASE
                WHEN return_24h > 0.02 THEN 1
                WHEN return_24h < -0.02 THEN -1
                ELSE 0
            END AS regime_proxy_24h
        FROM derived
        ORDER BY open_time
    """)

    count = con.execute("SELECT COUNT(*) FROM ethusd_features_1h_v2").fetchone()[0]
    print(f"Feature v2 table created with {count} rows")

    sample = con.execute("""
        SELECT
            open_time, close, return_1h, return_4h, return_24h,
            vol_24h, vol_72h, atr_14_pct, rsi_14,
            close_vs_sma_7, close_vs_sma_24, sma_7_vs_24, sma_24_vs_72,
            rel_volume_24, rel_trades_24, taker_buy_ratio,
            hour_of_day, day_of_week,
            dist_to_bb_upper_pct, dist_to_bb_lower_pct,
            regime_proxy_24h
        FROM ethusd_features_1h_v2
        ORDER BY open_time DESC
        LIMIT 10
    """).fetchdf()
    print(sample)

    con.close()


if __name__ == "__main__":
    main()