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
        CREATE OR REPLACE TABLE ethusd_regime_1h AS
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
                return_1h,
                log_return_1h,
                vol_24h,
                vol_72h,
                atr_14_pct,
                rsi_14,
                rel_volume_24,
                close_vs_sma_7,
                close_vs_sma_24,
                sma_7_vs_24,
                sma_24_vs_72,
                dist_to_bb_upper_pct,
                dist_to_bb_lower_pct,
                hour_of_day,
                day_of_week,
                LAG(close, 1) OVER (ORDER BY open_time) AS close_lag_1
            FROM ethusd_features_1h_v2
            ORDER BY open_time
        ),
        stage2 AS (
            SELECT
                *,
                ABS(return_1h) AS abs_return_1h,
                CASE
                    WHEN close_lag_1 IS NULL THEN 0
                    WHEN close > close_lag_1 THEN 1
                    WHEN close < close_lag_1 THEN -1
                    ELSE 0
                END AS bar_direction
            FROM base
        ),
        stage3 AS (
            SELECT
                *,
                AVG(abs_return_1h) OVER (
                    ORDER BY open_time
                    ROWS BETWEEN 23 PRECEDING AND CURRENT ROW
                ) AS abs_return_24,
                AVG(abs_return_1h) OVER (
                    ORDER BY open_time
                    ROWS BETWEEN 71 PRECEDING AND CURRENT ROW
                ) AS abs_return_72,
                AVG(bar_direction) OVER (
                    ORDER BY open_time
                    ROWS BETWEEN 23 PRECEDING AND CURRENT ROW
                ) AS trend_bias_24,
                AVG(close - close_lag_1) OVER (
                    ORDER BY open_time
                    ROWS BETWEEN 23 PRECEDING AND CURRENT ROW
                ) AS avg_delta_24,
                STDDEV_SAMP(close) OVER (
                    ORDER BY open_time
                    ROWS BETWEEN 23 PRECEDING AND CURRENT ROW
                ) AS close_std_24,
                AVG(close) OVER (
                    ORDER BY open_time
                    ROWS BETWEEN 23 PRECEDING AND CURRENT ROW
                ) AS close_mean_24,
                STDDEV_SAMP(close) OVER (
                    ORDER BY open_time
                    ROWS BETWEEN 71 PRECEDING AND CURRENT ROW
                ) AS close_std_72,
                AVG(close) OVER (
                    ORDER BY open_time
                    ROWS BETWEEN 71 PRECEDING AND CURRENT ROW
                ) AS close_mean_72
            FROM stage2
        ),
        regime_features AS (
            SELECT
                *,
                (close_std_24 / NULLIF(close_mean_24, 0)) AS bb_width_24,
                (close_std_72 / NULLIF(close_mean_72, 0)) AS bb_width_72,
                (close_vs_sma_7 - close_vs_sma_24) AS short_trend_spread,
                (sma_7_vs_24 - sma_24_vs_72) AS trend_acceleration,
                (atr_14_pct / NULLIF(vol_72h, 0)) AS atr_to_vol_ratio
            FROM stage3
        )
        SELECT
            *,
            CASE
                WHEN atr_14_pct IS NULL OR bb_width_24 IS NULL OR sma_7_vs_24 IS NULL THEN 'unknown'
                WHEN atr_14_pct < 0.006 AND bb_width_24 < 0.010 THEN 'compression'
                WHEN atr_14_pct >= 0.006 AND atr_14_pct < 0.012 AND ABS(short_trend_spread) < 0.01 THEN 'range'
                WHEN sma_7_vs_24 > 0.005 AND trend_acceleration >= 0 AND rsi_14 >= 52 THEN 'trend_up'
                WHEN sma_7_vs_24 < -0.005 AND trend_acceleration <= 0 AND rsi_14 <= 48 THEN 'trend_down'
                WHEN atr_14_pct >= 0.012 OR bb_width_24 >= 0.020 THEN 'high_volatility'
                ELSE 'range'
            END AS regime_label,
            CASE
                WHEN atr_14_pct < 0.006 AND bb_width_24 < 0.010 THEN 1
                WHEN atr_14_pct >= 0.006 AND atr_14_pct < 0.012 AND ABS(short_trend_spread) < 0.01 THEN 2
                WHEN sma_7_vs_24 > 0.005 AND trend_acceleration >= 0 AND rsi_14 >= 52 THEN 3
                WHEN sma_7_vs_24 < -0.005 AND trend_acceleration <= 0 AND rsi_14 <= 48 THEN 4
                WHEN atr_14_pct >= 0.012 OR bb_width_24 >= 0.020 THEN 5
                ELSE 2
            END AS regime_id
        FROM regime_features
        ORDER BY open_time
    """)

    sample = con.execute("""
        SELECT
            open_time, close, atr_14_pct, bb_width_24, vol_24h, vol_72h,
            sma_7_vs_24, trend_acceleration, rsi_14, rel_volume_24,
            regime_label, regime_id
        FROM ethusd_regime_1h
        ORDER BY open_time DESC
        LIMIT 20
    """).fetchdf()

    counts = con.execute("""
        SELECT regime_label, COUNT(*) AS cnt
        FROM ethusd_regime_1h
        GROUP BY regime_label
        ORDER BY cnt DESC
    """).fetchdf()

    print("Regime counts:")
    print(counts.to_string(index=False))
    print("\nRecent sample:")
    print(sample.to_string(index=False))

    con.close()


if __name__ == "__main__":
    main()