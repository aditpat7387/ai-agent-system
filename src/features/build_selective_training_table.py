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
        CREATE OR REPLACE TABLE ethusd_training_1h_v4 AS
        WITH joined AS (
            SELECT
                e.open_time,
                e.close_time,
                e.symbol,
                e.interval,
                e.open,
                e.high,
                e.low,
                e.close,
                e.volume,
                e.return_1h,
                e.log_return_1h,
                e.vol_24h,
                e.vol_72h,
                e.atr_14_pct,
                e.rsi_14,
                e.rel_volume_24,
                e.close_vs_sma_7,
                e.close_vs_sma_24,
                e.sma_7_vs_24,
                e.sma_24_vs_72,
                e.dist_to_bb_upper_pct,
                e.dist_to_bb_lower_pct,
                e.hour_of_day,
                e.day_of_week,
                e.regime_label,
                e.regime_id,
                t.event_horizon_bars,
                t.event_up_threshold,
                t.event_down_threshold,
                t.event_label,
                t.event_outcome,
                t.event_up_hit_bar,
                t.event_down_hit_bar
            FROM ethusd_regime_1h e
            INNER JOIN ethusd_event_targets_1h t
                ON e.open_time = t.open_time
            WHERE t.event_label IS NOT NULL
              AND e.regime_label IS NOT NULL
              AND e.regime_label <> 'unknown'
        ),
        scored AS (
            SELECT
                *,
                CASE
                    WHEN regime_label IN ('trend_up', 'trend_down') THEN 1
                    WHEN regime_label = 'high_volatility' AND rel_volume_24 >= 1.0 THEN 1
                    ELSE 0
                END AS tradable_flag,
                CASE
                    WHEN regime_label IN ('trend_up', 'trend_down') AND event_label = 1 THEN 'high'
                    WHEN regime_label = 'high_volatility' AND event_label = 1 THEN 'medium'
                    WHEN regime_label = 'range' AND event_label = 1 THEN 'low'
                    ELSE 'low'
                END AS confidence_bucket
            FROM joined
        )
        SELECT
            *,
            CASE
                WHEN tradable_flag = 1 AND confidence_bucket IN ('high', 'medium') THEN 1
                ELSE 0
            END AS model_allowed_flag
        FROM scored
        ORDER BY open_time
    """)

    summary = con.execute("""
        SELECT
            regime_label,
            tradable_flag,
            confidence_bucket,
            model_allowed_flag,
            COUNT(*) AS cnt
        FROM ethusd_training_1h_v4
        GROUP BY regime_label, tradable_flag, confidence_bucket, model_allowed_flag
        ORDER BY regime_label, tradable_flag, confidence_bucket, model_allowed_flag
    """).fetchdf()

    sample = con.execute("""
        SELECT
            open_time, close, regime_label, event_label, event_outcome,
            tradable_flag, confidence_bucket, model_allowed_flag,
            atr_14_pct, rsi_14, rel_volume_24, sma_7_vs_24, event_horizon_bars
        FROM ethusd_training_1h_v4
        ORDER BY open_time DESC
        LIMIT 20
    """).fetchdf()

    print("Training table summary:")
    print(summary.to_string(index=False))
    print("\nRecent sample:")
    print(sample.to_string(index=False))

    con.close()


if __name__ == "__main__":
    main()