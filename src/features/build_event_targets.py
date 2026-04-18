import duckdb
import yaml
import pandas as pd


def load_config(path="configs/data_sources.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def label_event(path_returns, up_thresh=0.006, down_thresh=-0.004):
    for r in path_returns:
        if r >= up_thresh:
            return 1
        if r <= down_thresh:
            return 0
    return 0


def main():
    cfg = load_config()
    db_path = cfg["storage"]["db_path"]
    con = duckdb.connect(db_path)

    df = con.execute("""
        SELECT
            open_time,
            close,
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
            regime_label,
            regime_id
        FROM ethusd_regime_1h
        ORDER BY open_time
    """).fetchdf()

    horizon_bars = 6
    up_thresh = 0.006
    down_thresh = -0.004

    event_labels = []
    event_outcomes = []
    event_up_hit = []
    event_down_hit = []
    event_horizon = []

    closes = df["close"].values
    n = len(df)

    for i in range(n):
        future_end = min(i + horizon_bars, n - 1)
        if future_end <= i:
            event_labels.append(None)
            event_outcomes.append(None)
            event_up_hit.append(None)
            event_down_hit.append(None)
            event_horizon.append(None)
            continue

        start_price = closes[i]
        path_returns = (closes[i + 1:future_end + 1] / start_price) - 1.0

        up_hit_idx = None
        down_hit_idx = None

        for j, r in enumerate(path_returns, start=1):
            if up_hit_idx is None and r >= up_thresh:
                up_hit_idx = j
            if down_hit_idx is None and r <= down_thresh:
                down_hit_idx = j
            if up_hit_idx is not None and down_hit_idx is not None:
                break

        if up_hit_idx is None and down_hit_idx is None:
            label = 0
            outcome = "none"
        elif up_hit_idx is not None and (down_hit_idx is None or up_hit_idx < down_hit_idx):
            label = 1
            outcome = "up_first"
        else:
            label = 0
            outcome = "down_first"

        event_labels.append(label)
        event_outcomes.append(outcome)
        event_up_hit.append(up_hit_idx)
        event_down_hit.append(down_hit_idx)
        event_horizon.append(horizon_bars)

    df["event_horizon_bars"] = event_horizon
    df["event_up_threshold"] = up_thresh
    df["event_down_threshold"] = down_thresh
    df["event_label"] = event_labels
    df["event_outcome"] = event_outcomes
    df["event_up_hit_bar"] = event_up_hit
    df["event_down_hit_bar"] = event_down_hit

    con.register("event_df", df)

    con.execute("""
        CREATE OR REPLACE TABLE ethusd_event_targets_1h AS
        SELECT * FROM event_df
    """)

    summary = con.execute("""
        SELECT
            regime_label,
            event_label,
            COUNT(*) AS cnt
        FROM ethusd_event_targets_1h
        WHERE event_label IS NOT NULL
        GROUP BY regime_label, event_label
        ORDER BY regime_label, event_label
    """).fetchdf()

    sample = con.execute("""
        SELECT
            open_time, close, regime_label,
            event_horizon_bars, event_up_threshold, event_down_threshold,
            event_outcome, event_label, event_up_hit_bar, event_down_hit_bar
        FROM ethusd_event_targets_1h
        ORDER BY open_time DESC
        LIMIT 20
    """).fetchdf()

    print("Event label counts by regime:")
    print(summary.to_string(index=False))
    print("\nRecent sample:")
    print(sample.to_string(index=False))

    con.close()


if __name__ == "__main__":
    main()