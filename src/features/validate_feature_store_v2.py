import duckdb


def main():
    con = duckdb.connect("data/market.duckdb")

    summary = con.execute("""
        SELECT
            COUNT(*) AS total_rows,
            MIN(open_time) AS min_open_time,
            MAX(open_time) AS max_open_time
        FROM ethusd_features_1h_v2
    """).fetchdf()
    print("Summary:")
    print(summary)

    nulls = con.execute("""
        SELECT
            SUM(CASE WHEN rsi_14 IS NULL THEN 1 ELSE 0 END) AS null_rsi_14,
            SUM(CASE WHEN atr_14_pct IS NULL THEN 1 ELSE 0 END) AS null_atr_14_pct,
            SUM(CASE WHEN bb_mid_24 IS NULL THEN 1 ELSE 0 END) AS null_bb_mid_24,
            SUM(CASE WHEN dist_to_bb_upper_pct IS NULL THEN 1 ELSE 0 END) AS null_bb_dist_upper,
            SUM(CASE WHEN hour_of_day IS NULL THEN 1 ELSE 0 END) AS null_hour_of_day,
            SUM(CASE WHEN day_of_week IS NULL THEN 1 ELSE 0 END) AS null_day_of_week
        FROM ethusd_features_1h_v2
    """).fetchdf()
    print("\nNull summary:")
    print(nulls)

    sample = con.execute("""
        SELECT
            open_time, close, return_1h, return_4h, return_24h,
            vol_24h, atr_14_pct, rsi_14,
            close_vs_sma_7, close_vs_sma_24,
            sma_7_vs_24, sma_24_vs_72,
            rel_volume_24, rel_trades_24, taker_buy_ratio,
            hour_of_day, day_of_week,
            dist_to_bb_upper_pct, dist_to_bb_lower_pct,
            target_direction_4h_3class
        FROM ethusd_training_1h_v2
        ORDER BY open_time DESC
        LIMIT 10
    """).fetchdf()
    print("\nRecent feature sample:")
    print(sample)

    con.close()


if __name__ == "__main__":
    main()