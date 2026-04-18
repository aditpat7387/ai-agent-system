import duckdb


def main():
    con = duckdb.connect("data/market.duckdb")

    summary = con.execute("""
        SELECT
            COUNT(*) AS total_rows,
            MIN(open_time) AS min_open_time,
            MAX(open_time) AS max_open_time
        FROM ethusd_features_1h
    """).fetchdf()
    print("Summary:")
    print(summary)

    nulls = con.execute("""
        SELECT
            SUM(CASE WHEN return_1h IS NULL THEN 1 ELSE 0 END) AS null_return_1h,
            SUM(CASE WHEN return_4h IS NULL THEN 1 ELSE 0 END) AS null_return_4h,
            SUM(CASE WHEN return_24h IS NULL THEN 1 ELSE 0 END) AS null_return_24h,
            SUM(CASE WHEN vol_24h IS NULL THEN 1 ELSE 0 END) AS null_vol_24h,
            SUM(CASE WHEN sma_72 IS NULL THEN 1 ELSE 0 END) AS null_sma_72,
            SUM(CASE WHEN rel_volume_24 IS NULL THEN 1 ELSE 0 END) AS null_rel_volume_24
        FROM ethusd_features_1h
    """).fetchdf()
    print("\nNull summary:")
    print(nulls)

    sample = con.execute("""
        SELECT
            open_time, close,
            return_1h, log_return_1h, return_4h, return_24h,
            vol_24h, vol_72h,
            close_vs_sma_7, close_vs_sma_24,
            sma_7_vs_24, sma_24_vs_72,
            rel_volume_24, rel_trades_24, taker_buy_ratio,
            regime_proxy_24h
        FROM ethusd_features_1h
        ORDER BY open_time DESC
        LIMIT 10
    """).fetchdf()
    print("\nRecent feature sample:")
    print(sample)

    con.close()


if __name__ == "__main__":
    main()