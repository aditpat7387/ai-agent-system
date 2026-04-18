import duckdb


def main():
    con = duckdb.connect("data/market.duckdb")

    con.execute("""
        CREATE OR REPLACE TABLE ethusd_training_1h AS
        SELECT
            *,
            LEAD(close, 1) OVER (ORDER BY open_time) AS future_close_1h,
            LEAD(close, 4) OVER (ORDER BY open_time) AS future_close_4h,
            LEAD(close, 24) OVER (ORDER BY open_time) AS future_close_24h,

            (LEAD(close, 1) OVER (ORDER BY open_time) / close - 1.0) AS target_return_1h,
            (LEAD(close, 4) OVER (ORDER BY open_time) / close - 1.0) AS target_return_4h,
            (LEAD(close, 24) OVER (ORDER BY open_time) / close - 1.0) AS target_return_24h,

            CASE
                WHEN (LEAD(close, 1) OVER (ORDER BY open_time) / close - 1.0) > 0 THEN 1
                ELSE 0
            END AS target_direction_1h,

            CASE
                WHEN (LEAD(close, 4) OVER (ORDER BY open_time) / close - 1.0) > 0 THEN 1
                ELSE 0
            END AS target_direction_4h,

            CASE
                WHEN (LEAD(close, 24) OVER (ORDER BY open_time) / close - 1.0) > 0 THEN 1
                ELSE 0
            END AS target_direction_24h
        FROM ethusd_features_1h
        ORDER BY open_time
    """)

    sample = con.execute("""
        SELECT
            open_time, close,
            return_1h, return_4h, vol_24h, sma_7_vs_24,
            target_return_1h, target_return_4h, target_return_24h,
            target_direction_1h, target_direction_4h, target_direction_24h
        FROM ethusd_training_1h
        ORDER BY open_time DESC
        LIMIT 10
    """).fetchdf()

    print(sample)
    con.close()


if __name__ == "__main__":
    main()