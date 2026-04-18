import duckdb

con = duckdb.connect("data/market.duckdb")

summary = con.execute("""
    SELECT
        COUNT(*) AS rows,
        MIN(open_time) AS min_open_time,
        MAX(open_time) AS max_open_time,
        MIN(low) AS min_low,
        MAX(high) AS max_high,
        AVG(close) AS avg_close,
        AVG(volume) AS avg_volume
    FROM ethusd_market_1h_canonical
""").fetchdf()

print(summary)

recent = con.execute("""
    SELECT open_time, open, high, low, close, volume
    FROM ethusd_market_1h_canonical
    ORDER BY open_time DESC
    LIMIT 10
""").fetchdf()

print("\nRecent candles:")
print(recent)

quality = con.execute("""
    SELECT
        SUM(bad_high_flag) AS bad_high_rows,
        SUM(bad_low_flag) AS bad_low_rows,
        SUM(negative_volume_flag) AS negative_volume_rows,
        SUM(gap_flag) AS gap_rows
    FROM ethusd_market_1h_quality_report
""").fetchdf()

print("\nQuality summary:")
print(quality)

con.close()