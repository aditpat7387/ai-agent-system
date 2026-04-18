import duckdb

con = duckdb.connect("data/market.duckdb")
df = con.execute("""
    SELECT timestamp, price, total_volume
    FROM ethusd_market_raw
    ORDER BY timestamp
""").fetchdf()

print(df.describe())
print(df.head())
print(df.tail())