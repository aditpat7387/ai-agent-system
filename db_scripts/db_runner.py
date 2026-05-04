import duckdb
from pathlib import Path

db_path = r"data/market.duckdb"
sql_path = Path(r"db_scripts/duckdb_debug_queries.sql")

sql_text = sql_path.read_text(encoding="utf-8")
queries = [q.strip() for q in sql_text.split(";") if q.strip()]

con = duckdb.connect(db_path)

for i, query in enumerate(queries, 1):
    print(f"\n{'='*80}")
    print(f"QUERY {i}")
    print(f"{'='*80}")
    print(query)
    try:
        df = con.execute(query).df()
        print(df.to_string(index=False))
    except Exception as e:
        print(f"ERROR: {e}")

con.close()