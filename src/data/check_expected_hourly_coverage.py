import duckdb
import yaml


def load_config(path="configs/data_sources.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    cfg = load_config()
    db_path = cfg["storage"]["db_path"]

    con = duckdb.connect(db_path)

    result = con.execute("""
        WITH bounds AS (
            SELECT
                MIN(open_time) AS min_ts,
                MAX(open_time) AS max_ts
            FROM ethusd_market_1h_canonical
        ),
        expected AS (
            SELECT generate_series AS expected_open_time
            FROM bounds,
            generate_series(min_ts, max_ts, INTERVAL '1 hour')
        ),
        actual AS (
            SELECT open_time
            FROM ethusd_market_1h_canonical
        )
        SELECT
            COUNT(*) AS expected_rows,
            (SELECT COUNT(*) FROM actual) AS actual_rows,
            COUNT(*) - (SELECT COUNT(*) FROM actual) AS missing_rows
        FROM expected
    """).fetchdf()

    print(result)

    missing = con.execute("""
        WITH bounds AS (
            SELECT
                MIN(open_time) AS min_ts,
                MAX(open_time) AS max_ts
            FROM ethusd_market_1h_canonical
        ),
        expected AS (
            SELECT generate_series AS expected_open_time
            FROM bounds,
            generate_series(min_ts, max_ts, INTERVAL '1 hour')
        ),
        actual AS (
            SELECT open_time
            FROM ethusd_market_1h_canonical
        )
        SELECT expected.expected_open_time
        FROM expected
        LEFT JOIN actual
          ON expected.expected_open_time = actual.open_time
        WHERE actual.open_time IS NULL
        ORDER BY expected.expected_open_time
        LIMIT 20
    """).fetchdf()

    print("\nMissing candle sample:")
    print(missing)

    con.close()


if __name__ == "__main__":
    main()