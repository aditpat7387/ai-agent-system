from pathlib import Path
from datetime import datetime, timezone
import uuid
import duckdb
import yaml
import pandas as pd
import numpy as np


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def utc_now():
    return datetime.now(timezone.utc)


def safe_float(x, default=np.nan):
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def build_volatility_proxy(df, price_col="close", window=20):
    ret_1 = df[price_col].astype(float).pct_change()
    vol = ret_1.rolling(window).std()
    fallback = ret_1.std()
    if pd.isna(fallback) or fallback <= 0:
        fallback = 0.01
    return vol.fillna(fallback)


def normalize_prediction_columns(df):
    prob_source = None
    label_source = None
    actual_source = None

    if "specialist_pred_proba" in df.columns:
        prob_source = "specialist_pred_proba"
    elif "pred_proba" in df.columns:
        df["specialist_pred_proba"] = df["pred_proba"]
        prob_source = "pred_proba"
    elif "cal_pred_proba" in df.columns:
        df["specialist_pred_proba"] = df["cal_pred_proba"]
        prob_source = "cal_pred_proba"
    else:
        raise ValueError(
            "Missing prediction probability column. Expected one of: "
            "specialist_pred_proba, pred_proba, cal_pred_proba"
        )

    if "specialist_pred_label" in df.columns:
        label_source = "specialist_pred_label"
    elif "pred_label" in df.columns:
        df["specialist_pred_label"] = df["pred_label"]
        label_source = "pred_label"
    else:
        raise ValueError(
            "Missing prediction label column. Expected one of: "
            "specialist_pred_label, pred_label"
        )

    if "actual_label" in df.columns:
        actual_source = "actual_label"
    elif "event_label" in df.columns:
        df["actual_label"] = df["event_label"]
        actual_source = "event_label"
    else:
        df["actual_label"] = np.nan
        actual_source = "not_available"

    return df, prob_source, label_source, actual_source


def ensure_tables(con, runtime_table, signals_table, trades_table, metrics_table):
    con.execute(f"""
        CREATE TABLE IF NOT EXISTS {runtime_table} (
            strategy_name VARCHAR PRIMARY KEY,
            last_processed_open_time TIMESTAMP,
            current_equity DOUBLE,
            open_trade_id VARCHAR,
            open_entry_time TIMESTAMP,
            open_entry_price DOUBLE,
            open_fill_entry DOUBLE,
            open_pred_proba DOUBLE,
            open_pred_label INTEGER,
            open_actual_label INTEGER,
            open_vol20 DOUBLE,
            open_stop_pct DOUBLE,
            open_take_profit_pct DOUBLE,
            open_stop_price DOUBLE,
            open_take_profit_price DOUBLE,
            open_qty DOUBLE,
            open_bars_held INTEGER,
            updated_at TIMESTAMP
        )
    """)

    con.execute(f"""
        CREATE TABLE IF NOT EXISTS {signals_table} (
            run_ts TIMESTAMP,
            strategy_name VARCHAR,
            open_time TIMESTAMP,
            regime_label VARCHAR,
            close DOUBLE,
            specialist_pred_proba DOUBLE,
            specialist_pred_label INTEGER,
            actual_label INTEGER,
            decision VARCHAR,
            reject_reason VARCHAR,
            threshold DOUBLE,
            required_regime VARCHAR,
            required_label INTEGER,
            current_equity_before DOUBLE,
            current_equity_after DOUBLE
        )
    """)

    con.execute(f"""
        CREATE TABLE IF NOT EXISTS {trades_table} (
            run_ts TIMESTAMP,
            trade_id VARCHAR,
            strategy_name VARCHAR,
            entry_time TIMESTAMP,
            exit_time TIMESTAMP,
            side VARCHAR,
            regime_label VARCHAR,
            threshold DOUBLE,
            entry_price DOUBLE,
            exit_price DOUBLE,
            fill_entry DOUBLE,
            fill_exit DOUBLE,
            specialist_pred_proba DOUBLE,
            specialist_pred_label INTEGER,
            actual_label INTEGER,
            vol20 DOUBLE,
            stop_pct DOUBLE,
            take_profit_pct DOUBLE,
            stop_price DOUBLE,
            take_profit_price DOUBLE,
            qty DOUBLE,
            bars_held INTEGER,
            exit_reason VARCHAR,
            gross_return DOUBLE,
            net_return DOUBLE,
            pnl_dollars DOUBLE,
            account_equity_after DOUBLE
        )
    """)

    con.execute(f"""
        CREATE TABLE IF NOT EXISTS {metrics_table} (
            run_ts TIMESTAMP,
            strategy_name VARCHAR,
            last_processed_open_time TIMESTAMP,
            current_equity DOUBLE,
            total_trades INTEGER,
            open_positions INTEGER,
            win_rate DOUBLE,
            avg_net_return DOUBLE,
            total_net_return DOUBLE,
            max_drawdown DOUBLE,
            gross_profit DOUBLE,
            gross_loss DOUBLE,
            profit_factor DOUBLE,
            expectancy_dollars DOUBLE
        )
    """)


def get_runtime_state(con, runtime_table, strategy_name, initial_equity):
    df = con.execute(f"""
        SELECT *
        FROM {runtime_table}
        WHERE strategy_name = ?
    """, [strategy_name]).fetchdf()

    if df.empty:
        return {
            "strategy_name": strategy_name,
            "last_processed_open_time": None,
            "current_equity": float(initial_equity),
            "open_trade_id": None,
            "open_entry_time": None,
            "open_entry_price": None,
            "open_fill_entry": None,
            "open_pred_proba": None,
            "open_pred_label": None,
            "open_actual_label": None,
            "open_vol20": None,
            "open_stop_pct": None,
            "open_take_profit_pct": None,
            "open_stop_price": None,
            "open_take_profit_price": None,
            "open_qty": None,
            "open_bars_held": 0,
            "updated_at": utc_now(),
        }

    row = df.iloc[0].to_dict()
    return row


def upsert_runtime_state(con, runtime_table, state):
    payload = pd.DataFrame([state])
    con.register("runtime_upsert_df", payload)
    con.execute(f"""
        DELETE FROM {runtime_table}
        WHERE strategy_name = (
            SELECT strategy_name FROM runtime_upsert_df LIMIT 1
        )
    """)
    con.execute(f"""
        INSERT INTO {runtime_table}
        SELECT * FROM runtime_upsert_df
    """)


def fetch_new_rows(con, predictions_table, last_processed_open_time=None, start_time=None):
    if last_processed_open_time is not None:
        query = f"""
            SELECT *
            FROM {predictions_table}
            WHERE open_time > ?
            ORDER BY open_time
        """
        return con.execute(query, [last_processed_open_time]).fetchdf()

    if start_time is not None:
        query = f"""
            SELECT *
            FROM {predictions_table}
            WHERE open_time >= ?
            ORDER BY open_time
        """
        return con.execute(query, [start_time]).fetchdf()

    query = f"""
        SELECT *
        FROM {predictions_table}
        ORDER BY open_time
    """
    return con.execute(query).fetchdf()


def append_df_to_table(con, df, table_name, temp_name):
    if df is None or df.empty:
        return
    con.register(temp_name, df)
    con.execute(f"INSERT INTO {table_name} SELECT * FROM {temp_name}")


def export_table_to_csv(con, table_name, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sql_path = str(output_path).replace("\\", "/")
    con.execute(f"""
        COPY (
            SELECT * FROM {table_name}
        ) TO '{sql_path}' WITH (HEADER, DELIMITER ',')
    """)


def calculate_metrics(con, trades_table, strategy_name, current_equity, last_processed_open_time, has_open_position):
    trades_df = con.execute(f"""
        SELECT *
        FROM {trades_table}
        WHERE strategy_name = ?
        ORDER BY exit_time
    """, [strategy_name]).fetchdf()

    if trades_df.empty:
        return pd.DataFrame([{
            "run_ts": utc_now(),
            "strategy_name": strategy_name,
            "last_processed_open_time": last_processed_open_time,
            "current_equity": float(current_equity),
            "total_trades": 0,
            "open_positions": int(has_open_position),
            "win_rate": 0.0,
            "avg_net_return": 0.0,
            "total_net_return": 0.0,
            "max_drawdown": 0.0,
            "gross_profit": 0.0,
            "gross_loss": 0.0,
            "profit_factor": 0.0,
            "expectancy_dollars": 0.0,
        }])

    initial_equity = None
    if len(trades_df) > 0:
        first_trade = trades_df.iloc[0]
        initial_equity = float(first_trade["account_equity_after"] - first_trade["pnl_dollars"])
    if initial_equity is None or initial_equity <= 0:
        initial_equity = float(current_equity)

    trades_df["equity_curve"] = trades_df["account_equity_after"] / initial_equity - 1.0
    trades_df["running_peak"] = trades_df["equity_curve"].cummax()
    trades_df["drawdown"] = trades_df["equity_curve"] - trades_df["running_peak"]

    gross_profit = float(trades_df.loc[trades_df["pnl_dollars"] > 0, "pnl_dollars"].sum())
    gross_loss = float(abs(trades_df.loc[trades_df["pnl_dollars"] < 0, "pnl_dollars"].sum()))
    profit_factor = 999.0 if gross_loss == 0 and gross_profit > 0 else (gross_profit / gross_loss if gross_loss > 0 else 0.0)

    return pd.DataFrame([{
        "run_ts": utc_now(),
        "strategy_name": strategy_name,
        "last_processed_open_time": last_processed_open_time,
        "current_equity": float(current_equity),
        "total_trades": int(len(trades_df)),
        "open_positions": int(has_open_position),
        "win_rate": float((trades_df["net_return"] > 0).mean()),
        "avg_net_return": float(trades_df["net_return"].mean()),
        "total_net_return": float(current_equity / initial_equity - 1.0),
        "max_drawdown": float(trades_df["drawdown"].min()),
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "profit_factor": float(profit_factor),
        "expectancy_dollars": float(trades_df["pnl_dollars"].mean()),
    }])


def blank_open_position_fields(state):
    state["open_trade_id"] = None
    state["open_entry_time"] = None
    state["open_entry_price"] = None
    state["open_fill_entry"] = None
    state["open_pred_proba"] = None
    state["open_pred_label"] = None
    state["open_actual_label"] = None
    state["open_vol20"] = None
    state["open_stop_pct"] = None
    state["open_take_profit_pct"] = None
    state["open_stop_price"] = None
    state["open_take_profit_price"] = None
    state["open_qty"] = None
    state["open_bars_held"] = 0
    return state


def main():
    project_root = Path.cwd()
    print(f"[INFO] Current working directory: {project_root}")
    print(f"[INFO] Project root resolved to: {project_root}")

    data_cfg = load_yaml(project_root / "configs" / "data_sources.yaml")
    paper_cfg = load_yaml(project_root / "configs" / "paper_trading.yaml")

    db_path_cfg = data_cfg["storage"]["db_path"]
    db_path = (project_root / db_path_cfg).resolve()
    print(f"[INFO] Resolved DB path: {db_path}")

    strategy_name = paper_cfg["strategy"]["strategy_name"]
    threshold = float(paper_cfg["strategy"]["entry_threshold"])
    required_regime = str(paper_cfg["strategy"]["required_regime"])
    required_label = int(paper_cfg["strategy"]["required_label"])

    initial_equity = float(paper_cfg["risk"]["initial_equity"])
    risk_pct = float(paper_cfg["risk"]["risk_pct"])
    fee_bps = float(paper_cfg["risk"]["fee_bps"])
    slippage_bps = float(paper_cfg["risk"]["slippage_bps"])
    stop_vol_mult = float(paper_cfg["risk"]["stop_vol_mult"])
    rr_multiple = float(paper_cfg["risk"]["rr_multiple"])
    max_hold_bars = int(paper_cfg["risk"]["max_hold_bars"])
    min_stop_pct = float(paper_cfg["risk"]["min_stop_pct"])

    predictions_table = paper_cfg["data"]["predictions_table"]
    start_time = paper_cfg["data"]["start_time"]

    output_dir = (project_root / paper_cfg["storage"]["output_dir"]).resolve()
    runtime_table = paper_cfg["storage"]["runtime_table"]
    signals_table = paper_cfg["storage"]["signals_table"]
    trades_table = paper_cfg["storage"]["trades_table"]
    metrics_table = paper_cfg["storage"]["metrics_table"]

    allow_same_bar_reentry = bool(paper_cfg["runtime"]["allow_same_bar_reentry"])
    export_csv_each_run = bool(paper_cfg["runtime"]["export_csv_each_run"])

    output_dir.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db_path))

    ensure_tables(con, runtime_table, signals_table, trades_table, metrics_table)

    table_check = con.execute("""
        SELECT COUNT(*)
        FROM information_schema.tables
        WHERE table_name = ?
    """, [predictions_table]).fetchone()[0]

    if table_check == 0:
        raise ValueError(f"Missing predictions table: {predictions_table}")

    state = get_runtime_state(con, runtime_table, strategy_name, initial_equity)
    print(f"[INFO] Loaded runtime state for strategy: {strategy_name}")
    print(f"[INFO] Last processed open_time: {state['last_processed_open_time']}")
    print(f"[INFO] Current equity: {state['current_equity']}")
    print(f"[INFO] Has open trade: {state['open_trade_id'] is not None}")

    rows = fetch_new_rows(
        con=con,
        predictions_table=predictions_table,
        last_processed_open_time=state["last_processed_open_time"],
        start_time=start_time,
    )

    print(f"[INFO] New prediction rows fetched: {len(rows)}")

    if rows.empty:
        metrics_df = calculate_metrics(
            con=con,
            trades_table=trades_table,
            strategy_name=strategy_name,
            current_equity=state["current_equity"],
            last_processed_open_time=state["last_processed_open_time"],
            has_open_position=state["open_trade_id"] is not None,
        )
        append_df_to_table(con, metrics_df, metrics_table, "metrics_append_df")

        if export_csv_each_run:
            export_table_to_csv(con, runtime_table, output_dir / f"{runtime_table}.csv")
            export_table_to_csv(con, signals_table, output_dir / f"{signals_table}.csv")
            export_table_to_csv(con, trades_table, output_dir / f"{trades_table}.csv")
            export_table_to_csv(con, metrics_table, output_dir / f"{metrics_table}.csv")

        print("[INFO] No new rows to process.")
        print(metrics_df.to_string(index=False))
        con.close()
        return

    rows = rows.copy()
    rows, prob_source, label_source, actual_source = normalize_prediction_columns(rows)

    print(f"[INFO] Probability source column: {prob_source}")
    print(f"[INFO] Label source column: {label_source}")
    print(f"[INFO] Actual label source column: {actual_source}")

    if "vol20" not in rows.columns or rows["vol20"].isna().all():
        rows["vol20"] = build_volatility_proxy(rows, price_col="close", window=20)
    else:
        fallback_vol = build_volatility_proxy(rows, price_col="close", window=20)
        rows["vol20"] = rows["vol20"].fillna(fallback_vol)

    rows["open_time"] = pd.to_datetime(rows["open_time"])
    rows["specialist_pred_proba"] = pd.to_numeric(rows["specialist_pred_proba"], errors="coerce")
    rows["specialist_pred_label"] = pd.to_numeric(rows["specialist_pred_label"], errors="coerce")
    rows["actual_label"] = pd.to_numeric(rows["actual_label"], errors="coerce")
    rows["close"] = pd.to_numeric(rows["close"], errors="coerce")
    rows["vol20"] = pd.to_numeric(rows["vol20"], errors="coerce")

    run_ts = utc_now()
    signals = []
    closed_trades = []

    fee = fee_bps / 10000.0
    slip = slippage_bps / 10000.0

    for _, row in rows.iterrows():
        row_time = row["open_time"]
        regime_label = str(row["regime_label"])
        close_px = safe_float(row["close"])
        pred_proba = safe_float(row["specialist_pred_proba"])
        pred_label = int(row["specialist_pred_label"]) if not pd.isna(row["specialist_pred_label"]) else None
        actual_label = int(row["actual_label"]) if not pd.isna(row["actual_label"]) else None

        if not np.isfinite(close_px):
            state["last_processed_open_time"] = row_time
            continue

        current_equity_before = float(state["current_equity"])
        closed_this_bar = False

        if state["open_trade_id"] is not None:
            state["open_bars_held"] = int(state["open_bars_held"]) + 1
            exit_reason = None
            raw_exit_price = None

            if close_px <= float(state["open_stop_price"]):
                raw_exit_price = float(state["open_stop_price"])
                exit_reason = "stop_loss"
            elif close_px >= float(state["open_take_profit_price"]):
                raw_exit_price = float(state["open_take_profit_price"])
                exit_reason = "take_profit"
            elif int(state["open_bars_held"]) >= max_hold_bars:
                raw_exit_price = float(close_px)
                exit_reason = "time_exit"

            if exit_reason is not None:
                fill_exit = raw_exit_price * (1 - slip)
                fill_entry = float(state["open_fill_entry"])
                gross_return = (fill_exit - fill_entry) / fill_entry
                net_return = gross_return - (2 * fee)
                pnl_dollars = float(state["open_qty"]) * float(state["open_entry_price"]) * net_return
                state["current_equity"] = float(state["current_equity"]) + pnl_dollars

                closed_trades.append({
                    "run_ts": run_ts,
                    "trade_id": state["open_trade_id"],
                    "strategy_name": strategy_name,
                    "entry_time": state["open_entry_time"],
                    "exit_time": row_time,
                    "side": "long",
                    "regime_label": required_regime,
                    "threshold": threshold,
                    "entry_price": float(state["open_entry_price"]),
                    "exit_price": raw_exit_price,
                    "fill_entry": fill_entry,
                    "fill_exit": fill_exit,
                    "specialist_pred_proba": float(state["open_pred_proba"]),
                    "specialist_pred_label": int(state["open_pred_label"]),
                    "actual_label": state["open_actual_label"],
                    "vol20": float(state["open_vol20"]),
                    "stop_pct": float(state["open_stop_pct"]),
                    "take_profit_pct": float(state["open_take_profit_pct"]),
                    "stop_price": float(state["open_stop_price"]),
                    "take_profit_price": float(state["open_take_profit_price"]),
                    "qty": float(state["open_qty"]),
                    "bars_held": int(state["open_bars_held"]),
                    "exit_reason": exit_reason,
                    "gross_return": float(gross_return),
                    "net_return": float(net_return),
                    "pnl_dollars": float(pnl_dollars),
                    "account_equity_after": float(state["current_equity"]),
                })

                state = blank_open_position_fields(state)
                closed_this_bar = True

        decision = "reject"
        reject_reason = None

        can_enter_after_close = allow_same_bar_reentry or (not closed_this_bar)

        if state["open_trade_id"] is not None:
            decision = "reject"
            reject_reason = "position_already_open"
        elif not can_enter_after_close:
            decision = "reject"
            reject_reason = "same_bar_reentry_blocked"
        elif regime_label != required_regime:
            decision = "reject"
            reject_reason = "non_required_regime"
        elif pred_label is None or pd.isna(pred_proba):
            decision = "reject"
            reject_reason = "missing_prediction"
        elif pred_label != required_label:
            decision = "reject"
            reject_reason = "pred_label_not_required_label"
        elif pred_proba < threshold:
            decision = "reject"
            reject_reason = "below_threshold"
        else:
            vol = safe_float(row["vol20"], default=0.01)
            if not np.isfinite(vol) or vol <= 0:
                vol = 0.01

            stop_pct = max(stop_vol_mult * vol, min_stop_pct)
            take_profit_pct = stop_pct * rr_multiple
            stop_price = close_px * (1 - stop_pct)
            take_profit_price = close_px * (1 + take_profit_pct)
            stop_distance_dollars = close_px * stop_pct

            if stop_distance_dollars > 0:
                risk_dollars = float(state["current_equity"]) * risk_pct
                qty = risk_dollars / stop_distance_dollars
                fill_entry = close_px * (1 + slip)

                state["open_trade_id"] = str(uuid.uuid4())
                state["open_entry_time"] = row_time
                state["open_entry_price"] = float(close_px)
                state["open_fill_entry"] = float(fill_entry)
                state["open_pred_proba"] = float(pred_proba)
                state["open_pred_label"] = int(pred_label)
                state["open_actual_label"] = actual_label
                state["open_vol20"] = float(vol)
                state["open_stop_pct"] = float(stop_pct)
                state["open_take_profit_pct"] = float(take_profit_pct)
                state["open_stop_price"] = float(stop_price)
                state["open_take_profit_price"] = float(take_profit_price)
                state["open_qty"] = float(qty)
                state["open_bars_held"] = 0

                decision = "enter_trade"
                reject_reason = None
            else:
                decision = "reject"
                reject_reason = "invalid_stop_distance"

        state["last_processed_open_time"] = row_time
        state["updated_at"] = run_ts

        signals.append({
            "run_ts": run_ts,
            "strategy_name": strategy_name,
            "open_time": row_time,
            "regime_label": regime_label,
            "close": float(close_px),
            "specialist_pred_proba": pred_proba if not pd.isna(pred_proba) else np.nan,
            "specialist_pred_label": pred_label if pred_label is not None else np.nan,
            "actual_label": actual_label if actual_label is not None else np.nan,
            "decision": decision,
            "reject_reason": reject_reason,
            "threshold": threshold,
            "required_regime": required_regime,
            "required_label": required_label,
            "current_equity_before": current_equity_before,
            "current_equity_after": float(state["current_equity"]),
        })

    signals_df = pd.DataFrame(signals)
    trades_df = pd.DataFrame(closed_trades)

    append_df_to_table(con, signals_df, signals_table, "signals_append_df")
    append_df_to_table(con, trades_df, trades_table, "trades_append_df")
    upsert_runtime_state(con, runtime_table, state)

    metrics_df = calculate_metrics(
        con=con,
        trades_table=trades_table,
        strategy_name=strategy_name,
        current_equity=state["current_equity"],
        last_processed_open_time=state["last_processed_open_time"],
        has_open_position=state["open_trade_id"] is not None,
    )
    append_df_to_table(con, metrics_df, metrics_table, "metrics_append_df")

    if export_csv_each_run:
        export_table_to_csv(con, runtime_table, output_dir / f"{runtime_table}.csv")
        export_table_to_csv(con, signals_table, output_dir / f"{signals_table}.csv")
        export_table_to_csv(con, trades_table, output_dir / f"{trades_table}.csv")
        export_table_to_csv(con, metrics_table, output_dir / f"{metrics_table}.csv")

    print(f"[INFO] Processed rows: {len(rows)}")
    print(f"[INFO] Signals logged this run: {len(signals_df)}")
    print(f"[INFO] Trades closed this run: {len(trades_df)}")
    print(f"[INFO] Open trade after run: {state['open_trade_id'] is not None}")
    print(f"[INFO] Current equity after run: {state['current_equity']}")
    print("\n[INFO] Latest metrics:")
    print(metrics_df.to_string(index=False))

    con.close()


if __name__ == "__main__":
    main()