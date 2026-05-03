# =============================================================================
# run_logger.py
# Writes per-run and per-step audit logs to DuckDB
# Every orchestrator cycle calls this at the end
# =============================================================================

from pathlib import Path
from datetime import datetime, timezone
import duckdb
import yaml
import traceback


def load_config(config_path: str = "configs/agent_config.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_log_tables(con: duckdb.DuckDBPyConnection) -> None:
    con.execute("""
        CREATE TABLE IF NOT EXISTS run_log (
            run_id          VARCHAR,
            run_start_utc   TIMESTAMP,
            run_end_utc     TIMESTAMP,
            total_steps     INTEGER,
            steps_passed    INTEGER,
            steps_failed    INTEGER,
            aborted         BOOLEAN,
            abort_reason    VARCHAR,
            new_rows_added  INTEGER,
            signals_emitted INTEGER,
            drift_detected  BOOLEAN,
            retrained       BOOLEAN,
            notes           VARCHAR
        )
    """)

    con.execute("""
        CREATE TABLE IF NOT EXISTS agent_step_log (
            run_id          VARCHAR,
            step_name       VARCHAR,
            step_start_utc  TIMESTAMP,
            step_end_utc    TIMESTAMP,
            status          VARCHAR,
            error_message   VARCHAR,
            result_json     VARCHAR
        )
    """)

    con.execute("""
        CREATE TABLE IF NOT EXISTS signal_log (
            signal_id           VARCHAR,
            signal_time_utc     TIMESTAMP,
            regime_label        VARCHAR,
            pred_proba          DOUBLE,
            pred_label          INTEGER,
            open_time           TIMESTAMP,
            close_price         DOUBLE,
            email_sent          BOOLEAN,
            toast_sent          BOOLEAN,
            notes               VARCHAR
        )
    """)

    con.execute("""
        CREATE TABLE IF NOT EXISTS drift_log (
            run_id              VARCHAR,
            check_time_utc      TIMESTAMP,
            current_pf          DOUBLE,
            baseline_pf         DOUBLE,
            current_win_rate    DOUBLE,
            baseline_win_rate   DOUBLE,
            drift_detected      BOOLEAN,
            trades_evaluated    INTEGER,
            action_taken        VARCHAR
        )
    """)

    con.execute("""
        CREATE TABLE IF NOT EXISTS model_registry (
            model_id            VARCHAR,
            model_version       VARCHAR,
            model_path          VARCHAR,
            trained_at_utc      TIMESTAMP,
            promoted_at_utc     TIMESTAMP,
            oos_profit_factor   DOUBLE,
            oos_win_rate        DOUBLE,
            oos_total_return    DOUBLE,
            oos_max_drawdown    DOUBLE,
            is_active           BOOLEAN,
            notes               VARCHAR
        )
    """)


def log_run(
    con, run_id, run_start, run_end, step_results,
    aborted=False, abort_reason=None, new_rows_added=0,
    signals_emitted=0, drift_detected=False, retrained=False, notes=None,
):
    steps_passed = sum(1 for s in step_results if s.get("status") == "success")
    steps_failed = sum(1 for s in step_results if s.get("status") == "failed")
    con.execute("""
        INSERT INTO run_log VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, [
        str(run_id),
        run_start,
        run_end,
        int(len(step_results)),
        int(steps_passed),
        int(steps_failed),
        bool(aborted),
        str(abort_reason) if abort_reason else None,
        int(new_rows_added),
        int(signals_emitted),
        bool(drift_detected),
        bool(retrained),
        str(notes) if notes else None,
    ])


def log_step(
    con, run_id, step_name, step_start, step_end,
    status, error_message=None, result_json=None,
):
    con.execute("""
        INSERT INTO agent_step_log VALUES (?, ?, ?, ?, ?, ?, ?)
    """, [
        str(run_id), str(step_name), step_start, step_end,
        str(status),
        str(error_message) if error_message else None,
        str(result_json) if result_json else None,
    ])


def log_signal(
    con, signal_id, signal_time, regime_label, pred_proba,
    pred_label, open_time, close_price, email_sent, toast_sent, notes=None,
):
    con.execute("""
        INSERT INTO signal_log VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, [
        str(signal_id), signal_time, str(regime_label),
        float(pred_proba), int(pred_label), open_time,
        float(close_price), bool(email_sent), bool(toast_sent),
        str(notes) if notes else None,
    ])


def log_drift(
    con, run_id, check_time, current_pf, baseline_pf,
    current_win_rate, baseline_win_rate, drift_detected,
    trades_evaluated, action_taken,
):
    con.execute("""
        INSERT INTO drift_log VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, [
        str(run_id), check_time,
        float(current_pf), float(baseline_pf),
        float(current_win_rate), float(baseline_win_rate),
        bool(drift_detected), int(trades_evaluated),
        str(action_taken),
    ])


def log_model_promotion(
    con, model_id, model_version, model_path,
    trained_at, promoted_at, oos_pf, oos_win_rate,
    oos_return, oos_drawdown, notes=None,
):
    con.execute("UPDATE model_registry SET is_active = FALSE")
    con.execute("""
        INSERT INTO model_registry VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, [
        str(model_id), str(model_version), str(model_path),
        trained_at, promoted_at,
        float(oos_pf), float(oos_win_rate),
        float(oos_return), float(oos_drawdown),
        True,
        str(notes) if notes else None,
    ])


if __name__ == "__main__":
    cfg     = load_config()
    db_path = Path(cfg["paths"]["project_root"]) / cfg["paths"]["db_path"]
    con     = duckdb.connect(str(db_path))
    ensure_log_tables(con)
    print("[INFO] All log tables created/verified successfully")
    tables     = con.execute("SHOW TABLES").df()
    log_tables = ["run_log", "agent_step_log", "signal_log", "drift_log", "model_registry"]
    for t in log_tables:
        exists = t in tables["name"].tolist()
        print(f"  Table {t}: {'FOUND' if exists else 'MISSING'}")
    con.close()
