import sys
import traceback
import duckdb
from pathlib import Path
from datetime import datetime, timezone, timedelta

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


def _ensure_signal_log_table(con, signal_log_table: str):
    con.execute(f"""
        CREATE TABLE IF NOT EXISTS {signal_log_table} (
            open_time   TIMESTAMP,
            rel_score   DOUBLE,
            close       DOUBLE,
            regime      VARCHAR,
            emitted_at  TIMESTAMP,
            run_id      VARCHAR
        )
    """)

    try:
        existing = [r[0] for r in con.execute(
            f"SELECT column_name FROM information_schema.columns WHERE table_name = '{signal_log_table}'"
        ).fetchall()]
        if "run_id" not in existing:
            con.execute(f"ALTER TABLE {signal_log_table} ADD COLUMN run_id VARCHAR")
    except Exception:
        pass


def _is_duplicate(con, open_time, signal_log_table: str, ttl_hours: int = 72) -> bool:
    try:
        cutoff = datetime.now(timezone.utc) - timedelta(hours=ttl_hours)
        row = con.execute(
            f"""
            SELECT COUNT(*)
            FROM {signal_log_table}
            WHERE open_time = ?
              AND emitted_at >= ?
            """,
            [open_time, cutoff.replace(tzinfo=None)],
        ).fetchone()
        return bool(row and row[0] > 0)
    except Exception:
        return False


def _log_signal(con, signal_log_table: str, open_time, rel_score: float,
                close: float, regime: str, emitted_at: datetime, run_id: str):
    con.execute(
        f"""
        INSERT INTO {signal_log_table}
            (open_time, rel_score, close, regime, emitted_at, run_id)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        [open_time, rel_score, close, regime, emitted_at.replace(tzinfo=None), run_id],
    )


def run_signal_agent(cfg: dict, context: dict) -> dict:
    sig_cfg        = cfg.get("signal_agent", {})
    paths          = cfg["paths"]
    tables         = cfg["tables"]
    db_path        = PROJECT_ROOT / paths["db_path"]

    threshold      = float(sig_cfg.get("threshold", 0.78))
    dedup_ttl_hrs  = int(sig_cfg.get("dedup_ttl_hours", 72))
    signal_log_tbl = tables.get("signal_log", "signal_log")

    predictions = context.get("predictions", [])
    run_id      = context.get("run_id", "")

    con = duckdb.connect(str(db_path))
    try:
        _ensure_signal_log_table(con, signal_log_tbl)

        if not predictions:
            return {
                "status": "success",
                "signals_emitted": 0,
                "signal_details": [],
                "message": "No predictions in context",
            }

        print(f"[SIGNAL] {len(predictions)} predictions | rel_score >= {threshold}")
        if predictions:
            sample = predictions[0]
            print(f"[SIGNAL] Sample keys:      {list(sample.keys())}")
            print(f"[SIGNAL] Sample rel_score: {sample.get('rel_score', 'N/A')}")
            print(f"[SIGNAL] Sample proba:     {sample.get('proba', 'N/A')}")

        high_conf = [
            p for p in predictions
            if float(p.get("rel_score", 0.0)) >= threshold
            and int(p.get("pred_label", p.get("label", 0))) == 1
        ]

        if not high_conf:
            return {
                "status": "success",
                "signals_emitted": 0,
                "signal_details": [],
                "message": f"No predictions above rel_score={threshold}",
            }

        emitted = []
        suppressed = []
        now_utc = datetime.now(timezone.utc)

        for p in high_conf:
            open_time = p.get("open_time")
            rel_score = float(p.get("rel_score", 0.0))
            close     = float(p.get("close", 0.0))
            regime    = str(p.get("regime", ""))

            if _is_duplicate(con, open_time, signal_log_tbl, dedup_ttl_hrs):
                suppressed.append(open_time)
                print(f"[SIGNAL] Duplicate suppressed — {open_time} already sent within last {dedup_ttl_hrs}h")
                continue

            _log_signal(con, signal_log_tbl, open_time, rel_score, close, regime, now_utc, run_id)
            emitted.append({
                "open_time": str(open_time),
                "rel_score": round(rel_score, 6),
                "close": close,
                "regime": regime,
            })
            print(f"[SIGNAL] EMITTED — {open_time} | rel={rel_score:.4f} | close={close} | regime={regime}")

        msg = (
            f"Emitted {len(emitted)} signal(s)"
            if emitted else
            f"All {len(high_conf)} high-conf signals suppressed as duplicates (TTL={dedup_ttl_hrs}h)"
        )

        return {
            "status": "success",
            "signals_emitted": len(emitted),
            "signal_details": emitted,
            "suppressed": len(suppressed),
            "message": msg,
        }

    except Exception:
        return {
            "status": "failed",
            "error": traceback.format_exc(),
            "signals_emitted": 0,
            "signal_details": [],
        }
    finally:
        con.close()