# signal_agent.py — fixed threshold key + replaced win10toast with plyer
import traceback
import smtplib
import duckdb
from email.mime.text import MIMEText
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def send_email(cfg: dict, subject: str, body: str):
    email_cfg = cfg["signal_agent"]["email"]
    msg = MIMEText(body, "html")
    msg["Subject"] = subject
    msg["From"]    = email_cfg["sender"]
    msg["To"]      = email_cfg["recipient"]
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(email_cfg["sender"], email_cfg["app_password"])
        server.sendmail(email_cfg["sender"], email_cfg["recipient"], msg.as_string())


def send_toast(title: str, message: str, duration: int = 10):
    try:
        from plyer import notification
        notification.notify(
            title=title,
            message=message,
            app_name="Claudex",
            timeout=duration,
        )
    except Exception as e:
        print(f"[SIGNAL] Toast failed (non-fatal): {e}")


def _ensure_signal_log(con):
    """
    Drop and recreate signal_log if it has the old 10-column M5 schema.
    The correct schema has exactly 5 columns. Safe to run every call.
    """
    existing_cols = con.execute(
        "SELECT COUNT(*) FROM information_schema.columns "
        "WHERE table_name = 'signal_log'"
    ).fetchone()[0]

    if existing_cols != 5:
        con.execute("DROP TABLE IF EXISTS signal_log")
        print("[SIGNAL] Migrated signal_log — old schema replaced")

    con.execute("""
        CREATE TABLE IF NOT EXISTS signal_log (
            open_time  TIMESTAMP,
            rel_score  DOUBLE,
            close      DOUBLE,
            regime     VARCHAR,
            emitted_at TIMESTAMP
        )
    """)


def _is_duplicate(db_path: Path, open_time) -> bool:
    """Returns True if this open_time was already emitted in signal_log."""
    try:
        con = duckdb.connect(str(db_path))
        _ensure_signal_log(con)
        count = con.execute(
            "SELECT COUNT(*) FROM signal_log WHERE open_time = ?",
            [open_time]
        ).fetchone()[0]
        con.close()
        return count > 0
    except Exception as e:
        print(f"[SIGNAL] Dedup check failed (non-fatal): {e}")
        return False   # allow signal through if check errors


def _log_signal(db_path: Path, open_time, rel_score: float,
                close: float, regime: str):
    """Persist emitted signal to signal_log for deduplication."""
    try:
        con = duckdb.connect(str(db_path))
        _ensure_signal_log(con)
        con.execute(
            "INSERT INTO signal_log (open_time, rel_score, close, regime, emitted_at) "
            "VALUES (?, ?, ?, ?, ?)",
            [
                open_time,
                float(rel_score),
                float(close),
                str(regime),
                datetime.now(timezone.utc).replace(tzinfo=None),
            ]
        )
        con.close()
    except Exception as e:
        print(f"[SIGNAL] Signal log write failed (non-fatal): {e}")


def run_signal_agent(cfg: dict, context: dict) -> dict:
    predictions = context.get("predictions", [])
    sa_cfg      = cfg.get("signal_agent", {})
    paths       = cfg.get("paths", {})

    db_path   = PROJECT_ROOT / paths.get("db_path", "data/market.duckdb")
    threshold = float(sa_cfg.get("threshold", 0.78))
    score_col = sa_cfg.get("score_col", "rel_score")
    email_cfg = sa_cfg.get("email", {})
    toast_cfg = sa_cfg.get("toast", {})

    print(f"[SIGNAL] {len(predictions)} predictions | {score_col} >= {threshold}")
    if predictions:
        sample = predictions[0]
        print(f"[SIGNAL] Sample keys:      {list(sample.keys())}")
        print(f"[SIGNAL] Sample rel_score: {sample.get('rel_score', 'MISSING')}")
        print(f"[SIGNAL] Sample proba:     {sample.get('proba', 'MISSING')}")

    # Gate on rel_score (percentile rank), fall back to raw proba
    high_conf = [
        p for p in predictions
        if p.get(score_col, p.get("proba", 0.0)) >= threshold
        and p.get("pred_label", 0) == 1
    ]

    if not high_conf:
        return {
            "status":          "success",
            "signals_emitted": 0,
            "signal_details":  [],
            "message":         f"No signals above {score_col} >= {threshold}",
        }

    # Emit at most 1 signal per run — highest scoring row
    best = max(high_conf, key=lambda p: p.get(score_col, p.get("proba", 0.0)))

    score_val = best.get(score_col, best.get("proba", 0.0))
    open_time = best.get("open_time", "unknown")
    regime    = best.get("regime", "compression")
    close     = best.get("close", 0.0)

    # ── Deduplication: suppress if this bar was already signalled ────────
    if _is_duplicate(db_path, open_time):
        print(f"[SIGNAL] Duplicate suppressed — {open_time} already sent")
        return {
            "status":          "success",
            "signals_emitted": 0,
            "signal_details":  [],
            "message":         f"Duplicate suppressed — {open_time} already sent",
        }

    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    subject = f"🚨 Claudex Signal — ETHUSDT {score_val*100:.1f}% rel-score"
    body = f"""
    <h2>High-Confidence Trade Signal</h2>
    <table>
      <tr><td><b>Symbol</b></td><td>ETHUSDT</td></tr>
      <tr><td><b>Regime</b></td><td>{regime}</td></tr>
      <tr><td><b>Rel Score</b></td><td>{score_val*100:.1f}%</td></tr>
      <tr><td><b>Raw Proba</b></td><td>{best.get('proba', 0)*100:.1f}%</td></tr>
      <tr><td><b>Close Price</b></td><td>{close}</td></tr>
      <tr><td><b>Signal Time</b></td><td>{open_time}</td></tr>
      <tr><td><b>Emitted At</b></td><td>{now_utc}</td></tr>
      <tr><td><b>High-conf count</b></td><td>{len(high_conf)} rows this batch</td></tr>
    </table>
    <p><i>This is a paper trading signal. Verify before any real action.</i></p>
    """

    if email_cfg.get("enabled"):
        try:
            send_email(cfg, subject, body)
            print(f"[SIGNAL] Email sent — {score_val*100:.1f}% rel-score at {open_time}")
        except Exception as e:
            print(f"[SIGNAL] Email failed: {e}")

    if toast_cfg.get("enabled"):
        send_toast(
            title="Claudex Signal",
            message=f"ETHUSDT {score_val*100:.1f}% rel-score — check email",
            duration=toast_cfg.get("duration", 10),
        )
        print("[SIGNAL] Toast sent")

    # Persist to signal_log so next run skips this bar
    _log_signal(db_path, open_time, score_val, close, regime)

    return {
        "status":          "success",
        "signals_emitted": 1,
        "signal_details":  [{
            "open_time":   open_time,
            "proba":       round(best.get("proba", 0.0), 4),
            "rel_score":   round(score_val, 4),
            "regime":      regime,
            "close":       close,
            "batch_count": len(high_conf),
        }],
        "message": f"1 signal emitted (best of {len(high_conf)} above {score_col} >= {threshold})",
    }