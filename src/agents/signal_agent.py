# =============================================================================
# signal_agent.py — M8 Signal Emitter
# Emits high-confidence trade signals via email + Windows Toast
# Only fires when calibrated probability >= 0.85
# =============================================================================

import smtplib
import traceback
from email.mime.text import MIMEText
from datetime import datetime, timezone


def _send_email(cfg: dict, subject: str, body: str):
    email_cfg = cfg["signal_agent"]["email"]
    msg = MIMEText(body, "html")
    msg["Subject"] = subject
    msg["From"]    = email_cfg["sender"]
    msg["To"]      = email_cfg["recipient"]
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(email_cfg["sender"], email_cfg["app_password"])
        server.sendmail(email_cfg["sender"], email_cfg["recipient"], msg.as_string())


def _send_toast(title: str, message: str, duration: int = 10):
    try:
        from win10toast import ToastNotifier
        ToastNotifier().show_toast(title, message, duration=duration, threaded=True)
    except Exception:
        pass  # Toast failure never blocks the pipeline


def run_signal_agent(cfg: dict, context: dict) -> dict:
    predictions = context.get("predictions", [])
    threshold   = cfg.get("signal_agent", {}).get("threshold", 0.85)
    email_cfg   = cfg.get("signal_agent", {}).get("email", {})
    toast_cfg   = cfg.get("signal_agent", {}).get("toast", {})

    high_conf = [p for p in predictions if p.get("proba", 0) >= threshold]

    if not high_conf:
        return {
            "status":          "success",
            "signals_emitted": 0,
            "signal_details":  [],
            "message":         f"No predictions above threshold {threshold}",
        }

    signals_emitted = 0
    signal_details  = []
    now_utc         = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    for signal in high_conf:
        proba     = round(signal["proba"], 4)
        open_time = signal.get("open_time", "unknown")
        regime    = signal.get("regime", "compression")

        subject = f"🚨 Claudex Signal — ETHUSDT {proba*100:.1f}% confidence"
        body = f"""
        <h2>⚡ High-Confidence Trade Signal</h2>
        <table>
          <tr><td><b>Symbol</b></td><td>ETHUSDT</td></tr>
          <tr><td><b>Regime</b></td><td>{regime}</td></tr>
          <tr><td><b>Confidence</b></td><td>{proba*100:.1f}%</td></tr>
          <tr><td><b>Threshold</b></td><td>{threshold*100:.0f}%</td></tr>
          <tr><td><b>Signal Time</b></td><td>{open_time}</td></tr>
          <tr><td><b>Emitted At</b></td><td>{now_utc}</td></tr>
        </table>
        <p><i>This is a paper trading signal. Verify before any real action.</i></p>
        """

        # ── Email ──────────────────────────────────────────────────────────
        if email_cfg.get("enabled"):
            try:
                _send_email(cfg, subject, body)
                print(f"[SIGNAL] Email sent — {proba*100:.1f}% confidence at {open_time}")
            except Exception as e:
                print(f"[SIGNAL] Email failed: {e}")

        # ── Toast ──────────────────────────────────────────────────────────
        if toast_cfg.get("enabled"):
            _send_toast(
                "⚡ Claudex Signal",
                f"ETHUSDT {proba*100:.1f}% confidence — check email",
                duration=toast_cfg.get("duration", 10),
            )
            print(f"[SIGNAL] Toast sent")

        signals_emitted += 1
        signal_details.append({"open_time": open_time, "proba": proba, "regime": regime})

    return {
        "status":          "success",
        "signals_emitted": signals_emitted,
        "signal_details":  signal_details,
        "message":         f"{signals_emitted} signal(s) emitted above threshold {threshold}",
    }