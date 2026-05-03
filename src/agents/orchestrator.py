# =============================================================================
# orchestrator.py
# Master loop for the Claudex agent pipeline
# Called by Windows Task Scheduler every hour
# =============================================================================


import sys
import uuid
import json
import traceback
import yaml
import duckdb
from pathlib import Path
from datetime import datetime, timezone


# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


from src.agents.run_logger import (
    ensure_log_tables,
    log_run,
    log_step,
)
from src.agents.retrain_subagent import run_retrain_subagent



# =============================================================================
# Config Loader
# =============================================================================


def load_config() -> dict:
    config_path = PROJECT_ROOT / "configs" / "agent_config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)



# =============================================================================
# Agent Tool Imports
# =============================================================================


def import_agents():
    """
    Lazy import all agents so a missing agent does not crash the orchestrator
    at startup.
    """
    agents = {}

    try:
        from src.agents.data_agent import run_data_agent
        agents["data_agent"] = run_data_agent
    except ImportError as e:
        agents["data_agent"] = None
        print(f"[WARN] data_agent not available: {e}")

    try:
        from src.agents.feature_agent import run_feature_agent
        agents["feature_agent"] = run_feature_agent
    except ImportError as e:
        agents["feature_agent"] = None
        print(f"[WARN] feature_agent not available: {e}")

    try:
        from src.agents.drift_checker import run_drift_checker
        agents["drift_checker"] = run_drift_checker
    except ImportError as e:
        agents["drift_checker"] = None
        print(f"[WARN] drift_checker not available: {e}")

    try:
        from src.agents.prediction_agent import run_prediction_agent
        agents["prediction_agent"] = run_prediction_agent
    except ImportError as e:
        agents["prediction_agent"] = None
        print(f"[WARN] prediction_agent not available: {e}")

    try:
        from src.agents.paper_trade_agent import run_paper_trade_agent
        agents["paper_trade_agent"] = run_paper_trade_agent
    except ImportError as e:
        agents["paper_trade_agent"] = None
        print(f"[WARN] paper_trade_agent not available: {e}")

    try:
        from src.agents.signal_agent import run_signal_agent
        agents["signal_agent"] = run_signal_agent
    except ImportError as e:
        agents["signal_agent"] = None
        print(f"[WARN] signal_agent not available: {e}")

    try:
        from src.agents.dashboard_agent import run_dashboard_agent
        agents["dashboard_agent"] = run_dashboard_agent
    except ImportError as e:
        agents["dashboard_agent"] = None
        print(f"[WARN] dashboard_agent not available: {e}")

    return agents



# =============================================================================
# Step Runner
# =============================================================================


def run_step(
    con: duckdb.DuckDBPyConnection,
    run_id: str,
    step_name: str,
    agent_fn,
    cfg: dict,
    context: dict,
    abort_on_failure: bool = True,
) -> dict:
    step_start = datetime.now(timezone.utc)
    print(f"\n[STEP] {step_name} starting at {step_start.strftime('%H:%M:%S')} UTC")

    if agent_fn is None:
        step_end = datetime.now(timezone.utc)
        result = {
            "step_name": step_name,
            "status": "skipped",
            "reason": "agent_not_implemented",
            "duration_sec": (step_end - step_start).total_seconds(),
        }
        log_step(con, run_id, step_name, step_start, step_end,
                 status="skipped", error_message="agent_not_implemented",
                 result_json=json.dumps(result))
        print(f"[SKIP] {step_name} — agent not yet implemented")
        return result

    try:
        output = agent_fn(cfg=cfg, context=context)
        step_end = datetime.now(timezone.utc)

        result = {
            "step_name": step_name,
            "status": "success",
            "output": output,
            "duration_sec": (step_end - step_start).total_seconds(),
        }

        log_step(con, run_id, step_name, step_start, step_end,
                 status="success",
                 result_json=json.dumps({
                     k: v for k, v in output.items()
                     if isinstance(v, (str, int, float, bool, type(None)))
                 }))

        print(f"[OK]   {step_name} completed in {result['duration_sec']:.1f}s — {output}")
        return result

    except Exception as e:
        step_end = datetime.now(timezone.utc)
        error_msg = traceback.format_exc()

        result = {
            "step_name": step_name,
            "status": "failed",
            "error": str(e),
            "duration_sec": (step_end - step_start).total_seconds(),
        }

        log_step(con, run_id, step_name, step_start, step_end,
                 status="failed", error_message=error_msg,
                 result_json=json.dumps(result))

        print(f"[FAIL] {step_name} failed: {e}")

        if abort_on_failure:
            raise RuntimeError(f"Pipeline aborted at step [{step_name}]: {e}")

        return result



# =============================================================================
# Predictions Hydration
# Loads rel_score + prediction rows from DuckDB into context after
# prediction_agent runs, so signal_agent can threshold on rel_score.
# =============================================================================


def hydrate_predictions(con: duckdb.DuckDBPyConnection, cfg: dict, context: dict) -> None:
    """
    Reads the predictions table written by prediction_agent and populates
    context["predictions"] as a list of dicts for signal_agent consumption.
    Falls back gracefully — never raises, never blocks the pipeline.
    """
    try:
        pred_table = cfg["tables"]["predictions"]

        # Detect which score columns exist (graceful compat with old schema)
        existing_cols = {
            row[0] for row in con.execute(
                f"SELECT column_name FROM information_schema.columns "
                f"WHERE table_name = '{pred_table}'"
            ).fetchall()
        }

        # Build SELECT dynamically — always include rel_score if present
        select_cols = ["open_time", "close", "pred_label", "pred_proba"]
        if "cal_proba"  in existing_cols: select_cols.append("cal_proba")
        if "rel_score"  in existing_cols: select_cols.append("rel_score")

        rows = con.execute(
            f"SELECT {', '.join(select_cols)} FROM {pred_table} "
            f"ORDER BY open_time DESC"
        ).fetchdf()

        if rows.empty:
            context["predictions"] = []
            print("[ORCH] predictions table is empty — no signals possible this cycle")
            return

        context["predictions"] = [
            {
                "open_time":  str(r["open_time"]),
                "close":      float(r["close"]),
                "pred_label": int(r["pred_label"]),
                "proba":      float(r["pred_proba"]),          # backward compat key
                "cal_proba":  float(r.get("cal_proba", r["pred_proba"])),
                "rel_score":  float(r.get("rel_score", 0.0)), # 0.0 if col missing
                "regime":     "compression",
            }
            for _, r in rows.iterrows()
        ]

        n_signals = sum(1 for p in context["predictions"] if p["rel_score"] >= cfg["signal_agent"]["threshold"])
        print(f"[ORCH] Hydrated {len(context['predictions'])} predictions "
              f"({n_signals} above rel_score >= {cfg['signal_agent']['threshold']})")

    except Exception as e:
        print(f"[WARN] hydrate_predictions failed — signal_agent will get empty list: {e}")
        context["predictions"] = []



# =============================================================================
# Main Orchestrator Loop
# =============================================================================


def main():
    run_id    = str(uuid.uuid4())[:8]
    run_start = datetime.now(timezone.utc)

    print("=" * 70)
    print(f"[ORCHESTRATOR] Run ID: {run_id}")
    print(f"[ORCHESTRATOR] Started: {run_start.strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print("=" * 70)

    cfg             = load_config()
    db_path         = PROJECT_ROOT / cfg["paths"]["db_path"]
    abort_on_failure = cfg["orchestrator"]["abort_on_step_failure"]

    con = duckdb.connect(str(db_path))
    ensure_log_tables(con)

    agents       = import_agents()
    step_results = []
    aborted      = False
    abort_reason = None

    # Shared context dict — all state passes through here and DuckDB
    context = {
        "run_id":          run_id,
        "run_start":       run_start.isoformat(),
        "project_root":    str(PROJECT_ROOT),
        "new_rows_added":  0,
        "drift_detected":  False,
        "retrained":       False,
        "signals_emitted": 0,
        "predictions":     [],   # ← populated by hydrate_predictions() after prediction_agent
        "model_path":      str(PROJECT_ROOT / cfg["prediction_agent"]["model_path"]),
    }

    pipeline_steps = [
        ("data_agent",        agents.get("data_agent")),
        ("feature_agent",     agents.get("feature_agent")),
        ("drift_checker",     agents.get("drift_checker")),
        ("prediction_agent",  agents.get("prediction_agent")),
        ("paper_trade_agent", agents.get("paper_trade_agent")),
        ("signal_agent",      agents.get("signal_agent")),
        ("dashboard_agent",   agents.get("dashboard_agent")),
    ]

    try:
        for step_name, agent_fn in pipeline_steps:
            if step_name in cfg and not cfg[step_name].get("enabled", True):
                print(f"[SKIP] {step_name} — disabled in agent_config.yaml")
                continue

            result = run_step(
                con=con,
                run_id=run_id,
                step_name=step_name,
                agent_fn=agent_fn,
                cfg=cfg,
                context=context,
                abort_on_failure=abort_on_failure,
            )
            step_results.append(result)

            # ── Update shared context from step output ────────────────────────
            if result["status"] == "success" and "output" in result:
                out = result["output"]
                if "new_rows_added"  in out: context["new_rows_added"]   = out["new_rows_added"]
                if "drift_detected"  in out: context["drift_detected"]   = out["drift_detected"]
                if "signals_emitted" in out: context["signals_emitted"] += out["signals_emitted"]

            # ── ADDED: hydrate predictions into context after prediction_agent ─
            if step_name == "prediction_agent" and result["status"] == "success":
                hydrate_predictions(con, cfg, context)

            # ── Reactive retrain subagent — spawned after drift_checker ───────
            if step_name == "drift_checker" and context.get("drift_detected"):
                retrain_step = run_step(
                    con=con,
                    run_id=run_id,
                    step_name="retrain_subagent",
                    agent_fn=run_retrain_subagent,
                    cfg=cfg,
                    context=context,
                    abort_on_failure=False,  # never abort pipeline on retrain failure
                )
                step_results.append(retrain_step)
                if retrain_step["status"] == "success":
                    context["retrained"] = (
                        retrain_step.get("output", {}).get("action") == "model_swapped"
                    )

    except RuntimeError as e:
        aborted      = True
        abort_reason = str(e)
        print(f"\n[ABORT] Pipeline aborted: {abort_reason}")

    finally:
        run_end  = datetime.now(timezone.utc)
        duration = (run_end - run_start).total_seconds()

        log_run(
            con=con,
            run_id=run_id,
            run_start=run_start,
            run_end=run_end,
            step_results=step_results,
            aborted=aborted,
            abort_reason=abort_reason,
            new_rows_added=context.get("new_rows_added", 0),
            signals_emitted=context.get("signals_emitted", 0),
            drift_detected=context.get("drift_detected", False),
            retrained=context.get("retrained", False),
        )

        con.close()

        print("\n" + "=" * 70)
        print(f"[ORCHESTRATOR] Run ID:     {run_id}")
        print(f"[ORCHESTRATOR] Duration:   {duration:.1f}s")
        print(f"[ORCHESTRATOR] Steps:      {len(step_results)}")
        print(f"[ORCHESTRATOR] Passed:     {sum(1 for s in step_results if s['status'] == 'success')}")
        print(f"[ORCHESTRATOR] Skipped:    {sum(1 for s in step_results if s['status'] == 'skipped')}")
        print(f"[ORCHESTRATOR] Failed:     {sum(1 for s in step_results if s['status'] == 'failed')}")
        print(f"[ORCHESTRATOR] Aborted:    {aborted}")
        print(f"[ORCHESTRATOR] New rows:   {context.get('new_rows_added', 0)}")
        print(f"[ORCHESTRATOR] Drift:      {context.get('drift_detected', False)}")
        print(f"[ORCHESTRATOR] Retrained:  {context.get('retrained', False)}")
        print(f"[ORCHESTRATOR] Signals:    {context.get('signals_emitted', 0)}")
        print("=" * 70)


if __name__ == "__main__":
    main()