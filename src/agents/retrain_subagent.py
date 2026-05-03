# =============================================================================
# retrain_subagent.py — Self-Improvement Agent (Enhanced: baseline auto-update)
# After a successful model swap, persists new OOS PF as the drift baseline
# =============================================================================

import sys
import traceback
import joblib
import json
import duckdb
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timezone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from scipy.stats import rankdata

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

CONFIG_PATH      = PROJECT_ROOT / "configs" / "agent_config.yaml"
LAST_RETRAIN_TS  = PROJECT_ROOT / "models" / "artifacts" / ".last_retrain_ts"


def _update_baseline_in_config(new_pf: float, new_winrate: float, new_oos_return: float):
    cfg = yaml.safe_load(CONFIG_PATH.read_text())
    if cfg.get("drift_checker", {}).get("baseline_pf_locked", False):
        print("RETRAIN  Baseline locked — skipping auto-update")
        return False

    old_pf = cfg["drift_checker"]["baseline_profit_factor"]
    cfg["drift_checker"]["baseline_profit_factor"] = round(new_pf, 6)
    cfg["drift_checker"]["baseline_win_rate"]       = round(new_winrate, 6)
    cfg["drift_checker"]["baseline_oos_return"]     = round(new_oos_return, 6)
    cfg["drift_checker"]["baseline_last_updated"]   = datetime.now(timezone.utc).isoformat()

    CONFIG_PATH.write_text(yaml.dump(cfg, default_flow_style=False, sort_keys=False))
    print(f"RETRAIN  Baseline auto-updated: PF {old_pf:.4f} → {new_pf:.4f}")
    return True


def _log_feature_importance(db_path: Path, run_id: str,
                             model, feature_cols: list,
                             new_pf: float, oos_trades: int):
    try:
        importances = model.feature_importances_
    except AttributeError:
        print("RETRAIN  feature_importances_ unavailable — logging uniform weights")
        importances = np.ones(len(feature_cols)) / len(feature_cols)

    records = []
    for rank, (col, score) in enumerate(
        sorted(zip(feature_cols, importances), key=lambda x: x[1], reverse=True), start=1
    ):
        records.append({
            "run_id":        run_id,
            "logged_at":     datetime.now(timezone.utc).isoformat(),
            "model_oos_pf":  round(new_pf, 6),
            "oos_trades":    oos_trades,
            "rank":          rank,
            "feature_name":  col,
            "importance":    round(float(score), 8),
        })

    df = pd.DataFrame(records)
    con = duckdb.connect(str(db_path))
    con.execute("""
        CREATE TABLE IF NOT EXISTS retrain_feature_importance_log (
            run_id       VARCHAR,
            logged_at    VARCHAR,
            model_oos_pf DOUBLE,
            oos_trades   INTEGER,
            rank         INTEGER,
            feature_name VARCHAR,
            importance   DOUBLE
        )
    """)
    con.register("fi_df", df)
    con.execute("INSERT INTO retrain_feature_importance_log SELECT * FROM fi_df")
    con.close()
    print(f"RETRAIN  Feature importance logged: {len(records)} features")
    print(f"RETRAIN  Top-5: {[r['feature_name'] for r in records[:5]]}")


def run_retrain_subagent(cfg: dict, context: dict) -> dict:
    """
    Typed tool contract
    INPUT : cfg     — full agent_config.yaml as dict
    INPUT : context — shared orchestrator context dict
    OUTPUT: dict with status, action, new_model_pf, error
    """
    rtcfg   = cfg["retrain_subagent"]
    paths   = cfg["paths"]
    db_path = PROJECT_ROOT / paths["db_path"]

    min_train_rows    = rtcfg.get("min_train_rows", 400)
    promotion_min_pf  = rtcfg.get("promotion_min_pf", 1.20)
    oos_test_pct      = rtcfg.get("oos_test_pct", 0.20)
    max_iter          = rtcfg.get("max_iter", 300)
    learning_rate     = rtcfg.get("learning_rate", 0.05)
    max_leaf_nodes    = rtcfg.get("max_leaf_nodes", 31)
    cooldown_hours    = rtcfg.get("retrain_cooldown_hours", 24)

    # ── Cooldown check: skip if retrained less than cooldown_hours ago ────
    if LAST_RETRAIN_TS.exists():
        try:
            last_ts     = float(LAST_RETRAIN_TS.read_text().strip())
            hours_since = (datetime.now(timezone.utc).timestamp() - last_ts) / 3600
            if hours_since < cooldown_hours:
                print(f"RETRAIN  Cooldown active — last retrain {hours_since:.1f}h ago "
                      f"(min {cooldown_hours}h)")
                return {
                    "status": "skipped",
                    "action": "cooldown",
                    "reason": f"retrained {hours_since:.1f}h ago, cooldown is {cooldown_hours}h",
                }
        except Exception:
            pass   # corrupt timestamp file — allow retrain through

    con = duckdb.connect(str(db_path))
    try:
        training_table = rtcfg.get("training_table", "ethusd_features_1h_v2")
        label_table    = rtcfg.get("label_table",    "ethusd_event_targets_1h")
        label_col      = rtcfg.get("label_col",      "event_label")
        regime_filter  = rtcfg.get("regime_filter",  "compression")
        join_on        = rtcfg.get("join_on",        "open_time")

        df = con.execute(f"""
            SELECT f.*, t.{label_col}
            FROM {training_table} f
            INNER JOIN {label_table} t
                ON CAST(f.{join_on} AS TIMESTAMP) = CAST(t.{join_on} AS TIMESTAMP)
            WHERE t.regime_label = '{regime_filter}'
              AND t.{label_col}  IS NOT NULL
            ORDER BY f.{join_on}
        """).fetchdf()
        con.close()

        if len(df) < min_train_rows:
            return {
                "status": "skipped",
                "action": "insufficient_data",
                "reason": f"only {len(df)} compression rows, need {min_train_rows}",
            }

        # Load exact feature list from feature_cols.json
        feature_cols_path = PROJECT_ROOT / rtcfg.get(
            "feature_cols_path", "models/artifacts/feature_cols.json"
        )
        if feature_cols_path.exists():
            feature_cols = json.loads(feature_cols_path.read_text())["feature_cols"]
        else:
            exclude = {
                "open_time", "close_time", "symbol", "interval",
                "open", "high", "low", "close", "volume",
                "quote_asset_volume", "number_of_trades",
                "taker_buy_base_volume", "taker_buy_quote_volume",
                label_col, "regime_label",
            }
            feature_cols = [c for c in df.columns if c not in exclude
                            and pd.api.types.is_numeric_dtype(df[c])]
            print(f"RETRAIN  WARNING: feature_cols.json not found, derived {len(feature_cols)} cols")

        feature_cols = [c for c in feature_cols if c in df.columns]
        print(f"RETRAIN  Training on {len(df)} rows, {len(feature_cols)} features")

        X = df[feature_cols].values   # numpy — no column names, no sklearn warning
        y = df[label_col].values

        split_idx      = int(len(df) * (1 - oos_test_pct))
        X_train, X_oos = X[:split_idx], X[split_idx:]
        y_train, y_oos = y[:split_idx], y[split_idx:]

        if len(X_oos) < 5:
            return {
                "status": "skipped",
                "action": "insufficient_oos",
                "reason": f"only {len(X_oos)} OOS rows",
            }

        print(f"RETRAIN  Training HistGradientBoostingClassifier on {len(X_train)} rows...")
        new_model = HistGradientBoostingClassifier(
            max_iter=max_iter,
            learning_rate=learning_rate,
            max_leaf_nodes=max_leaf_nodes,
            random_state=42,
        )
        new_model.fit(X_train, y_train)

        # Isotonic calibration on OOS
        raw_probas = new_model.predict_proba(X_oos)[:, 1]
        calibrator = IsotonicRegression(out_of_bounds="clip")
        calibrator.fit(raw_probas, y_oos)
        cal_probas = calibrator.transform(raw_probas)

        # OOS metrics using relative rank (matches live signal logic)
        band_min    = float(cfg["paper_trade_agent"]["band_min"])
        n           = len(cal_probas)
        ranks       = np.argsort(np.argsort(cal_probas)) + 1   # 1..n
        rel_scores  = ranks / n                                  # (0, 1]
        pred_labels = (rel_scores >= band_min).astype(int)

        print(f"RETRAIN  OOS rel_score range: {rel_scores.min():.3f}–{rel_scores.max():.3f} | "
              f"signals (>={band_min}): {pred_labels.sum()}")

        wins         = int(np.sum((pred_labels == 1) & (y_oos == 1)))
        losses       = int(np.sum((pred_labels == 1) & (y_oos == 0)))
        total_trades = wins + losses

        if total_trades == 0:
            return {
                "status": "skipped",
                "action": "no_trades_in_oos",
                "reason": "no signals above band_min in OOS window",
            }

        avg_win      = 0.012
        avg_loss     = 0.015
        gross_profit = wins   * avg_win
        gross_loss   = losses * avg_loss
        new_pf       = gross_profit / gross_loss if gross_loss > 0 else 999.0
        new_winrate  = wins / total_trades
        new_oos_ret  = float(np.mean(cal_probas[pred_labels == 1])) - 0.5

        print(f"RETRAIN  OOS PF: {new_pf:.4f} | Win rate: {new_winrate:.4f} | Trades: {total_trades}")

        if new_pf < promotion_min_pf:
            return {
                "status":       "success",
                "action":       "model_rejected",
                "reason":       f"OOS PF {new_pf:.4f} < promotion threshold {promotion_min_pf}",
                "new_model_pf": round(new_pf, 4),
            }

        current_pf = cfg["drift_checker"]["baseline_profit_factor"]

        model_save_path      = PROJECT_ROOT / rtcfg.get(
            "model_save_path", "models/artifacts/hgb_compression_retrained.joblib")
        calibrator_save_path = PROJECT_ROOT / rtcfg.get(
            "calibrator_save_path", "models/artifacts/compression_calibrator.joblib")
        feature_cols_save    = PROJECT_ROOT / rtcfg.get(
            "feature_cols_path", "models/artifacts/feature_cols.json")

        model_save_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(new_model,  model_save_path)
        joblib.dump(calibrator, calibrator_save_path)
        feature_cols_save.write_text(json.dumps(
            {"feature_cols": feature_cols, "source_table": training_table,
             "label_table": label_table, "label_col": label_col},
            indent=2
        ))

        print(f"RETRAIN  Model promoted → {model_save_path.name}")

        # ── Write cooldown timestamp BEFORE baseline update ───────────────
        LAST_RETRAIN_TS.parent.mkdir(parents=True, exist_ok=True)
        LAST_RETRAIN_TS.write_text(str(datetime.now(timezone.utc).timestamp()))

        baseline_updated = _update_baseline_in_config(new_pf, new_winrate, new_oos_ret)

        try:
            _log_feature_importance(
                db_path=db_path,
                run_id=context.get("run_id", "unknown"),
                model=new_model,
                feature_cols=feature_cols,
                new_pf=new_pf,
                oos_trades=total_trades,
            )
        except Exception as fi_err:
            print(f"RETRAIN  Feature importance logging failed (non-fatal): {fi_err}")

        return {
            "status":            "success",
            "action":            "model_swapped",
            "new_model_pf":      round(new_pf, 4),
            "new_model_winrate": round(new_winrate, 4),
            "old_baseline_pf":   round(current_pf, 4),
            "baseline_updated":  baseline_updated,
            "oos_trades":        total_trades,
            "swapped_at":        datetime.now(timezone.utc).isoformat(),
        }

    except Exception:
        try:
            con.close()
        except Exception:
            pass
        return {
            "status": "failed",
            "error":  traceback.format_exc(),
            "action": "error",
        }