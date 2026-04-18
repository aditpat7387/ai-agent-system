import duckdb
import yaml
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss


def load_config(path="configs/data_sources.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_volatility_proxy(df, price_col="close", window=20):
    ret_1 = df[price_col].astype(float).pct_change()
    vol = ret_1.rolling(window).std()
    fallback = ret_1.std()
    if pd.isna(fallback) or fallback <= 0:
        fallback = 0.01
    return vol.fillna(fallback)


class IdentityCalibrator:
    def fit(self, p, y):
        return self

    def predict(self, p):
        p = np.asarray(p, dtype=float)
        return np.clip(p, 1e-6, 1 - 1e-6)


class SigmoidCalibrator:
    def __init__(self):
        self.model = LogisticRegression(solver="lbfgs")

    def fit(self, p, y):
        p = np.asarray(p, dtype=float).reshape(-1, 1)
        y = np.asarray(y, dtype=int)
        self.model.fit(p, y)
        return self

    def predict(self, p):
        p = np.asarray(p, dtype=float).reshape(-1, 1)
        out = self.model.predict_proba(p)[:, 1]
        return np.clip(out, 1e-6, 1 - 1e-6)


class IsotonicCalibrator:
    def __init__(self):
        self.model = IsotonicRegression(out_of_bounds="clip")

    def fit(self, p, y):
        p = np.asarray(p, dtype=float)
        y = np.asarray(y, dtype=int)
        self.model.fit(p, y)
        return self

    def predict(self, p):
        p = np.asarray(p, dtype=float)
        out = self.model.predict(p)
        return np.clip(out, 1e-6, 1 - 1e-6)


def time_split_for_calibration(df, calibration_frac=0.20, min_calibration_rows=200):
    n = len(df)
    cal_n = max(int(n * calibration_frac), min_calibration_rows)
    cal_n = min(cal_n, n - 1) if n > 1 else 0
    split_idx = max(n - cal_n, 1)
    train_df = df.iloc[:split_idx].copy()
    cal_df = df.iloc[split_idx:].copy()
    return train_df, cal_df


def fit_probability_calibrator(
    train_df,
    cal_df,
    raw_prob_col="pred_proba",
    label_col="event_label",
    method="sigmoid",
    min_isotonic_rows=1000,
):
    y_cal = cal_df[label_col].astype(int).values
    p_cal_raw = np.clip(cal_df[raw_prob_col].astype(float).values, 1e-6, 1 - 1e-6)

    unique_classes = np.unique(y_cal)
    if len(unique_classes) < 2:
        calibrator = IdentityCalibrator()
        calibrator.fit(p_cal_raw, y_cal)
        meta = {
            "calibration_method_requested": method,
            "calibration_method_used": "identity",
            "calibration_reason": "single_class_in_calibration_slice",
            "calibration_rows": int(len(cal_df)),
        }
        return calibrator, meta

    if method == "isotonic" and len(cal_df) < min_isotonic_rows:
        method_used = "sigmoid"
        reason = "isotonic_requested_but_calibration_slice_too_small"
    else:
        method_used = method
        reason = "ok"

    if method_used == "sigmoid":
        calibrator = SigmoidCalibrator()
    elif method_used == "isotonic":
        calibrator = IsotonicCalibrator()
    else:
        calibrator = IdentityCalibrator()

    calibrator.fit(p_cal_raw, y_cal)

    p_calibrated = calibrator.predict(p_cal_raw)

    meta = {
        "calibration_method_requested": method,
        "calibration_method_used": method_used,
        "calibration_reason": reason,
        "calibration_rows": int(len(cal_df)),
        "raw_brier": float(brier_score_loss(y_cal, p_cal_raw)),
        "cal_brier": float(brier_score_loss(y_cal, p_calibrated)),
        "raw_log_loss": float(log_loss(y_cal, p_cal_raw, labels=[0, 1])),
        "cal_log_loss": float(log_loss(y_cal, p_calibrated, labels=[0, 1])),
        "raw_prob_mean": float(np.mean(p_cal_raw)),
        "cal_prob_mean": float(np.mean(p_calibrated)),
        "positive_rate_cal_slice": float(np.mean(y_cal)),
    }

    return calibrator, meta


def apply_calibration(df, calibrator, raw_prob_col="pred_proba"):
    out = df.copy()
    raw_probs = np.clip(out[raw_prob_col].astype(float).values, 1e-6, 1 - 1e-6)
    out["cal_pred_proba"] = calibrator.predict(raw_probs)
    return out


def simulate_trade(entry_price, future_prices, future_times, side, stop_pct, tp_pct, fee_bps, slippage_bps, max_hold_bars):
    fee = fee_bps / 10000.0
    slip = slippage_bps / 10000.0

    if side == "long":
        stop_price = entry_price * (1 - stop_pct)
        tp_price = entry_price * (1 + tp_pct)
        fill_entry = entry_price * (1 + slip)
    else:
        stop_price = entry_price * (1 + stop_pct)
        tp_price = entry_price * (1 - tp_pct)
        fill_entry = entry_price * (1 - slip)

    chosen_exit_idx = min(max_hold_bars, len(future_prices) - 1)
    exit_price = float(future_prices[chosen_exit_idx])
    exit_time = future_times[chosen_exit_idx]
    exit_reason = "time_exit"

    for j in range(1, min(max_hold_bars, len(future_prices) - 1) + 1):
        px = float(future_prices[j])

        if side == "long":
            if px <= stop_price:
                exit_price = stop_price
                exit_time = future_times[j]
                exit_reason = "stop_loss"
                chosen_exit_idx = j
                break
            if px >= tp_price:
                exit_price = tp_price
                exit_time = future_times[j]
                exit_reason = "take_profit"
                chosen_exit_idx = j
                break
        else:
            if px >= stop_price:
                exit_price = stop_price
                exit_time = future_times[j]
                exit_reason = "stop_loss"
                chosen_exit_idx = j
                break
            if px <= tp_price:
                exit_price = tp_price
                exit_time = future_times[j]
                exit_reason = "take_profit"
                chosen_exit_idx = j
                break

    if side == "long":
        fill_exit = exit_price * (1 - slip)
        gross_ret = (fill_exit - fill_entry) / fill_entry
    else:
        fill_exit = exit_price * (1 + slip)
        gross_ret = (fill_entry - fill_exit) / fill_entry

    net_ret = gross_ret - (2 * fee)

    return {
        "stop_price": float(stop_price),
        "tp_price": float(tp_price),
        "fill_entry": float(fill_entry),
        "fill_exit": float(fill_exit),
        "gross_ret": float(gross_ret),
        "net_ret": float(net_ret),
        "exit_reason": exit_reason,
        "exit_time": exit_time,
        "bars_held": int(chosen_exit_idx),
    }


def run_paper_trader(
    df,
    confidence_threshold=0.60,
    fee_bps=4.0,
    slippage_bps=3.0,
    initial_equity=100000.0,
    risk_pct=0.01,
    stop_vol_mult=2.0,
    rr_multiple=2.0,
    max_hold_bars=12,
    allowed_regimes=None,
):
    if allowed_regimes is None:
        allowed_regimes = {"trend_up", "trend_down"}

    account_equity = initial_equity
    trades = []
    rejected = []

    n = len(df)
    i = 0

    while i < n:
        row = df.iloc[i]

        regime = str(row["regime_label"])
        raw_proba = float(row["pred_proba"])
        cal_proba = float(row["cal_pred_proba"])
        pred = int(row["pred_label"])
        actual = int(row["event_label"])

        if regime not in allowed_regimes:
            rejected.append({
                "open_time": row["open_time"],
                "close": float(row["close"]),
                "regime_label": regime,
                "raw_pred_proba": raw_proba,
                "cal_pred_proba": cal_proba,
                "pred_label": pred,
                "actual_label": actual,
                "reject_reason": "regime_not_allowed",
                "threshold": confidence_threshold,
            })
            i += 1
            continue

        if cal_proba < confidence_threshold:
            rejected.append({
                "open_time": row["open_time"],
                "close": float(row["close"]),
                "regime_label": regime,
                "raw_pred_proba": raw_proba,
                "cal_pred_proba": cal_proba,
                "pred_label": pred,
                "actual_label": actual,
                "reject_reason": "low_calibrated_confidence",
                "threshold": confidence_threshold,
            })
            i += 1
            continue

        side = "long" if regime == "trend_up" else "short"
        if (side == "long" and pred != 1) or (side == "short" and pred != 0):
            rejected.append({
                "open_time": row["open_time"],
                "close": float(row["close"]),
                "regime_label": regime,
                "raw_pred_proba": raw_proba,
                "cal_pred_proba": cal_proba,
                "pred_label": pred,
                "actual_label": actual,
                "reject_reason": "direction_mismatch",
                "threshold": confidence_threshold,
            })
            i += 1
            continue

        entry_price = float(row["close"])
        vol = float(row["vol20"])
        if not np.isfinite(vol) or vol <= 0:
            vol = 0.01

        stop_pct = max(stop_vol_mult * vol, 0.005)
        tp_pct = stop_pct * rr_multiple

        future_end = min(i + max_hold_bars, n - 1)
        future_slice = df.iloc[i:future_end + 1]

        future_prices = future_slice["close"].astype(float).tolist()
        future_times = future_slice["open_time"].tolist()

        if len(future_prices) < 2:
            i += 1
            continue

        risk_dollars = account_equity * risk_pct
        stop_distance_dollars = entry_price * stop_pct
        if stop_distance_dollars <= 0:
            i += 1
            continue

        qty = risk_dollars / stop_distance_dollars

        sim = simulate_trade(
            entry_price=entry_price,
            future_prices=future_prices,
            future_times=future_times,
            side=side,
            stop_pct=stop_pct,
            tp_pct=tp_pct,
            fee_bps=fee_bps,
            slippage_bps=slippage_bps,
            max_hold_bars=max_hold_bars,
        )

        pnl_dollars = qty * entry_price * sim["net_ret"]
        account_equity = account_equity + pnl_dollars

        trades.append({
            "threshold": confidence_threshold,
            "entry_time": row["open_time"],
            "exit_time": sim["exit_time"],
            "entry_price": entry_price,
            "exit_price": float(sim["fill_exit"]),
            "fill_entry": float(sim["fill_entry"]),
            "fill_exit": float(sim["fill_exit"]),
            "side": side,
            "regime_label": regime,
            "raw_pred_proba": raw_proba,
            "cal_pred_proba": cal_proba,
            "pred_label": pred,
            "actual_label": actual,
            "vol20": float(vol),
            "stop_pct": float(stop_pct),
            "take_profit_pct": float(tp_pct),
            "stop_price": float(sim["stop_price"]),
            "take_profit_price": float(sim["tp_price"]),
            "qty": float(qty),
            "bars_held": int(sim["bars_held"]),
            "exit_reason": sim["exit_reason"],
            "gross_return": float(sim["gross_ret"]),
            "net_return": float(sim["net_ret"]),
            "pnl_dollars": float(pnl_dollars),
            "account_equity": float(account_equity),
        })

        i = i + sim["bars_held"] + 1

    trades_df = pd.DataFrame(trades)
    rejected_df = pd.DataFrame(rejected)

    if trades_df.empty:
        summary = {
            "threshold": confidence_threshold,
            "trades": 0,
            "win_rate": 0.0,
            "avg_net_return": 0.0,
            "median_net_return": 0.0,
            "total_net_return": 0.0,
            "max_drawdown": 0.0,
            "avg_raw_pred_proba": 0.0,
            "avg_cal_pred_proba": 0.0,
            "avg_qty": 0.0,
            "avg_bars_held": 0.0,
            "take_profit_count": 0,
            "stop_loss_count": 0,
            "time_exit_count": 0,
            "rejections": int(len(rejected_df)),
            "fee_bps": fee_bps,
            "slippage_bps": slippage_bps,
            "risk_pct": risk_pct,
            "stop_vol_mult": stop_vol_mult,
            "rr_multiple": rr_multiple,
            "allowed_regimes": ",".join(sorted(allowed_regimes)),
        }
        return trades_df, rejected_df, pd.DataFrame([summary])

    trades_df["equity_curve"] = trades_df["account_equity"] / initial_equity - 1.0
    trades_df["running_peak"] = trades_df["equity_curve"].cummax()
    trades_df["drawdown"] = trades_df["equity_curve"] - trades_df["running_peak"]

    summary = {
        "threshold": confidence_threshold,
        "trades": int(len(trades_df)),
        "win_rate": float((trades_df["net_return"] > 0).mean()),
        "avg_net_return": float(trades_df["net_return"].mean()),
        "median_net_return": float(trades_df["net_return"].median()),
        "total_net_return": float(trades_df["account_equity"].iloc[-1] / initial_equity - 1.0),
        "max_drawdown": float(trades_df["drawdown"].min()),
        "avg_raw_pred_proba": float(trades_df["raw_pred_proba"].mean()),
        "avg_cal_pred_proba": float(trades_df["cal_pred_proba"].mean()),
        "avg_qty": float(trades_df["qty"].mean()),
        "avg_bars_held": float(trades_df["bars_held"].mean()),
        "take_profit_count": int((trades_df["exit_reason"] == "take_profit").sum()),
        "stop_loss_count": int((trades_df["exit_reason"] == "stop_loss").sum()),
        "time_exit_count": int((trades_df["exit_reason"] == "time_exit").sum()),
        "rejections": int(len(rejected_df)),
        "fee_bps": fee_bps,
        "slippage_bps": slippage_bps,
        "risk_pct": risk_pct,
        "stop_vol_mult": stop_vol_mult,
        "rr_multiple": rr_multiple,
        "allowed_regimes": ",".join(sorted(allowed_regimes)),
    }

    return trades_df, rejected_df, pd.DataFrame([summary])


def main():
    cfg = load_config()
    db_path = cfg["storage"]["db_path"]
    pred_table = cfg["storage"].get("predictions_table", "ethusd_wfo_predictions_v4")

    out_dir = Path("artifacts/backtests")
    out_dir.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(db_path)
    df = con.execute(f"""
        SELECT *
        FROM {pred_table}
        ORDER BY open_time
    """).fetchdf()
    con.close()

    if df.empty:
        raise ValueError(f"No prediction rows found in {pred_table}")

    required = ["pred_proba", "event_label", "close", "regime_label", "pred_label", "open_time"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.dropna(subset=required).copy()
    df["pred_label"] = df["pred_label"].astype(int)
    df["event_label"] = df["event_label"].astype(int)
    df["vol20"] = build_volatility_proxy(df, price_col="close", window=20)
    df = df.reset_index(drop=True)

    calibration_method = "isotonic"   # options: sigmoid, isotonic, identity
    calibration_frac = 0.20
    min_calibration_rows = 200
    min_isotonic_rows = 200

    baseline_threshold = 0.60
    sweep_thresholds = [0.50, 0.60, 0.70, 0.80, 0.90, 0.95]

    fee_bps = 4.0
    slippage_bps = 3.0
    initial_equity = 100000.0
    risk_pct = 0.01
    stop_vol_mult = 2.0
    rr_multiple = 2.0
    max_hold_bars = 12
    allowed_regimes = {"trend_up", "trend_down"}

    fit_df, cal_df = time_split_for_calibration(
        df,
        calibration_frac=calibration_frac,
        min_calibration_rows=min_calibration_rows,
    )

    calibrator, cal_meta = fit_probability_calibrator(
        train_df=fit_df,
        cal_df=cal_df,
        raw_prob_col="pred_proba",
        label_col="event_label",
        method=calibration_method,
        min_isotonic_rows=min_isotonic_rows,
    )

    df = apply_calibration(df, calibrator, raw_prob_col="pred_proba")

    calibration_summary_df = pd.DataFrame([{
        "calibration_method_requested": cal_meta.get("calibration_method_requested"),
        "calibration_method_used": cal_meta.get("calibration_method_used"),
        "calibration_reason": cal_meta.get("calibration_reason"),
        "calibration_rows": cal_meta.get("calibration_rows"),
        "raw_brier": cal_meta.get("raw_brier", np.nan),
        "cal_brier": cal_meta.get("cal_brier", np.nan),
        "raw_log_loss": cal_meta.get("raw_log_loss", np.nan),
        "cal_log_loss": cal_meta.get("cal_log_loss", np.nan),
        "raw_prob_mean": cal_meta.get("raw_prob_mean", np.nan),
        "cal_prob_mean": cal_meta.get("cal_prob_mean", np.nan),
        "positive_rate_cal_slice": cal_meta.get("positive_rate_cal_slice", np.nan),
        "calibration_frac": calibration_frac,
        "min_calibration_rows": min_calibration_rows,
        "min_isotonic_rows": min_isotonic_rows,
    }])

    trades_df, rejected_df, summary_df = run_paper_trader(
        df=df,
        confidence_threshold=baseline_threshold,
        fee_bps=fee_bps,
        slippage_bps=slippage_bps,
        initial_equity=initial_equity,
        risk_pct=risk_pct,
        stop_vol_mult=stop_vol_mult,
        rr_multiple=rr_multiple,
        max_hold_bars=max_hold_bars,
        allowed_regimes=allowed_regimes,
    )

    summary_df["calibration_method_used"] = cal_meta.get("calibration_method_used")
    summary_df["calibration_rows"] = cal_meta.get("calibration_rows")

    sweep_summaries = []
    for threshold in sweep_thresholds:
        _, _, s = run_paper_trader(
            df=df,
            confidence_threshold=threshold,
            fee_bps=fee_bps,
            slippage_bps=slippage_bps,
            initial_equity=initial_equity,
            risk_pct=risk_pct,
            stop_vol_mult=stop_vol_mult,
            rr_multiple=rr_multiple,
            max_hold_bars=max_hold_bars,
            allowed_regimes=allowed_regimes,
        )
        s["calibration_method_used"] = cal_meta.get("calibration_method_used")
        s["calibration_rows"] = cal_meta.get("calibration_rows")
        sweep_summaries.append(s)

    sweep_df = pd.concat(sweep_summaries, ignore_index=True).sort_values("threshold").reset_index(drop=True)

    con = duckdb.connect(db_path)
    con.register("df_calibrated", df)
    con.register("trades_df", trades_df)
    con.register("rejected_df", rejected_df)
    con.register("summary_df", summary_df)
    con.register("sweep_df", sweep_df)
    con.register("calibration_summary_df", calibration_summary_df)

    con.execute("""
        CREATE OR REPLACE TABLE ethusd_predictions_calibrated_v7 AS
        SELECT * FROM df_calibrated
    """)
    con.execute("""
        CREATE OR REPLACE TABLE ethusd_paper_trades_v7 AS
        SELECT * FROM trades_df
    """)
    con.execute("""
        CREATE OR REPLACE TABLE ethusd_paper_rejections_v7 AS
        SELECT * FROM rejected_df
    """)
    con.execute("""
        CREATE OR REPLACE TABLE ethusd_paper_summary_v7 AS
        SELECT * FROM summary_df
    """)
    con.execute("""
        CREATE OR REPLACE TABLE ethusd_threshold_sweep_v7 AS
        SELECT * FROM sweep_df
    """)
    con.execute("""
        CREATE OR REPLACE TABLE ethusd_calibration_summary_v7 AS
        SELECT * FROM calibration_summary_df
    """)
    con.close()

    df.to_csv(out_dir / "ethusd_predictions_calibrated_v7.csv", index=False)
    trades_df.to_csv(out_dir / "ethusd_paper_trades_v7.csv", index=False)
    rejected_df.to_csv(out_dir / "ethusd_paper_rejections_v7.csv", index=False)
    summary_df.to_csv(out_dir / "ethusd_paper_summary_v7.csv", index=False)
    sweep_df.to_csv(out_dir / "ethusd_threshold_sweep_v7.csv", index=False)
    calibration_summary_df.to_csv(out_dir / "ethusd_calibration_summary_v7.csv", index=False)

    print("Calibration summary v7:")
    print(calibration_summary_df.to_string(index=False))

    print("\nPaper trading summary v7:")
    print(summary_df.to_string(index=False))

    print("\nThreshold sweep v7:")
    print(sweep_df.to_string(index=False))

    print("\nRecent trades:")
    if trades_df.empty:
        print("No trades generated for baseline threshold.")
    else:
        print(trades_df.tail(20).to_string(index=False))

    print("\nRecent rejections:")
    if rejected_df.empty:
        print("No rejections generated for baseline threshold.")
    else:
        print(rejected_df.tail(20).to_string(index=False))


if __name__ == "__main__":
    main()