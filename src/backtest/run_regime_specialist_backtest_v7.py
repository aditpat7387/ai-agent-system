from pathlib import Path
import duckdb
import yaml
import pandas as pd
import numpy as np


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


def build_walk_forward_windows(df, min_train_rows, test_rows, step_rows=None, expanding=True):
    n = len(df)
    if n <= min_train_rows:
        raise ValueError(
            f"Not enough rows for walk-forward. rows={n}, min_train_rows={min_train_rows}"
        )

    if step_rows is None:
        step_rows = test_rows

    windows = []
    train_end = min_train_rows
    window_id = 1

    while train_end + test_rows <= n:
        if expanding:
            train_start = 0
        else:
            train_start = max(0, train_end - min_train_rows)

        test_start = train_end
        test_end = train_end + test_rows

        train_slice = df.iloc[train_start:train_end]
        test_slice = df.iloc[test_start:test_end]

        windows.append({
            "wf_window_id": window_id,
            "train_start_idx": int(train_start),
            "train_end_idx": int(train_end),
            "test_start_idx": int(test_start),
            "test_end_idx": int(test_end),
            "train_rows": int(len(train_slice)),
            "test_rows": int(len(test_slice)),
            "train_start_time": train_slice["open_time"].iloc[0],
            "train_end_time": train_slice["open_time"].iloc[-1],
            "test_start_time": test_slice["open_time"].iloc[0],
            "test_end_time": test_slice["open_time"].iloc[-1],
        })

        train_end += step_rows
        window_id += 1

    if not windows:
        raise ValueError(
            f"Unable to form walk-forward windows with min_train_rows={min_train_rows}, "
            f"test_rows={test_rows}, step_rows={step_rows}, total_rows={n}"
        )

    return windows


def probability_band_pass(pred_proba, band_min=None, band_max=None):
    if pd.isna(pred_proba):
        return False
    if band_min is not None and pred_proba < band_min:
        return False
    if band_max is not None and pred_proba >= band_max:
        return False
    return True


def run_specialist_backtest(
    df,
    threshold=0.18,
    fee_bps=4.0,
    slippage_bps=3.0,
    initial_equity=100000.0,
    risk_pct=0.01,
    stop_vol_mult=2.0,
    rr_multiple=2.0,
    max_hold_bars=12,
    train_rows=None,
    test_rows=None,
    strategy_name="compression_specialist_v7",
    band_name=None,
    band_min=None,
    band_max=None,
):
    account_equity = initial_equity
    trades = []
    rejections = []

    n = len(df)
    i = 0

    effective_rule_name = band_name if band_name is not None else f"ge_{threshold:.2f}"

    while i < n:
        row = df.iloc[i]
        regime = str(row["regime_label"])

        if regime != "compression":
            rejections.append({
                "open_time": row["open_time"],
                "regime_label": regime,
                "close": float(row["close"]),
                "reject_reason": "non_compression_regime",
                "specialist_pred_proba": np.nan,
                "specialist_pred_label": np.nan,
                "actual_label": np.nan,
                "threshold": threshold,
                "band_name": effective_rule_name,
                "band_min": band_min,
                "band_max": band_max,
            })
            i += 1
            continue

        pred_proba = row.get("specialist_pred_proba", np.nan)
        pred_label = row.get("specialist_pred_label", np.nan)
        actual_label = row.get("actual_label", np.nan)

        if pd.isna(pred_proba) or pd.isna(pred_label):
            rejections.append({
                "open_time": row["open_time"],
                "regime_label": regime,
                "close": float(row["close"]),
                "reject_reason": "missing_specialist_prediction",
                "specialist_pred_proba": np.nan,
                "specialist_pred_label": np.nan,
                "actual_label": np.nan if pd.isna(actual_label) else int(actual_label),
                "threshold": threshold,
                "band_name": effective_rule_name,
                "band_min": band_min,
                "band_max": band_max,
            })
            i += 1
            continue

        pred_proba = float(pred_proba)
        pred_label = int(pred_label)
        actual_label = np.nan if pd.isna(actual_label) else int(actual_label)

        band_ok = probability_band_pass(pred_proba, band_min=band_min, band_max=band_max)
        threshold_ok = pred_proba >= threshold if band_min is None and band_max is None else band_ok

        if (not threshold_ok) or pred_label != 1:
            rejections.append({
                "open_time": row["open_time"],
                "regime_label": regime,
                "close": float(row["close"]),
                "reject_reason": "outside_band_or_negative_prediction",
                "specialist_pred_proba": pred_proba,
                "specialist_pred_label": pred_label,
                "actual_label": actual_label,
                "threshold": threshold,
                "band_name": effective_rule_name,
                "band_min": band_min,
                "band_max": band_max,
            })
            i += 1
            continue

        entry_price = float(row["close"])
        vol = float(row["vol20"])
        if not np.isfinite(vol) or vol <= 0:
            vol = 0.01

        side = "long"
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
        account_equity += pnl_dollars

        trades.append({
            "strategy_name": strategy_name,
            "band_name": effective_rule_name,
            "band_min": band_min,
            "band_max": band_max,
            "threshold": threshold,
            "entry_time": row["open_time"],
            "exit_time": sim["exit_time"],
            "regime_label": regime,
            "side": side,
            "entry_price": entry_price,
            "exit_price": float(sim["fill_exit"]),
            "specialist_pred_proba": pred_proba,
            "specialist_pred_label": pred_label,
            "actual_label": actual_label,
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
    rejections_df = pd.DataFrame(rejections)

    if trades_df.empty:
        summary = {
            "strategy_name": strategy_name,
            "band_name": effective_rule_name,
            "band_min": band_min,
            "band_max": band_max,
            "threshold": threshold,
            "train_rows": int(train_rows) if train_rows is not None else 0,
            "test_rows": int(test_rows) if test_rows is not None else 0,
            "trades": 0,
            "win_rate": 0.0,
            "avg_net_return": 0.0,
            "total_net_return": 0.0,
            "max_drawdown": 0.0,
            "final_equity": float(initial_equity),
            "rejections": int(len(rejections_df)),
            "avg_specialist_pred_proba": 0.0,
            "avg_bars_held": 0.0,
            "take_profit_count": 0,
            "stop_loss_count": 0,
            "time_exit_count": 0,
            "profit_factor": 0.0,
            "expectancy_dollars": 0.0,
        }
        return trades_df, rejections_df, pd.DataFrame([summary])

    trades_df["equity_curve"] = trades_df["account_equity"] / initial_equity - 1.0
    trades_df["running_peak"] = trades_df["equity_curve"].cummax()
    trades_df["drawdown"] = trades_df["equity_curve"] - trades_df["running_peak"]

    gross_profit = trades_df.loc[trades_df["pnl_dollars"] > 0, "pnl_dollars"].sum()
    gross_loss = abs(trades_df.loc[trades_df["pnl_dollars"] < 0, "pnl_dollars"].sum())
    profit_factor = 999.0 if gross_loss == 0 and gross_profit > 0 else (gross_profit / gross_loss if gross_loss > 0 else 0.0)

    summary = {
        "strategy_name": strategy_name,
        "band_name": effective_rule_name,
        "band_min": band_min,
        "band_max": band_max,
        "threshold": threshold,
        "train_rows": int(train_rows) if train_rows is not None else 0,
        "test_rows": int(test_rows) if test_rows is not None else 0,
        "trades": int(len(trades_df)),
        "win_rate": float((trades_df["net_return"] > 0).mean()),
        "avg_net_return": float(trades_df["net_return"].mean()),
        "total_net_return": float(trades_df["account_equity"].iloc[-1] / initial_equity - 1.0),
        "max_drawdown": float(trades_df["drawdown"].min()),
        "final_equity": float(trades_df["account_equity"].iloc[-1]),
        "rejections": int(len(rejections_df)),
        "avg_specialist_pred_proba": float(trades_df["specialist_pred_proba"].mean()),
        "avg_bars_held": float(trades_df["bars_held"].mean()),
        "take_profit_count": int((trades_df["exit_reason"] == "take_profit").sum()),
        "stop_loss_count": int((trades_df["exit_reason"] == "stop_loss").sum()),
        "time_exit_count": int((trades_df["exit_reason"] == "time_exit").sum()),
        "profit_factor": float(profit_factor),
        "expectancy_dollars": float(trades_df["pnl_dollars"].mean()),
    }

    return trades_df, rejections_df, pd.DataFrame([summary])


def evaluate_promotion_criteria(summary_row):
    passes_min_trades = int(summary_row["oos_trades"]) >= 8
    passes_active_windows = float(summary_row["active_window_ratio"]) >= 0.50
    passes_positive_oos = float(summary_row["oos_total_net_return"]) > 0
    passes_drawdown = float(summary_row["oos_max_drawdown"]) >= -0.03
    passes_median_window = float(summary_row["median_window_oos_return"]) >= 0

    promotable = all([
        passes_min_trades,
        passes_active_windows,
        passes_positive_oos,
        passes_drawdown,
        passes_median_window,
    ])

    robustness_score = (
        float(summary_row["oos_total_net_return"]) * 1000.0
        + float(summary_row["profit_factor"]) * 10.0
        + float(summary_row["active_window_ratio"]) * 15.0
        + float(summary_row["nonzero_trade_windows"]) * 1.5
        + max(float(summary_row["oos_max_drawdown"]), -0.05) * 200.0
        + float(summary_row["median_window_oos_return"]) * 500.0
    )

    return {
        "passes_min_trades": passes_min_trades,
        "passes_active_windows": passes_active_windows,
        "passes_positive_oos": passes_positive_oos,
        "passes_drawdown": passes_drawdown,
        "passes_median_window": passes_median_window,
        "promotable": promotable,
        "robustness_score": float(robustness_score),
    }


def run_walk_forward_band_evaluation(
    df,
    band_name,
    band_min,
    band_max,
    fee_bps,
    slippage_bps,
    initial_equity,
    risk_pct,
    stop_vol_mult,
    rr_multiple,
    max_hold_bars,
    min_train_rows,
    test_rows,
    step_rows,
    strategy_name="compression_specialist_v7",
):
    windows = build_walk_forward_windows(
        df=df,
        min_train_rows=min_train_rows,
        test_rows=test_rows,
        step_rows=step_rows,
        expanding=True,
    )

    wf_window_summaries = []
    wf_trades = []
    wf_rejections = []
    rolling_equity = initial_equity

    for w in windows:
        train_df = df.iloc[w["train_start_idx"]:w["train_end_idx"]].copy().reset_index(drop=True)
        test_df = df.iloc[w["test_start_idx"]:w["test_end_idx"]].copy().reset_index(drop=True)

        _, _, train_summary_df = run_specialist_backtest(
            df=train_df,
            threshold=band_min if band_min is not None else 0.0,
            fee_bps=fee_bps,
            slippage_bps=slippage_bps,
            initial_equity=initial_equity,
            risk_pct=risk_pct,
            stop_vol_mult=stop_vol_mult,
            rr_multiple=rr_multiple,
            max_hold_bars=max_hold_bars,
            train_rows=len(train_df),
            test_rows=0,
            strategy_name=strategy_name,
            band_name=band_name,
            band_min=band_min,
            band_max=band_max,
        )

        test_trades_df, test_rejections_df, test_summary_df = run_specialist_backtest(
            df=test_df,
            threshold=band_min if band_min is not None else 0.0,
            fee_bps=fee_bps,
            slippage_bps=slippage_bps,
            initial_equity=rolling_equity,
            risk_pct=risk_pct,
            stop_vol_mult=stop_vol_mult,
            rr_multiple=rr_multiple,
            max_hold_bars=max_hold_bars,
            train_rows=len(train_df),
            test_rows=len(test_df),
            strategy_name=strategy_name,
            band_name=band_name,
            band_min=band_min,
            band_max=band_max,
        )

        train_summary = train_summary_df.iloc[0].to_dict()
        test_summary = test_summary_df.iloc[0].to_dict()

        if not test_trades_df.empty:
            test_trades_df = test_trades_df.copy()
            test_trades_df["wf_window_id"] = w["wf_window_id"]
            test_trades_df["wf_train_start_time"] = w["train_start_time"]
            test_trades_df["wf_train_end_time"] = w["train_end_time"]
            test_trades_df["wf_test_start_time"] = w["test_start_time"]
            test_trades_df["wf_test_end_time"] = w["test_end_time"]
            wf_trades.append(test_trades_df)

        if not test_rejections_df.empty:
            test_rejections_df = test_rejections_df.copy()
            test_rejections_df["wf_window_id"] = w["wf_window_id"]
            test_rejections_df["wf_train_start_time"] = w["train_start_time"]
            test_rejections_df["wf_train_end_time"] = w["train_end_time"]
            test_rejections_df["wf_test_start_time"] = w["test_start_time"]
            test_rejections_df["wf_test_end_time"] = w["test_end_time"]
            wf_rejections.append(test_rejections_df)

        ending_equity = float(test_summary["final_equity"])
        wf_window_summaries.append({
            "wf_window_id": w["wf_window_id"],
            "band_name": band_name,
            "band_min": band_min,
            "band_max": band_max,
            "train_start_time": w["train_start_time"],
            "train_end_time": w["train_end_time"],
            "test_start_time": w["test_start_time"],
            "test_end_time": w["test_end_time"],
            "train_rows": w["train_rows"],
            "test_rows": w["test_rows"],
            "starting_equity": float(rolling_equity),
            "ending_equity": ending_equity,
            "train_trades": int(train_summary["trades"]),
            "train_win_rate": float(train_summary["win_rate"]),
            "train_total_net_return": float(train_summary["total_net_return"]),
            "train_max_drawdown": float(train_summary["max_drawdown"]),
            "test_trades": int(test_summary["trades"]),
            "test_win_rate": float(test_summary["win_rate"]),
            "test_avg_net_return": float(test_summary["avg_net_return"]),
            "test_total_net_return": float(test_summary["total_net_return"]),
            "test_max_drawdown": float(test_summary["max_drawdown"]),
            "test_final_equity": ending_equity,
            "test_rejections": int(test_summary["rejections"]),
            "test_avg_specialist_pred_proba": float(test_summary["avg_specialist_pred_proba"]),
            "test_take_profit_count": int(test_summary["take_profit_count"]),
            "test_stop_loss_count": int(test_summary["stop_loss_count"]),
            "test_time_exit_count": int(test_summary["time_exit_count"]),
            "has_test_trade": 1 if int(test_summary["trades"]) > 0 else 0,
        })

        rolling_equity = ending_equity

    wf_windows_df = pd.DataFrame(wf_window_summaries)
    wf_trades_df = pd.concat(wf_trades, ignore_index=True) if wf_trades else pd.DataFrame()
    wf_rejections_df = pd.concat(wf_rejections, ignore_index=True) if wf_rejections else pd.DataFrame()

    if not wf_trades_df.empty:
        wf_trades_df = wf_trades_df.sort_values(["wf_window_id", "entry_time"]).reset_index(drop=True)
        wf_trades_df["wf_equity_curve"] = wf_trades_df["account_equity"] / initial_equity - 1.0
        wf_trades_df["wf_running_peak"] = wf_trades_df["wf_equity_curve"].cummax()
        wf_trades_df["wf_drawdown"] = wf_trades_df["wf_equity_curve"] - wf_trades_df["wf_running_peak"]

        overall_final_equity = float(wf_trades_df["account_equity"].iloc[-1])
        overall_max_drawdown = float(wf_trades_df["wf_drawdown"].min())
        overall_win_rate = float((wf_trades_df["net_return"] > 0).mean())
        overall_avg_net_return = float(wf_trades_df["net_return"].mean())
        overall_avg_pred_proba = float(wf_trades_df["specialist_pred_proba"].mean())
        overall_avg_bars_held = float(wf_trades_df["bars_held"].mean())
        overall_take_profit_count = int((wf_trades_df["exit_reason"] == "take_profit").sum())
        overall_stop_loss_count = int((wf_trades_df["exit_reason"] == "stop_loss").sum())
        overall_time_exit_count = int((wf_trades_df["exit_reason"] == "time_exit").sum())
        gross_profit = wf_trades_df.loc[wf_trades_df["pnl_dollars"] > 0, "pnl_dollars"].sum()
        gross_loss = abs(wf_trades_df.loc[wf_trades_df["pnl_dollars"] < 0, "pnl_dollars"].sum())
        profit_factor = 999.0 if gross_loss == 0 and gross_profit > 0 else (gross_profit / gross_loss if gross_loss > 0 else 0.0)
        expectancy_dollars = float(wf_trades_df["pnl_dollars"].mean())
    else:
        overall_final_equity = float(initial_equity)
        overall_max_drawdown = 0.0
        overall_win_rate = 0.0
        overall_avg_net_return = 0.0
        overall_avg_pred_proba = 0.0
        overall_avg_bars_held = 0.0
        overall_take_profit_count = 0
        overall_stop_loss_count = 0
        overall_time_exit_count = 0
        profit_factor = 0.0
        expectancy_dollars = 0.0

    median_window_oos_return = float(wf_windows_df["test_total_net_return"].median()) if not wf_windows_df.empty else 0.0
    nonzero_trade_windows = int((wf_windows_df["has_test_trade"] == 1).sum()) if not wf_windows_df.empty else 0
    active_window_ratio = float(nonzero_trade_windows / len(wf_windows_df)) if len(wf_windows_df) > 0 else 0.0

    summary_row = {
        "strategy_name": strategy_name,
        "band_name": band_name,
        "band_min": band_min,
        "band_max": band_max,
        "walk_forward_windows": int(len(wf_windows_df)),
        "min_train_rows": int(min_train_rows),
        "test_rows_per_window": int(test_rows),
        "step_rows": int(step_rows),
        "oos_trades": int(len(wf_trades_df)),
        "oos_win_rate": overall_win_rate,
        "oos_avg_net_return": overall_avg_net_return,
        "oos_total_net_return": float(overall_final_equity / initial_equity - 1.0),
        "oos_max_drawdown": overall_max_drawdown,
        "oos_final_equity": overall_final_equity,
        "oos_rejections": int(len(wf_rejections_df)),
        "oos_avg_specialist_pred_proba": overall_avg_pred_proba,
        "oos_avg_bars_held": overall_avg_bars_held,
        "oos_take_profit_count": overall_take_profit_count,
        "oos_stop_loss_count": overall_stop_loss_count,
        "oos_time_exit_count": overall_time_exit_count,
        "profit_factor": float(profit_factor),
        "expectancy_dollars": float(expectancy_dollars),
        "nonzero_trade_windows": nonzero_trade_windows,
        "active_window_ratio": active_window_ratio,
        "median_window_oos_return": median_window_oos_return,
    }

    gates = evaluate_promotion_criteria(summary_row)
    summary_row.update(gates)

    return (
        wf_windows_df,
        wf_trades_df,
        wf_rejections_df,
        pd.DataFrame([summary_row]),
    )


def select_best_band(summary_df):
    promotable_df = summary_df[summary_df["promotable"] == True].copy()

    if not promotable_df.empty:
        promotable_df = promotable_df.sort_values(
            ["robustness_score", "oos_total_net_return", "active_window_ratio", "oos_trades"],
            ascending=[False, False, False, False]
        ).reset_index(drop=True)
        selected = promotable_df.iloc[[0]].copy()
        selected["selection_reason"] = "best_promotable_band"
        return selected

    fallback = summary_df.sort_values(
        ["robustness_score", "oos_total_net_return", "active_window_ratio", "oos_trades"],
        ascending=[False, False, False, False]
    ).head(1).copy()
    fallback["selection_reason"] = "best_non_promotable_band"
    return fallback


def main():
    project_root = Path.cwd()
    print(f"[INFO] Current working directory: {project_root}")
    print(f"[INFO] Project root resolved to: {project_root}")

    output_dir = project_root / "artifacts" / "backtests"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Output directory: {output_dir}")

    config_path = project_root / "configs" / "data_sources.yaml"
    print(f"[INFO] Loading config from: {config_path}")
    cfg = load_config(config_path)

    db_path_cfg = cfg["storage"]["db_path"]
    print(f"[INFO] Config DB path: {db_path_cfg}")
    db_path = (project_root / db_path_cfg).resolve()
    print(f"[INFO] Resolved DB path: {db_path}")

    predictions_table = "ethusd_predictions_calibrated_v7"
    strategy_name = "compression_specialist_v7"

    con = duckdb.connect(str(db_path))

    table_check = con.execute(f"""
        SELECT COUNT(*)
        FROM information_schema.tables
        WHERE table_name = '{predictions_table}'
    """).fetchone()[0]
    print(f"[INFO] Table check - {predictions_table}: {'FOUND' if table_check else 'MISSING'}")

    if table_check == 0:
        raise ValueError(f"Missing required table: {predictions_table}")

    print(f"[INFO] Reading {predictions_table}...")
    df = con.execute(f"""
        SELECT *
        FROM {predictions_table}
        ORDER BY open_time
    """).fetchdf()

    print(f"[INFO] Prediction rows: {len(df)}")

    if df.empty:
        raise ValueError(f"No rows found in {predictions_table}")

    required_cols = ["open_time", "close", "regime_label"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    feature_cols = [c for c in ["close", "fold", "vol20"] if c in df.columns]
    print(f"[INFO] Using {len(feature_cols)} feature columns")
    print(f"[INFO] Feature columns: {feature_cols}")

    df = df.copy()
    df, prob_source, label_source, actual_source = normalize_prediction_columns(df)
    print(f"[INFO] Probability source column: {prob_source}")
    print(f"[INFO] Label source column: {label_source}")
    print(f"[INFO] Actual label source column: {actual_source}")

    if "vol20" not in df.columns or df["vol20"].isna().all():
        df["vol20"] = build_volatility_proxy(df, price_col="close", window=20)
    else:
        fallback_vol = build_volatility_proxy(df, price_col="close", window=20)
        df["vol20"] = df["vol20"].fillna(fallback_vol)

    df["specialist_pred_proba"] = pd.to_numeric(df["specialist_pred_proba"], errors="coerce")
    df["specialist_pred_label"] = pd.to_numeric(df["specialist_pred_label"], errors="coerce")
    df["actual_label"] = pd.to_numeric(df["actual_label"], errors="coerce")
    df["open_time"] = pd.to_datetime(df["open_time"])

    baseline_threshold = 0.78
    fee_bps = 4.0
    slippage_bps = 3.0
    initial_equity = 100000.0
    risk_pct = 0.01
    stop_vol_mult = 2.0
    rr_multiple = 2.0
    max_hold_bars = 12

    min_train_rows = max(500, int(len(df) * 0.40))
    test_rows = max(120, int(len(df) * 0.10))
    step_rows = test_rows

    print(
        f"[INFO] Band walk-forward config: min_train_rows={min_train_rows}, "
        f"test_rows={test_rows}, step_rows={step_rows}"
    )

    band_specs = [
        {"band_name": "ge_0.78", "band_min": 0.78, "band_max": None},
        {"band_name": "0.78_to_0.90", "band_min": 0.78, "band_max": 0.90},
        {"band_name": "0.78_to_0.95", "band_min": 0.78, "band_max": 0.95},
    ]

    baseline_trades_df, baseline_rejections_df, baseline_summary_df = run_specialist_backtest(
        df=df,
        threshold=baseline_threshold,
        fee_bps=fee_bps,
        slippage_bps=slippage_bps,
        initial_equity=initial_equity,
        risk_pct=risk_pct,
        stop_vol_mult=stop_vol_mult,
        rr_multiple=rr_multiple,
        max_hold_bars=max_hold_bars,
        train_rows=0,
        test_rows=len(df),
        strategy_name=strategy_name,
        band_name="ge_0.78",
        band_min=0.78,
        band_max=None,
    )

    all_band_window_dfs = []
    all_band_trade_dfs = []
    all_band_rejection_dfs = []
    all_band_summary_dfs = []

    for spec in band_specs:
        print(f"[INFO] Evaluating band: {spec['band_name']}")

        wf_windows_df, wf_trades_df, wf_rejections_df, wf_summary_df = run_walk_forward_band_evaluation(
            df=df,
            band_name=spec["band_name"],
            band_min=spec["band_min"],
            band_max=spec["band_max"],
            fee_bps=fee_bps,
            slippage_bps=slippage_bps,
            initial_equity=initial_equity,
            risk_pct=risk_pct,
            stop_vol_mult=stop_vol_mult,
            rr_multiple=rr_multiple,
            max_hold_bars=max_hold_bars,
            min_train_rows=min_train_rows,
            test_rows=test_rows,
            step_rows=step_rows,
            strategy_name=strategy_name,
        )

        all_band_window_dfs.append(wf_windows_df)
        all_band_summary_dfs.append(wf_summary_df)

        if not wf_trades_df.empty:
            all_band_trade_dfs.append(wf_trades_df)
        if not wf_rejections_df.empty:
            all_band_rejection_dfs.append(wf_rejections_df)

    band_windows_df = pd.concat(all_band_window_dfs, ignore_index=True) if all_band_window_dfs else pd.DataFrame()
    band_trades_df = pd.concat(all_band_trade_dfs, ignore_index=True) if all_band_trade_dfs else pd.DataFrame()
    band_rejections_df = pd.concat(all_band_rejection_dfs, ignore_index=True) if all_band_rejection_dfs else pd.DataFrame()
    band_summary_df = pd.concat(all_band_summary_dfs, ignore_index=True) if all_band_summary_dfs else pd.DataFrame()

    selected_band_df = select_best_band(band_summary_df)

    con.register("baseline_trades_df", baseline_trades_df)
    con.register("baseline_rejections_df", baseline_rejections_df)
    con.register("baseline_summary_df", baseline_summary_df)
    con.register("band_windows_df", band_windows_df)
    con.register("band_trades_df", band_trades_df)
    con.register("band_rejections_df", band_rejections_df)
    con.register("band_summary_df", band_summary_df)
    con.register("selected_band_df", selected_band_df)

    con.execute("""
        CREATE OR REPLACE TABLE compression_specialist_band_baseline_trades_v7 AS
        SELECT * FROM baseline_trades_df
    """)
    con.execute("""
        CREATE OR REPLACE TABLE compression_specialist_band_baseline_rejections_v7 AS
        SELECT * FROM baseline_rejections_df
    """)
    con.execute("""
        CREATE OR REPLACE TABLE compression_specialist_band_baseline_summary_v7 AS
        SELECT * FROM baseline_summary_df
    """)
    con.execute("""
        CREATE OR REPLACE TABLE compression_specialist_band_walk_forward_windows_v7 AS
        SELECT * FROM band_windows_df
    """)
    con.execute("""
        CREATE OR REPLACE TABLE compression_specialist_band_walk_forward_trades_v7 AS
        SELECT * FROM band_trades_df
    """)
    con.execute("""
        CREATE OR REPLACE TABLE compression_specialist_band_walk_forward_rejections_v7 AS
        SELECT * FROM band_rejections_df
    """)
    con.execute("""
        CREATE OR REPLACE TABLE compression_specialist_band_walk_forward_summary_v7 AS
        SELECT * FROM band_summary_df
    """)
    con.execute("""
        CREATE OR REPLACE TABLE compression_specialist_band_walk_forward_selected_v7 AS
        SELECT * FROM selected_band_df
    """)
    con.close()

    baseline_trades_path = output_dir / "compression_specialist_band_baseline_trades_v7.csv"
    baseline_rejections_path = output_dir / "compression_specialist_band_baseline_rejections_v7.csv"
    baseline_summary_path = output_dir / "compression_specialist_band_baseline_summary_v7.csv"
    band_windows_path = output_dir / "compression_specialist_band_walk_forward_windows_v7.csv"
    band_trades_path = output_dir / "compression_specialist_band_walk_forward_trades_v7.csv"
    band_rejections_path = output_dir / "compression_specialist_band_walk_forward_rejections_v7.csv"
    band_summary_path = output_dir / "compression_specialist_band_walk_forward_summary_v7.csv"
    selected_band_path = output_dir / "compression_specialist_band_walk_forward_selected_v7.csv"

    baseline_trades_df.to_csv(baseline_trades_path, index=False)
    baseline_rejections_df.to_csv(baseline_rejections_path, index=False)
    baseline_summary_df.to_csv(baseline_summary_path, index=False)
    band_windows_df.to_csv(band_windows_path, index=False)
    band_trades_df.to_csv(band_trades_path, index=False)
    band_rejections_df.to_csv(band_rejections_path, index=False)
    band_summary_df.to_csv(band_summary_path, index=False)
    selected_band_df.to_csv(selected_band_path, index=False)

    print(f"[INFO] Saved: {baseline_trades_path}")
    print(f"[INFO] Saved: {baseline_rejections_path}")
    print(f"[INFO] Saved: {baseline_summary_path}")
    print(f"[INFO] Saved: {band_windows_path}")
    print(f"[INFO] Saved: {band_trades_path}")
    print(f"[INFO] Saved: {band_rejections_path}")
    print(f"[INFO] Saved: {band_summary_path}")
    print(f"[INFO] Saved: {selected_band_path}")

    print("\n[INFO] Baseline full-sample summary:")
    print(baseline_summary_df.to_string(index=False))

    print("\n[INFO] Band walk-forward summary:")
    print(band_summary_df.sort_values("robustness_score", ascending=False).to_string(index=False))

    print("\n[INFO] Selected band:")
    print(selected_band_df.to_string(index=False))

    print("\n[INFO] Band walk-forward windows:")
    print(band_windows_df.sort_values(["band_name", "wf_window_id"]).to_string(index=False))


if __name__ == "__main__":
    main()