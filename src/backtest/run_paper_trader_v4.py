import duckdb
import yaml
import pandas as pd
import numpy as np
from pathlib import Path


def load_config(path="configs/data_sources.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def apply_costs(entry_price, exit_price, side, fee_bps=4.0, slippage_bps=3.0):
    fee = (fee_bps / 10000.0)
    slip = (slippage_bps / 10000.0)

    if side == "long":
        fill_entry = entry_price * (1 + slip)
        fill_exit = exit_price * (1 - slip)
        gross = (fill_exit - fill_entry) / fill_entry
    else:
        fill_entry = entry_price * (1 - slip)
        fill_exit = exit_price * (1 + slip)
        gross = (fill_entry - fill_exit) / fill_entry

    net = gross - (2 * fee)
    return fill_entry, fill_exit, gross, net


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

    df = df.dropna(subset=["pred_proba", "event_label", "close"]).copy()
    df["pred_label"] = df["pred_label"].astype(int)

    confidence_threshold = 0.60
    fee_bps = 4.0
    slippage_bps = 3.0
    holding_bars = 6

    trades = []
    closes = df["close"].values
    probas = df["pred_proba"].values
    labels = df["pred_label"].values
    regimes = df["regime_label"].values
    times = df["open_time"].values
    actual = df["event_label"].astype(int).values
    n = len(df)

    for i in range(n):
        if probas[i] < confidence_threshold:
            continue

        entry_idx = i
        exit_idx = min(i + holding_bars, n - 1)
        if exit_idx <= entry_idx:
            continue

        side = "long" if labels[i] == 1 else "flat"
        if side == "flat":
            continue

        entry_price = closes[entry_idx]
        exit_price = closes[exit_idx]
        fill_entry, fill_exit, gross_ret, net_ret = apply_costs(
            entry_price, exit_price, side, fee_bps=fee_bps, slippage_bps=slippage_bps
        )

        trades.append({
            "entry_time": times[entry_idx],
            "exit_time": times[exit_idx],
            "entry_price": entry_price,
            "exit_price": exit_price,
            "fill_entry": fill_entry,
            "fill_exit": fill_exit,
            "side": side,
            "regime_label": regimes[i],
            "pred_proba": float(probas[i]),
            "pred_label": int(labels[i]),
            "actual_label": int(actual[i]),
            "gross_return": float(gross_ret),
            "net_return": float(net_ret),
        })

    trades_df = pd.DataFrame(trades)

    if trades_df.empty:
        raise ValueError("No trades were generated. Lower confidence_threshold or review predictions.")

    trades_df["equity_curve"] = (1 + trades_df["net_return"]).cumprod() - 1
    trades_df["running_peak"] = trades_df["equity_curve"].cummax()
    trades_df["drawdown"] = trades_df["equity_curve"] - trades_df["running_peak"]

    summary = {
        "trades": len(trades_df),
        "win_rate": float((trades_df["net_return"] > 0).mean()),
        "avg_net_return": float(trades_df["net_return"].mean()),
        "median_net_return": float(trades_df["net_return"].median()),
        "total_net_return": float((1 + trades_df["net_return"]).prod() - 1),
        "max_drawdown": float(trades_df["drawdown"].min()),
        "avg_pred_proba": float(trades_df["pred_proba"].mean()),
        "fee_bps": fee_bps,
        "slippage_bps": slippage_bps,
        "confidence_threshold": confidence_threshold,
    }

    con = duckdb.connect(db_path)
    con.register("trades_df", trades_df)
    con.register("summary_df", pd.DataFrame([summary]))

    con.execute("""
        CREATE OR REPLACE TABLE ethusd_paper_trades_v4 AS
        SELECT * FROM trades_df
    """)
    con.execute("""
        CREATE OR REPLACE TABLE ethusd_paper_summary_v4 AS
        SELECT * FROM summary_df
    """)
    con.close()

    trades_df.to_csv(out_dir / "ethusd_paper_trades_v4.csv", index=False)
    pd.DataFrame([summary]).to_csv(out_dir / "ethusd_paper_summary_v4.csv", index=False)

    print("Paper trading summary:")
    print(pd.DataFrame([summary]).to_string(index=False))
    print("\nRecent trades:")
    print(trades_df.tail(20).to_string(index=False))


if __name__ == "__main__":
    main()