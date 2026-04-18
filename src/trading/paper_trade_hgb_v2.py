from pathlib import Path
import numpy as np
import pandas as pd


PRED_PATH = Path("data/predictions/hist_gradient_boosting_binary_walk_forward_predictions.csv")
OUT_DIR = Path("models/reports")
OUT_DIR.mkdir(parents=True, exist_ok=True)

INITIAL_CAPITAL = 10_000.0
ENTRY_THRESHOLD = 0.55
EXIT_THRESHOLD = 0.50
FEE_RATE = 0.0004       # 0.04% per side
SLIPPAGE_RATE = 0.0005  # 0.05% per side


def load_predictions():
    df = pd.read_csv(PRED_PATH, parse_dates=["open_time"])
    df = df.sort_values("open_time").reset_index(drop=True)
    return df


def build_signals(df):
    signal = []
    current_position = 0

    for prob in df["y_prob"]:
        if current_position == 0:
            if prob >= ENTRY_THRESHOLD:
                current_position = 1
        else:
            if prob < EXIT_THRESHOLD:
                current_position = 0
        signal.append(current_position)

    df["signal"] = signal
    df["signal_exec"] = df["signal"].shift(1).fillna(0).astype(int)
    return df


def simulate(df):
    capital = INITIAL_CAPITAL
    equity = []
    cash = INITIAL_CAPITAL
    units = 0.0
    in_position = 0
    entry_price = None
    entry_time = None

    trades = []

    for i, row in df.iterrows():
        close_price = float(row["close"])
        signal_exec = int(row["signal_exec"])

        if in_position == 0 and signal_exec == 1:
            fill_price = close_price * (1 + SLIPPAGE_RATE)
            fee = cash * FEE_RATE
            deployable_cash = cash - fee
            units = deployable_cash / fill_price
            cash = 0.0
            in_position = 1
            entry_price = fill_price
            entry_time = row["open_time"]

            trades.append({
                "entry_time": entry_time,
                "entry_price": entry_price,
                "entry_prob": row["y_prob"],
                "exit_time": None,
                "exit_price": None,
                "exit_prob": None,
                "gross_return": None,
                "net_return": None,
                "fees_paid": fee,
                "slippage_paid": close_price * SLIPPAGE_RATE,
                "bars_held": None,
            })

        elif in_position == 1 and signal_exec == 0:
            fill_price = close_price * (1 - SLIPPAGE_RATE)
            gross_value = units * fill_price
            fee = gross_value * FEE_RATE
            cash = gross_value - fee

            gross_return = (fill_price / entry_price) - 1.0
            net_return = (cash / INITIAL_CAPITAL) - 1.0 if INITIAL_CAPITAL > 0 else 0.0

            last_trade = trades[-1]
            last_trade["exit_time"] = row["open_time"]
            last_trade["exit_price"] = fill_price
            last_trade["exit_prob"] = row["y_prob"]
            last_trade["gross_return"] = gross_return
            last_trade["net_return"] = (fill_price / entry_price) - 1.0 - 2 * FEE_RATE - 2 * SLIPPAGE_RATE
            last_trade["fees_paid"] = last_trade["fees_paid"] + fee
            last_trade["slippage_paid"] = last_trade["slippage_paid"] + close_price * SLIPPAGE_RATE
            last_trade["bars_held"] = int(
                (pd.Timestamp(row["open_time"]) - pd.Timestamp(entry_time)).total_seconds() / 3600
            )

            units = 0.0
            in_position = 0
            entry_price = None
            entry_time = None

        current_equity = cash if in_position == 0 else units * close_price
        equity.append(current_equity)

    df["equity"] = equity
    df["strategy_return"] = df["equity"].pct_change().fillna(0.0)
    df["buy_hold_equity"] = INITIAL_CAPITAL * (df["close"] / df["close"].iloc[0])
    df["rolling_peak"] = df["equity"].cummax()
    df["drawdown"] = (df["equity"] / df["rolling_peak"]) - 1.0

    trades_df = pd.DataFrame(trades)

    if not trades_df.empty:
        trades_df["is_win"] = trades_df["net_return"] > 0

    return df, trades_df


def compute_summary(df, trades_df):
    total_return = (df["equity"].iloc[-1] / INITIAL_CAPITAL) - 1.0
    buy_hold_return = (df["buy_hold_equity"].iloc[-1] / INITIAL_CAPITAL) - 1.0
    max_drawdown = df["drawdown"].min()

    summary = {
        "initial_capital": INITIAL_CAPITAL,
        "final_equity": float(df["equity"].iloc[-1]),
        "strategy_total_return": float(total_return),
        "buy_hold_return": float(buy_hold_return),
        "excess_return_vs_buy_hold": float(total_return - buy_hold_return),
        "max_drawdown": float(max_drawdown),
        "bars": int(len(df)),
        "time_in_market_pct": float(df["signal_exec"].mean()),
    }

    if not trades_df.empty:
        summary.update({
            "trade_count": int(len(trades_df)),
            "win_rate": float(trades_df["is_win"].mean()),
            "avg_trade_return": float(trades_df["net_return"].mean()),
            "median_trade_return": float(trades_df["net_return"].median()),
            "avg_bars_held": float(trades_df["bars_held"].mean()),
            "total_fees_paid": float(trades_df["fees_paid"].sum()),
            "total_slippage_proxy": float(trades_df["slippage_paid"].sum()),
        })
    else:
        summary.update({
            "trade_count": 0,
            "win_rate": None,
            "avg_trade_return": None,
            "median_trade_return": None,
            "avg_bars_held": None,
            "total_fees_paid": 0.0,
            "total_slippage_proxy": 0.0,
        })

    return pd.DataFrame([summary])


def main():
    df = load_predictions()
    df = build_signals(df)
    equity_df, trades_df = simulate(df)
    summary_df = compute_summary(equity_df, trades_df)

    equity_df.to_csv(OUT_DIR / "paper_trade_equity_curve.csv", index=False)
    trades_df.to_csv(OUT_DIR / "paper_trade_trade_log.csv", index=False)
    summary_df.to_csv(OUT_DIR / "paper_trade_summary.csv", index=False)

    print("\nPaper trading summary:")
    print(summary_df.to_string(index=False))

    print("\nRecent trades:")
    if not trades_df.empty:
        print(trades_df.tail(10).to_string(index=False))
    else:
        print("No trades generated. Try lowering ENTRY_THRESHOLD.")
        

if __name__ == "__main__":
    main()