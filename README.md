# Claudex

> A regime-aware machine learning pipeline for selective trading on ETH/USD 1H data.

![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square)
![DuckDB](https://img.shields.io/badge/DuckDB-1.0%2B-orange?style=flat-square)
![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-lightgrey?style=flat-square)

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Pipeline Stages](#pipeline-stages)
- [Key Scripts](#key-scripts)
- [Artifacts](#artifacts)
- [Data Layer](#data-layer)
- [Tech Stack](#tech-stack)
- [Research Findings](#research-findings)
- [Roadmap](#roadmap)
- [License](#license)

---

## Overview

**Claudex** is a research-to-production machine learning pipeline built from scratch on one year of ETH/USD hourly data (April 2025 – April 2026).

The core hypothesis is that one model does not perform equally well across all market conditions. Claudex addresses this by:

- Classifying market data into distinct **regimes** such as compression and expansion
- Building **specialist models** for the hardest regime, compression
- Validating predictions using **expanding walk-forward out-of-sample evaluation**
- Testing strategies through **realistic paper trading** with fees and slippage
- Surfacing all results through **operational dashboard artifacts**

---

## Project Structure

```text
Claudex/
├── .env
├── data.duckdb
├── requirements.txt
│
├── configs/
│   ├── data_sources.yaml
│   └── paper_trading.yaml
│
├── src/
│   ├── data/
│   │   ├── binance_client.py
│   │   ├── build_canonical_market_table.py
│   │   ├── build_market_quality_report.py
│   │   ├── check_expected_hourly_coverage.py
│   │   ├── duckdb_loader.py
│   │   ├── ingest_ethusd.py
│   │   ├── inspect_canonical_market.py
│   │   └── validate_market_data.py
│   │
│   ├── features/
│   │   ├── build_event_targets.py
│   │   ├── build_feature_store.py
│   │   ├── build_feature_store_v2.py
│   │   ├── build_regime_table.py
│   │   ├── build_selective_training_table.py
│   │   ├── build_selective_training_table_v2.py
│   │   ├── build_training_view.py
│   │   ├── build_training_view_v2.py
│   │   ├── validate_feature_store.py
│   │   └── validate_feature_store_v2.py
│   │
│   ├── models/
│   │   ├── compare_baselines_v1_v2.py
│   │   ├── evaluate_baseline_models.py
│   │   ├── train_baseline_models.py
│   │   ├── train_baseline_models_v2.py
│   │   └── train_selective_model_wfo.py
│   │
│   ├── training/
│   │   ├── train_regime_specialist_models_v7.py
│   │   └── tune_compression_threshold_v7.py
│   │
│   ├── analysis/
│   │   ├── analysis_regime_band_script.py
│   │   ├── analysis_script.py
│   │   ├── analyze_all_predictions_v7.py
│   │   ├── analyze_compression_vs_rest_v7.py
│   │   ├── analyze_regime_feature_balance_v7.py
│   │   └── initial_sanity_check.py
│   │
│   ├── backtest/
│   │   ├── run_compression_specialist_backtest_v7.py
│   │   ├── run_regime_specialist_backtest_v7.py
│   │   ├── run_paper_trader_v4.py
│   │   ├── run_paper_trader_v5.py
│   │   ├── run_paper_trader_v6.py
│   │   └── run_paper_trader_v7.py
│   │
│   ├── dashboard/
│   │   ├── app.py
│   │   └── generate_model_dashboard_v1.py
│   │
│   ├── paper_trading/
│   │   └── run_paper_trading_shadow_v1.py
│   │
│   ├── trading/
│   │   ├── paper_trade_hgb_v2.py
│   │   └── paper_trade_hgb_v2_strict.py
│   │
│   ├── strategy/
│   ├── agents/
│   └── utils/
│       └── healthcheck.py
│
├── models/
│   ├── artifacts/
│   │   ├── hist_gradient_boosting.joblib
│   │   ├── hist_gradient_boosting_binary.joblib
│   │   ├── hist_gradient_boosting_multiclass.joblib
│   │   ├── logistic_regression.joblib
│   │   ├── logistic_regression_binary.joblib
│   │   └── logistic_regression_multiclass.joblib
│   ├── registry/
│   └── reports/
│       ├── baseline_comparison_v1_v2.csv
│       ├── baseline_model_summary.json
│       ├── baseline_model_v2_summary.json
│       ├── hist_gradient_boosting_walk_forward_metrics.csv
│       ├── hist_gradient_boosting_binary_walk_forward_metrics.csv
│       ├── hist_gradient_boosting_multiclass_walk_forward_metrics.csv
│       ├── logistic_regression_walk_forward_metrics.csv
│       ├── logistic_regression_binary_walk_forward_metrics.csv
│       ├── logistic_regression_multiclass_walk_forward_metrics.csv
│       ├── paper_trade_equity_curve.csv
│       ├── paper_trade_summary.csv
│       ├── paper_trade_trade_log.csv
│       ├── paper_trade_v2_equity_curve.csv
│       ├── paper_trade_v2_summary.csv
│       └── paper_trade_v2_trade_log.csv
│
├── artifacts/
│   ├── backtests/
│   │   ├── regime_specialist_metrics_v7.csv
│   │   └── regime_specialist_predictions_v7.csv
│   ├── models/
│   │   ├── ethusd_selective_model_v4.joblib
│   │   └── ethusd_selective_model_wfo.joblib
│   └── paper_trading/
│       ├── compression_specialist_paper_metrics_v1.csv
│       ├── compression_specialist_paper_runtime_v1.csv
│       ├── compression_specialist_paper_signals_v1.csv
│       └── compression_specialist_paper_trades_v1.csv
│
├── data/
│   ├── market.duckdb
│   ├── features/
│   ├── predictions/
│   │   ├── hist_gradient_boosting_binary_walk_forward_predictions.csv
│   │   ├── hist_gradient_boosting_multiclass_walk_forward_predictions.csv
│   │   ├── hist_gradient_boosting_walk_forward_predictions.csv
│   │   ├── logistic_regression_binary_walk_forward_predictions.csv
│   │   ├── logistic_regression_multiclass_walk_forward_predictions.csv
│   │   └── logistic_regression_walk_forward_predictions.csv
│   ├── processed/
│   └── raw/
│       └── binance/
│           └── ethusdt_1h_2025-04-14_2026-04-14.parquet
│
├── logs/
└── notebooks/
```

---

## Getting Started

### Prerequisites

- Python 3.11+
- Windows 10 or 11 with PowerShell
- Binance API key for live data ingestion

### Installation

```powershell
git clone https://github.com/your-username/claudex.git
cd claudex
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

After copying `.env`, open it and fill in your Binance API credentials and DuckDB paths.

### Configuration

Open `configs/data_sources.yaml` to set database paths and table names.

Open `configs/paper_trading.yaml` to configure fees, slippage, position sizing, and decision thresholds.

---

## Pipeline Stages

### Stage 1 — Data Ingestion

```powershell
python src/data/ingest_ethusd.py
python src/data/build_canonical_market_table.py
python src/data/validate_market_data.py
```

Fetches one year of ETH/USD 1H OHLCV data from Binance and stores it as Parquet and in DuckDB.

**Output:** `data/raw/binance/ethusdt_1h_2025-04-14_2026-04-14.parquet`

---

### Stage 2 — Feature Engineering

```powershell
python src/features/build_feature_store_v2.py
python src/features/build_regime_table.py
python src/features/build_event_targets.py
python src/features/validate_feature_store_v2.py
```

Builds 50+ technical indicators, assigns regime labels, and creates forward-looking event targets.

**Output:** Feature store and regime tables written to `data.duckdb`

---

### Stage 3 — Baseline Model Training

```powershell
python src/models/train_baseline_models_v2.py
python src/models/evaluate_baseline_models.py
python src/models/compare_baselines_v1_v2.py
```

Trains HistGradientBoosting and Logistic Regression baselines using walk-forward cross-validation across binary and multiclass formulations.

**Output:** `models/artifacts/*.joblib`, `models/reports/*_walk_forward_metrics.csv`

---

### Stage 4 — Regime Diagnostics

```powershell
python src/analysis/analyze_all_predictions_v7.py
python src/analysis/analyze_compression_vs_rest_v7.py
python src/analysis/analyze_regime_feature_balance_v7.py
```

Analyzes model prediction behavior by regime. Identifies compression as the hardest regime and justifies a specialist approach.

**Output:** Regime summary CSVs and diagnostic JPG charts

---

### Stage 5 — Specialist Training and Threshold Tuning

```powershell
python src/training/train_regime_specialist_models_v7.py
python src/training/tune_compression_threshold_v7.py
```

Trains compression-specific models with isotonic calibration and sweeps decision thresholds from 0.75 to 0.90.

**Output:** Calibrated predictions and threshold sweep tables written to `data.duckdb`

---

### Stage 6 — Backtesting

```powershell
python src/backtest/run_compression_specialist_backtest_v7.py
python src/backtest/run_regime_specialist_backtest_v7.py
```

Runs event-driven backtests with volatility-adjusted stop-loss, take-profit, fee and slippage deduction, and expanding walk-forward splits.

**Output:** `artifacts/backtests/regime_specialist_metrics_v7.csv`

---

### Stage 7 — Paper Trading

```powershell
python src/backtest/run_paper_trader_v7.py
```

Simulates live trading behavior including position sizing, sequential trade execution, and full equity tracking.

**Output:** `artifacts/paper_trading/compression_specialist_paper_trades_v1.csv`

---

### Stage 8 — Dashboard Reporting

```powershell
python src/dashboard/generate_model_dashboard_v1.py
```

Generates an HTML insights dashboard with KPI cards, equity curve, trade diagnostics, and threshold analysis.

**Output:** HTML dashboard file ready for browser viewing

---

## Key Scripts

| Script | Purpose |
|--------|---------|
| `src/data/ingest_ethusd.py` | Binance data ingestion |
| `src/features/build_feature_store_v2.py` | Feature engineering |
| `src/features/build_regime_table.py` | Regime label generation |
| `src/models/train_baseline_models_v2.py` | Baseline model training |
| `src/training/train_regime_specialist_models_v7.py` | Compression specialist training |
| `src/training/tune_compression_threshold_v7.py` | Threshold sweep and calibration |
| `src/analysis/analyze_compression_vs_rest_v7.py` | Compression regime research |
| `src/backtest/run_compression_specialist_backtest_v7.py` | Specialist backtest |
| `src/backtest/run_regime_specialist_backtest_v7.py` | Full regime backtest |
| `src/backtest/run_paper_trader_v7.py` | Paper trading simulation |
| `src/dashboard/generate_model_dashboard_v1.py` | HTML dashboard generation |

---

## Artifacts

### backtests

| File | Description |
|------|-------------|
| `regime_specialist_metrics_v7.csv` | Per-regime strategy performance metrics |
| `regime_specialist_predictions_v7.csv` | Raw model prediction outputs |

### paper_trading

| File | Description |
|------|-------------|
| `compression_specialist_paper_trades_v1.csv` | Executed paper trade log |
| `compression_specialist_paper_metrics_v1.csv` | Performance summary |
| `compression_specialist_paper_signals_v1.csv` | Signal generation log |
| `compression_specialist_paper_runtime_v1.csv` | Runtime diagnostics |

### models/reports

| File | Description |
|------|-------------|
| `*_walk_forward_metrics.csv` | OOS walk-forward results per model variant |
| `paper_trade_equity_curve.csv` | Equity curve from paper trading |
| `paper_trade_v2_equity_curve.csv` | Equity curve from paper trading v2 |
| `baseline_comparison_v1_v2.csv` | Side-by-side baseline comparison |

---

## Data Layer

### data.duckdb — main analytical store

| Table | Description |
|-------|-------------|
| `ethusd_1h_market` | Canonical OHLCV |
| `feature_store_v2` | 50+ technical indicators |
| `regime_table` | Compression and expansion labels |
| `ethusd_predictions_calibrated_v7` | Calibrated model outputs |
| `compression_specialist_trades_v7` | Backtest trade results |
| `compression_specialist_summary_v7` | Backtest summary |
| `compression_specialist_threshold_sweep_v7` | Threshold sensitivity results |

### data/market.duckdb — market data store

| Table | Description |
|-------|-------------|
| `canonical_market` | Validated hourly OHLCV |
| `market_quality_report` | Coverage and quality flags |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.11 |
| Database | DuckDB |
| ML Models | HistGradientBoosting, LogisticRegression |
| Data Source | Binance REST API |
| Storage | Parquet, DuckDB, Joblib |
| Config | YAML |
| Reporting | HTML, CSV |
| Environment | Windows 11, PowerShell, venv |

---

## Research Findings

### Compression is the hardest regime

The compression regime contains 184 rows with a 2:1 negative-to-positive class imbalance (124 negative, 60 positive). Feature distributions in compression shift materially compared to the rest of the data, justifying specialist treatment.

### Calibration behavior

Raw model probabilities cluster near 1.0 for most predictions. Isotonic calibration improves interpretability and enables reliable threshold-based filtering for trade selection.

### Walk-forward results

Expanding walk-forward windows across HistGradientBoosting and Logistic Regression baselines confirm that OOS performance is directionally consistent. Full per-window breakdowns are in `models/reports/`.

### Compression specialist backtest

At threshold 0.18 the specialist generates 18 trades with a 50% win rate, -0.69% total return, and -4.97% maximum drawdown. Threshold sweep from 0.75 to 0.90 shows stable trade count behavior.

---

## Roadmap

- [x] Data ingestion and validation
- [x] Feature store and regime table
- [x] Baseline model training with walk-forward
- [x] Regime diagnostics and compression-vs-rest research
- [x] Compression specialist training and calibration
- [x] Threshold sweep and sensitivity analysis
- [x] Compression specialist backtest
- [x] Paper trading simulation
- [x] HTML dashboard reporting
- [ ] Walk-forward compression specialist robustness testing
- [ ] Windows Task Scheduler automated pipeline
- [ ] Live dashboard refresh after each scheduled run
- [ ] Multi-asset expansion to BTC and SOL
- [ ] Model drift detection and alerting

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

> Built incrementally from scratch. Every artifact earned, not scaffolded.