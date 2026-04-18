# Regime-Aware Trading Model Pipeline

A research-to-production style machine learning pipeline for **regime classification, probability diagnostics, calibration analysis, specialist modeling, and trading-oriented backtesting** on time-series market data.

This repository documents the work built from scratch across the project lifecycle: data storage, analysis scripts, model diagnostics, compression-vs-rest research, specialist strategy backtests, calibration studies, and dashboard-style reporting artifacts.

***

## What this project does

This project explores a **regime-aware modeling approach** rather than treating all market states as one uniform prediction problem.

At a high level, the workflow is:

1. Store and query model-ready datasets in **DuckDB**.
2. Analyze regime structure and class balance across market states.
3. Evaluate raw and calibrated probability behavior.
4. Isolate the **compression regime** as a harder sub-problem.
5. Build and test a **compression specialist** strategy.
6. Export interpretable CSV and image artifacts.
7. Surface current model insights through an HTML dashboard/reporting layer.

***

## Core ideas behind the work

### Regime-first modeling
Instead of assuming one model behaves equally well everywhere, the project examines model behavior by market regime and treats regime segmentation as a first-class analytical dimension.

### Compression specialist focus
A major thread of the work is understanding whether the **compression regime** should be handled differently from the rest of the market. This led to dedicated diagnostics, class balance checks, label studies, and a specialist backtest flow.

### Calibration and decision quality
The work goes beyond raw probabilities by studying calibration behavior, bucket performance, confidence concentration, and how probabilities translate into realized trade outcomes.

### Operational artifacts
The project does not stop at notebooks or ad hoc prints. Outputs are persisted as **CSV summaries, image diagnostics, and dashboard files** so the pipeline can be reviewed, monitored, and operationalized.

***

## Repository structure

```text
.
├── data.duckdb
├── src/
│   └── analysis/
│       ├── analysis_regime_band_script.py
│       └── analyze_compression_vs_rest_v7.py
├── output/
│   ├── generate_model_dashboard.py
│   └── model-insights/
│       ├── build_model_insights_dashboard_v2.py
│       ├── model-insights-dashboard.html
│       └── model-insights-dashboard-v2.html
├── skills/
│   └── website-building/
│       ├── informational.md
│       ├── webapp.md
│       └── game.md
├── calibration_distribution_data_v7.csv
├── confidence_bucket_performance_v7.csv
├── distribution_comparison_metrics_v7.csv
├── regime_*.csv
├── compression_vs_rest_*.csv
├── compression_specialist_*.csv
└── *.jpg
```

> Note: some project configuration and source files referenced during development may live outside the currently exported artifact set. This README covers both the visible repository files and the delivered analytical outputs produced during the project.

***

## Key components

### Data layer
- **DuckDB database** (`data.duckdb`) used as the central local analytical store.
- Supports repeatable querying of model outputs, regimes, labels, and derived evaluation tables.

### Analysis scripts
- `src/analysis/analysis_regime_band_script.py`  
  Used for regime band and probability-oriented analysis.
- `src/analysis/analyze_compression_vs_rest_v7.py`  
  Focused analysis of compression regime versus all other regimes.

### Dashboard/report generation
- `output/generate_model_dashboard.py`  
  Supporting script for building dashboard/report views.
- `output/model-insights/build_model_insights_dashboard_v2.py`  
  Script for generating a more polished model insights HTML dashboard.
- `output/model-insights/model-insights-dashboard.html`  
  Initial dashboard artifact.
- `output/model-insights/model-insights-dashboard-v2.html`  
  Improved presentation layer for current model insights.

***

## Analytical tracks completed so far

### 1. Probability distribution analysis
Artifacts in this track help answer whether model probabilities are informative, concentrated, or degenerate.

Relevant outputs:
- `calibration_distribution_data_v7.csv`
- `probability_distribution_analysis_v7.jpg`
- `probability_distribution_fixed_bucket_analysis_v7.jpg`
- `distribution_comparison_metrics_v7.csv`
- `confidence_bucket_performance_v7.csv`
- `raw_probability_bucket_performance_v7.csv`
- `calibrated_probability_bucket_performance_v7.csv`

What these are used for:
- Compare raw vs calibrated probabilities.
- Inspect bucket-level performance.
- Understand confidence concentration near extremes.
- Evaluate whether calibration improves interpretability or execution usefulness.

***

### 2. Regime diagnostics
This track evaluates how predictions and outcomes vary by regime, rather than only at the global level.

Relevant outputs:
- `all_prediction_diagnostics_v7.jpg`
- `overall_actual_counts_v7.csv`
- `overall_pred_counts_v7.csv`
- `regime_actual_summary_v7.csv`
- `regime_pred_summary_v7.csv`
- `regime_confusion_summary_v7.csv`
- `regime_raw_probability_bucket_summary_v7.csv`
- `regime_cal_probability_bucket_summary_v7.csv`
- `regime_side_probability_bucket_summary_v7.csv`
- `regime_side_trade_counts_v7.csv`
- `regime_side_probability_band_analysis_v7.jpg`

What these are used for:
- Compare actual vs predicted behavior by regime.
- Inspect class counts and confusion behavior.
- Understand regime-specific confidence patterns.
- Evaluate which regimes are easier or harder to model.

***

### 3. Class balance and label diagnostics
This work checks whether imbalance or label scarcity may be driving poor downstream behavior in some regimes.

Relevant outputs:
- `regime_class_counts_v7.csv`
- `regime_event_counts_v7.csv`
- `regime_feature_balance_v7.csv`
- `regime_class_balance_diagnostics_v7.jpg`

What these are used for:
- Measure positive/negative label distribution by regime.
- Identify sparse or unstable regime slices.
- Inform whether specialist models need reweighting, resampling, or different thresholds.

***

### 4. Compression vs rest research
This is one of the most important research branches in the repository. It isolates the compression regime as a dedicated modeling and trading question.

Relevant outputs:
- `compression_vs_rest_summary_v7.csv`
- `compression_vs_rest_feature_balance_v7.csv`
- `compression_vs_rest_label_summary_v7.csv`
- `compression_vs_rest_diagnostics_v7.jpg`

What this track is used for:
- Quantify whether compression differs materially from the rest of the data.
- Check whether feature distributions shift in compression.
- Inspect label behavior specific to compression.
- Justify a specialist approach instead of a one-size-fits-all model.

***

### 5. Compression specialist backtest
After identifying compression as a special case, the project moved into strategy-style testing of a dedicated compression specialist.

Relevant outputs:
- `compression_specialist_summary_v7.csv`
- `compression_specialist_trades_v7.csv`
- `compression_specialist_rejections_v7.csv`
- `compression_specialist_threshold_sweep_v7.csv`

What this track is used for:
- Evaluate trade selection quality under a compression-specific decision rule.
- Measure realized performance, win rate, drawdown, and trade behavior.
- Inspect rejected opportunities and threshold sensitivity.
- Support walk-forward and robustness extensions.

***

## Notable workflow evolution

This project has evolved through multiple practical stages:

- **Initial analytical setup** for regime-aware inspection.
- **Probability and calibration diagnostics** to understand output quality.
- **Regime-level performance slicing** to avoid misleading aggregate metrics.
- **Compression-vs-rest framing** to isolate the hardest regime.
- **Specialist backtesting** to move from diagnosis into actionability.
- **Threshold sweep analysis** to inspect decision sensitivity.
- **Dashboard/report generation** to communicate current state clearly.

This progression matters because the project was not built as a single one-off script. It was built incrementally as a research system that became increasingly operational.

***

## Typical end-to-end workflow

### Step 1: Query data and model outputs
Use DuckDB-backed tables to load predictions, labels, and regime context.

### Step 2: Run diagnostics
Generate summaries and visual artifacts covering:
- overall prediction behavior,
- regime-level behavior,
- class balance,
- calibration and bucket performance.

### Step 3: Investigate compression separately
Run the compression-vs-rest analysis to confirm whether the regime deserves dedicated handling.

### Step 4: Backtest the compression specialist
Evaluate trade outcomes, rejections, and threshold sensitivity from the specialist setup.

### Step 5: Publish insights
Generate dashboard-style HTML views so the latest model state can be inspected without digging through raw files.

***

## Artifact guide

### CSV outputs
CSV files are the main machine-readable outputs and are useful for:
- additional SQL or pandas analysis,
- dashboard feeding,
- auditability,
- sharing with teammates,
- threshold and trade review.

### Image outputs
JPG artifacts provide visual diagnostics for:
- probability concentration,
- calibration distribution,
- regime class balance,
- compression-vs-rest comparisons,
- regime-side behavior.

### HTML outputs
The dashboard artifacts are intended for human-friendly monitoring and storytelling:
- `model-insights-dashboard.html`
- `model-insights-dashboard-v2.html`

***

## Why this repository is useful

This repository is useful if you want to:
- build a **regime-aware ML evaluation framework**,
- move from raw model scores to **decision-aware analysis**,
- compare **calibrated vs raw probabilities**,
- isolate hard market conditions like **compression**,
- produce **artifact-driven diagnostics** instead of ad hoc notebook outputs,
- create a bridge from research outputs into an operational dashboard.

***

## Suggested future extensions

Good next steps for the project include:

- Add a fully automated **walk-forward evaluation** artifact set for the compression specialist.
- Promote the dashboard into a **scheduled reporting layer** that refreshes after each pipeline run.
- Add explicit **model registry/version metadata** for each artifact family.
- Add **config-driven orchestration** for thresholds, fees, slippage, and holding periods.
- Add **README-linked data dictionary** for all exported CSV fields.
- Split `src/` into clearer modules such as `training/`, `analysis/`, `backtest/`, and `reporting/` if the codebase continues to grow.

***

## Working principles used in this project

- Prefer **diagnostics before optimization**.
- Prefer **regime-aware slicing before aggregate conclusions**.
- Prefer **persisted artifacts** over ephemeral notebook output.
- Prefer **interpretable summaries** over opaque metric dumps.
- Prefer **specialist handling** when one regime behaves materially differently from the rest.

***

## Quick file highlights

If you only want the most important outputs, start here:

- `src/analysis/analyze_compression_vs_rest_v7.py`
- `compression_vs_rest_summary_v7.csv`
- `regime_confusion_summary_v7.csv`
- `distribution_comparison_metrics_v7.csv`
- `compression_specialist_summary_v7.csv`
- `compression_specialist_threshold_sweep_v7.csv`
- `output/model-insights/model-insights-dashboard-v2.html`

***

## Status

The repository currently represents a **working analytical foundation plus an insights presentation layer**. The main body of work completed so far covers:
- regime diagnostics,
- class balance analysis,
- calibration distribution studies,
- compression-vs-rest research,
- compression specialist backtesting,
- and HTML dashboard reporting.

It is well-positioned for the next phase: **robust walk-forward validation and scheduled insight delivery**.