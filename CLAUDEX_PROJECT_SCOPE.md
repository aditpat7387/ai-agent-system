# Claudex - Original Project Scope and Vision

IMPORTANT: Paste this file into any new conversation with Perplexity AI to restore full project context and prevent scope drift or hallucination.

---

## What This Project Is

Claudex is a self-improving, multi-agent AI trading system built entirely locally at zero ongoing cost.

It is NOT just a static ML pipeline.
It is NOT a one-time backtest tool.
It IS a living system that learns, predicts, trades on paper, improves itself, and signals the user for manual investing decisions when confidence is high.

---

## The Original Vision

Build a system of AI agents that learn from ETH/USD 1H data, predict market movements using regime-aware ML models, execute paper trades autonomously, improve themselves over time as new data arrives, and surface high-confidence signals to the user for real manual investing decisions - all running locally at zero cost.

---

## The Multi-Agent Architecture

The system is built around 7 autonomous agents, each owning a discrete responsibility.

Agent 1 - Data Agent
Responsibility: Continuously fetches latest ETH/USD 1H OHLCV from Binance API, validates completeness, stores in DuckDB.

Agent 2 - Feature Agent
Responsibility: Rebuilds feature store (50+ indicators) and regime labels whenever new data arrives.

Agent 3 - Prediction Agent
Responsibility: Runs the regime specialist model, produces calibrated confidence scores per regime.

Agent 4 - Paper Trading Agent
Responsibility: Executes simulated trades autonomously based on Prediction Agent signals with realistic fees and slippage.

Agent 5 - Self-Improvement Agent
Responsibility: Monitors model performance, detects drift, triggers retraining when performance degrades.

Agent 6 - Signal Agent
Responsibility: Emits a HIGH-CONFIDENCE SIGNAL to the user only when calibrated probability exceeds threshold (0.85+) for real manual investing.

Agent 7 - Dashboard and Reporting Agent
Responsibility: Auto-refreshes HTML dashboard with equity curve, KPIs, trade log, and regime diagnostics after every pipeline cycle.

### Agent Communication Flow

Data Agent
  runs Feature Agent
    runs Prediction Agent
      runs Paper Trading Agent
        runs Dashboard Agent
      runs Self-Improvement Agent
        if drift detected, retrains model, loops back to Prediction Agent
      runs Signal Agent
        if confidence above 0.85, notifies User for manual investing decision

---

## Claude Code Architecture Integration

We have access to the Claude Code source repository and have decided to adopt its internal agent architecture patterns as the backbone of the Claudex agent system. This runs fully locally with no Anthropic account or API key required.

### Why Claude Code Architecture

Problem: Agents do not reliably communicate state
Solution: Subagent spawning with typed input and output contracts

Problem: No recovery on agent failure
Solution: Pre and post execution hooks for logging, validation, and rollback

Problem: Agents cannot query data natively
Solution: MCP (Model Context Protocol) servers for DuckDB, filesystem, and model registry

Problem: Hard to add new capabilities
Solution: Tool dispatch system where each capability is a registered tool

Problem: No conditional agent triggering
Solution: Orchestrator loop with reactive subagent spawning

### Orchestrator Tool Map

Claude Code Orchestrator Agent is the master loop.

Tool: run_data_agent
Action: fetches Binance candles and stores in DuckDB

Tool: run_feature_agent
Action: rebuilds feature store and regime table

Tool: run_prediction_agent
Action: produces calibrated regime specialist predictions

Tool: run_trade_agent
Action: executes paper trades and logs results

Tool: run_drift_checker
Action: compares live OOS metrics vs baseline
If drift detected, spawns Subagent: retrain_specialist
Subagent validates new model OOS and swaps if better

Tool: run_signal_emitter
Action: sends email and desktop alert if probability above 0.85

Tool: run_dashboard_agent
Action: regenerates HTML dashboard

### MCP Servers Planned

duckdb-mcp
Exposes: All DuckDB tables (features, predictions, trades, regimes) queryable by any agent

filesystem-mcp
Exposes: Model joblib files, config YAMLs, artifact CSVs

model-registry-mcp
Exposes: Active model metadata, version history, drift scores

### Key Architecture Principles from Claude Code

1. Every agent is a tool with typed function signatures, not a loose script
2. Hooks on every tool call for pre-execution validation, post-execution logging, and error rollback
3. Subagents are spawned reactively - the retraining subagent only runs when drift is detected
4. State lives in DuckDB - no in-memory state passed between agents, all state is persisted and queryable
5. Orchestrator is stateless - the master loop reads from DuckDB on every cycle, no global variables

---

## Productionization at Zero Cost

The agreed production strategy is local-first, zero cloud cost, using only free tools.

Component: Orchestration and Scheduling
Tool: Windows Task Scheduler (Phase 1) then Claude Code Orchestrator (Phase 2)
Cost: Free

Component: Agent Framework
Tool: Claude Code architecture patterns (local, no API key)
Cost: Free

Component: MCP Servers
Tool: Local MCP servers for DuckDB, filesystem, model registry
Cost: Free

Component: Database
Tool: DuckDB (file-based, no server)
Cost: Free

Component: Model Storage
Tool: Joblib files on local disk
Cost: Free

Component: Notifications and Signals
Tool: Email via SMTP Gmail free tier or Windows Toast notification
Cost: Free

Component: Dashboard
Tool: Static HTML file auto-generated and opened in browser
Cost: Free

Component: Data Source
Tool: Binance REST API (free public endpoints, no auth needed for OHLCV)
Cost: Free

Component: Compute
Tool: Local machine only (Windows 11 laptop or desktop)
Cost: Free

Component: Version Control
Tool: Git and GitHub free tier
Cost: Free

### Zero-Cost Production Run Cycle (Hourly)

Step 1: Task Scheduler triggers Orchestrator every hour
Step 2: Data Agent fetches new candles and stores in DuckDB
Step 3: Feature Agent rebuilds feature store
Step 4: Prediction Agent generates calibrated regime specialist predictions
Step 5: Paper Trading Agent logs simulated trades
Step 6: Drift Checker compares OOS metrics vs baseline
         If drift detected, Retraining Subagent runs, validates, and swaps model if better
Step 7: Signal Agent sends email and desktop alert if confidence above 0.85
Step 8: Dashboard Agent regenerates HTML dashboard

---

## Current ML Foundation (Already Built)

The following components are complete and form the base for the agent system:

- Binance ETH/USD 1H data ingestion into DuckDB - DONE
- Feature store v2 with 50+ technical indicators - DONE
- Regime table with compression and expansion labeling - DONE
- Event targets with forward-looking labels - DONE
- Baseline models: HistGradientBoosting and LogisticRegression - DONE
- Walk-forward OOS validation - DONE
- Regime diagnostics (compression identified as hardest regime) - DONE
- Calibration study with isotonic calibration applied - DONE
- Compression specialist model v7 - DONE
- Threshold sweep from 0.75 to 0.90, optimal at 0.85 - DONE
- Compression specialist backtest with fees and slippage - DONE
- Paper trading simulation v7 - DONE
- HTML dashboard v1 and v2 - DONE

---

## What Has NOT Been Built Yet

Agent orchestration layer
Note: Each script needs to become an agent with typed tool contracts following Claude Code pattern

MCP servers
Note: DuckDB, filesystem, and model registry as local MCP servers

Automated scheduling
Note: Windows Task Scheduler triggering the orchestrator hourly

Self-improvement and retraining loop
Note: Drift detection plus conditional subagent retraining

High-confidence signal emitter
Note: Email and Windows Toast when probability exceeds 0.85

Live dashboard auto-refresh
Note: Dashboard regenerates automatically after every cycle

Walk-forward robustness test at 0.85
Note: Script written but not yet executed and validated

Model registry
Note: Version tracking, OOS history, drift scores per model

Multi-asset expansion
Note: BTC/USD and SOL/USD using the same pipeline

---

## Milestone Tracker

M1: Data + Features + Regime - COMPLETE
M2: Baseline Models + Walk-Forward - COMPLETE
M3: Compression Specialist + Calibration - COMPLETE
M4: Backtest + Paper Trading + Dashboard - COMPLETE
M5: Agent Orchestration Layer using Claude Code patterns - NOT STARTED
M6: MCP Servers for DuckDB, Filesystem, and Model Registry - NOT STARTED
M7: Automated Zero-Cost Production Pipeline - NOT STARTED
M8: Signal Emitter and User Alerts - NOT STARTED
M9: Self-Improvement and Retraining Loop - NOT STARTED
M10: Multi-Asset Expansion - NOT STARTED

---

## Profitability Assessment

Win rate
Current static pipeline: approximately 50% in compression regime
Target with full agent system: 55 to 62 percent with self-improving loop

Trade selectivity
Current: All compression rows traded
Target: Only 0.85+ confidence signals, approximately 15 to 20 percent of rows

Max drawdown
Current: -4.97% observed
Target: Manageable with stop-loss agent

Edge
Current: Weak but present in compression regime
Target: Strengthens significantly with retraining loop

Profitability
Current: Not yet profitable, fees eat the edge
Target: Viable at 0.85+ threshold with fee-aware position sizing

The compression regime specialist at threshold 0.85 is the real edge. The system must trade ONLY when it is highly confident. Selectivity is the profit mechanism.

---

## Key Design Principles (Never Deviate From These)

1. Zero cloud cost - everything runs locally, no paid APIs, no cloud compute
2. Regime-aware - one model does NOT fit all market conditions, always use specialist models per regime
3. Calibrated confidence - never use raw probabilities, always apply isotonic calibration before threshold decisions
4. Walk-forward only - no data leakage, always validate OOS with expanding windows
5. Paper trading first - never touch real money, system is for paper trading and manual signal alerts only
6. Self-improving - the system must detect its own degradation and retrain without human intervention
7. Local-first agents - agents are Python tools coordinated by Claude Code orchestrator pattern, not cloud functions
8. Claude Code architecture - use subagent spawning, tool dispatch, hooks, and MCP servers as the agent framework
9. State in DuckDB - no in-memory state between agents, all state is persisted and queryable
10. Selectivity over frequency - fewer high-confidence trades beat many low-confidence trades

---

## Tech Stack (Locked)

Language: Python 3.11
Database: DuckDB (local file)
ML Models: HistGradientBoosting and LogisticRegression from sklearn
Calibration: IsotonicRegression from sklearn
Agent Framework: Claude Code architecture (subagents, tools, hooks, MCP)
MCP Servers: Local Python MCP servers for DuckDB, filesystem, and model registry
Data Source: Binance REST API (free OHLCV)
Storage: Parquet, DuckDB, Joblib
Orchestration: Windows Task Scheduler (Phase 1) then Claude Code Orchestrator (Phase 2)
Config: YAML
Notifications: SMTP Email via Gmail free tier and Windows Toast
Dashboard: Static HTML auto-generated
Version Control: Git and GitHub
Environment: Windows 11, PowerShell, venv, VS Code

---

## Project Root

C:\Claudex\

---

## Instructions for AI (Perplexity or Claude)

When this file is provided at the start of a conversation, follow these rules:

1. Do not hallucinate features that are not listed in the Already Built section
2. Do not suggest cloud solutions - this is a zero-cost local project
3. Always refer to the agent architecture when discussing next steps
4. Do not treat this as a finished project - Milestones M5 through M10 are not started
5. The next immediate step is M5 - building the agent orchestration layer using Claude Code patterns
6. Compression regime is the focus - do not generalize to all regimes without justification
7. Threshold is 0.85 - do not suggest lower thresholds without walk-forward evidence
8. Claude Code architecture is the agent framework - use its subagent, tool, hook, and MCP patterns
9. All agent state must go through DuckDB - never design in-memory state passing between agents
10. Profitability comes from selectivity - the system wins by NOT trading most of the time

---

Last updated: April 18, 2026 - v3 (includes Claude Code Architecture)
Project owner: Claudex local machine, Kolhapur, Maharashtra, India
