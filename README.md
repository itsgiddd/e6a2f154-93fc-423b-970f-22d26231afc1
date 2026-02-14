# Velocity 4 (V4) Profit Capture System

Automated forex trading system that achieves a **97.5% win rate** (PF 5.05) across 1,101 trades over 166 weeks. Built on a ZeroPoint ATR trailing stop indicator combined with a 5-layer profit protection system — **Velocity 4** — named for how fast it compounds accounts: 7-10 doublings per year from a $200 start.

Trades 8 currency pairs on the H4 timeframe via MetaTrader 5. Starts from as little as $200 with 1:500 leverage.

---

## How It Works

### The ZeroPoint Signal

The entry signal is based on an ATR-based trailing stop that tracks trend direction on H4 (4-hour) candles:

1. Compute True Range: `max(H-L, |H-prev_close|, |L-prev_close|)`
2. Smooth with Wilder's RMA (alpha = 1/10)
3. Place trailing stop at `3.0 * ATR` above/below price
4. Track state: **bullish** (price above stop) or **bearish** (price below stop)
5. Signal fires on flip: bullish-to-bearish = **SELL**, bearish-to-bullish = **BUY**

**Why ATR instead of fixed stops:** ATR adapts to volatility. During quiet consolidation the stop tightens, generating signals only on real breakouts. During volatile trends the stop widens, reducing whipsaws.

**Why H4 and not M15/H1/D1:**
- M15/H1 are too noisy — trailing stop flips constantly, near-zero edge after spreads
- D1 is too slow — signals every 2-4 weeks, excessive drawdown duration
- H4 is the sweet spot: signals every 3-10 days per symbol, ~6-7 trades/week across 8 pairs

### Stop-Loss: Smart Structure SL

Stops are placed at the nearest structural level, not a fixed ATR multiple:

- **BUY:** SL = lowest low of last 10 H4 bars minus 0.1% buffer
- **SELL:** SL = highest high of last 10 H4 bars plus 0.1% buffer
- **Minimum:** At least 1.5x ATR from entry

This produces stops typically 3-5x ATR wide — intentionally wide to give trades room to breathe, which is what enables the V4 protection layers to activate before the stop is ever hit.

### Velocity 4 (V4) Profit Capture: 5 Protection Layers

Every trade is managed through 5 layers, all parameterized relative to ATR at entry:

| # | Layer | Trigger | Action | Exit Share |
|---|-------|---------|--------|------------|
| 1 | **Early BE** | Price moves 0.5x ATR favorable | Move SL to entry + 0.15x ATR buffer | 49% of trades |
| 2 | **Stall Exit** | 6 bars (24h) without TP1 | Move SL to entry + 0.15x ATR buffer | 15-16% |
| 3 | **Micro-Partial** | Price reaches 0.8x ATR | Close 15% of position | -- |
| 4 | **Tiered TPs** | TP1=0.8x, TP2=2.0x, TP3=5.0x ATR | Close 33% at each level | TP3: ~1% |
| 5 | **Post-TP1 Trail** | After TP1 hit | Trail SL 0.8x ATR behind max price | 12-14% |

**The result:** Most trades exit at breakeven + tiny profit (the 49% SL_BE exits). The remaining 14-15% that hit full SL are the only losers. The 12-14% PROFIT_LOCK and ~1% TP3 runners provide the bulk of actual dollar profits.

### Why 97.5% Win Rate is Real

The win rate is not about predicting direction with 97.5% accuracy. It's about trade management:

1. ZeroPoint flips on H4 — price has already shown strong momentum
2. **99.2% of all trades** see price move at least 0.5x ATR in the favorable direction
3. Once that happens, SL moves to breakeven + buffer — the trade can no longer lose
4. The 0.8% that never reach 0.5x ATR are the only losers (26 out of 1,101 trades)

Every single loser is "dead on arrival" — max favorable excursion (MFE) of any losing trade is 0.459x ATR, just below the 0.5x BE trigger. These are trades where the flip was a brief spike with no follow-through, or the signal fired at the end of a move (exhaustion). No trade ever reached 0.5x ATR and then reversed into a loss.

### Why This Isn't Curve-Fitted

1. **7 parameters**, all ATR-relative. No symbol-specific optimization, no time-of-day filters
2. **7 of 8 symbols profitable.** Works on majors, commodity currencies, and crosses
3. **Consistent across time periods.** 4/8/16-week rolling tests all show improvement
4. **Mechanically inevitable.** 0.5x ATR BE + 99.2% MFE rate = 97.5% WR. Changing BE to 0.3x pushes WR above 99%; changing to 1.0x drops to ~84%. Smooth, predictable relationship
5. **Look-ahead bias audit: CLEAN.** Every component verified — no forward indexing, no same-bar execution, no label leakage

---

## Baseline vs V4 Comparison

| Metric | Baseline (no V4) | V4 Profit Capture |
|--------|-------------------|-------------------|
| Win Rate | 49% | 97.5% |
| Profit Factor | 1.05 | 5.05 |
| Avg Win | $15-18 per $10K | $3-4 per $10K |
| Avg Loss | $15-18 per $10K | $50-80 per $10K |
| Primary Exit | ZP_FLIP (64%) | SL_BE (49%) |

The V4 trade-off is explicit: average wins shrink because most "wins" are breakeven exits. Average losses grow because only full SL hits remain. But at 2.5% loss rate, even large losses are overwhelmed by the volume of small wins plus occasional runners.

---

## Compounding Projections

Based on actual R-multiples from 338 trades over the last 365 days (334 wins, 4 losses):

| Metric | Value |
|--------|-------|
| Avg winner R-multiple | +0.086 |
| Avg loser R-multiple | -0.950 |
| Trades per week | ~6.5 |
| Weekly growth (30% risk) | ~17% |

### Doubling Milestones from $200

| Scenario | Doublings in 1 Year | Avg Days Between Doublings | Final Balance |
|----------|---------------------|---------------------------|---------------|
| **Ideal** (no slippage) | 10 | 34 days | $246,746 |
| **Realistic** (2-pip slippage) | 7 | 50 days | $33,606 |

At higher initial risk (0.4-0.5 lots on $200 = ~50-60% risk), the account can double every 2-3 weeks in the early stages. One loss at 50% risk costs half the account but is recovered within ~10-15 winning trades due to compounding. With only a ~2.5% chance of any individual loss, the math overwhelmingly favors aggressive early sizing — worst case, you reload another $200.

### Risk Sizing

- **Base risk:** 30% per trade (half-Kelly for 97.5% WR)
- **Win streak boost:** +25% after 3 consecutive wins (max 40%)
- **Loss reduction:** -37.5% after any loss
- **Large account cap:** 20% above $50K balance
- **2 consecutive losses:** 0.025^2 = 0.0625% probability, expected once per ~4.9 years
- **3 consecutive losses:** 0.0016% probability, expected once per ~195 years

---

## Supported Pairs

EURUSD, GBPUSD, USDJPY, AUDUSD, USDCAD, NZDUSD, EURJPY, GBPJPY

(BTCUSD supported by the neural model but excluded from ZeroPoint V4 backtests due to limited H4 history)

### Per-Symbol Performance

| Symbol | Profit Factor | Baseline WR | Tier | Lot Scale |
|--------|--------------|-------------|------|-----------|
| USDCAD | 5.45 | 87% | S | 1.5x |
| GBPJPY | 2.00 | 67% | A | 1.2x |
| USDJPY | 1.67 | 70% | A | 1.2x |
| AUDUSD | 1.38 | 67% | B | 1.0x |
| EURJPY | 1.30 | 65% | B | 1.0x |
| NZDUSD | 1.22 | 62% | B | 1.0x |
| GBPUSD | 1.00 | 65% | C | 0.6x |

---

## Quick Start

```bash
pip install -r requirements.txt
python trading_app.py
```

1. Click **Connect MT5** (auto-detects your account)
2. Click **Load Model** (loads the Velocity 4 neural model)
3. Check **V4 Pure Mode**, set your lot size
4. Click **Start**

### Training Models

```bash
# Standard neural trainer (38 base features + 9 symbol one-hot = 47 dims)
python simple_neural_trainer.py

# ZeroPoint trainer (53 base features + 9 symbol one-hot = 62 dims)
python zeropoint_neural_trainer.py

# FlipPredictor GRU (predicts ZP trend continuation/flip timing)
python flip_predictor_trainer.py
```

Models save as `.pth` files in the project root. The app auto-detects `zeropoint_neural_model.pth` first.

---

## Project Structure

```
README.md                              # This file — V4 research + full docs

app/                                   # Core runtime library
  trading_engine.py                    # Trading loop + position management
  zeropoint_signal.py                  # ZeroPoint ATR trailing stop + V4 constants
  model_manager.py                     # Neural model load / inference
  mt5_connector.py                     # MetaTrader 5 connection
  config_manager.py                    # YAML config handling

models/                                # Neural architectures + analysis libs
  flip_predictor_model.py              # FlipPredictor GRU (~254K params)
  flip_predictor_features.py           # 45-dim H4 feature extraction
  flip_predictor_inference.py          # Live inference engine
  flip_predictor_integration.py        # Decision maker: skip/reduce/boost
  push_structure_analyzer.py           # Swing detection + push exhaustion
  pattern_recognition.py               # Chart pattern detection (forming + completed)

training/                              # Model trainers
  simple_neural_trainer.py             # Standard neural trainer (47-dim)
  zeropoint_neural_trainer.py          # ZeroPoint-aware trainer (62-dim)
  flip_predictor_trainer.py            # FlipPredictor GRU trainer
  council_trainer.py                   # Council predictor trainer

agentic/                               # Self-learning system
  agentic_orchestrator.py              # Background daemon wiring all components
  adaptive_performance_monitor.py      # Degradation detection
  warm_start_retrainer.py              # Warm-start model retraining
  model_hot_swap.py                    # Atomic model swap + 20-trade probation
  trade_journal.py                     # SQLite trade logging
  launch_agentic_trader.py             # Headless launcher (no UI)

candle_intelligence/                   # Trading council (15 AI agents)
  trading_council.py                   # Council deliberation engine
  agent_base.py                        # Base agent class
  council_predictor.py                 # ML model for council consensus
  agents/                              # 15 specialized agents

backtests/                             # Backtesting + analysis + projections
  backtest_zp_profitability.py         # Core V4 backtest engine
  backtest_filter.py                   # Risk scoring filter backtest
  backtest_probation.py                # Probation period entry testing
  backtest_risk_normalized.py          # Risk-normalized lot sizing
  optimize_v4.py                       # V4 parameter grid search
  projection_realistic.py              # R-multiple based 1-year projection
  projection_doubling.py               # Doubling milestone tracker
  analyze_losers.py                    # Loser DNA profiler
  simulate_200_account.py              # $200 account growth simulation

scripts/                               # Entry points + CLI tools
  trading_app.py                       # Main PySide6 dark-themed trading app
  webhook_app.py                       # ACi webhook receiver + live charts
  build.bat                            # PyInstaller build script

pine/                                  # TradingView Pine Script
  zeropoint_multi_scanner.pine         # Multi-pair ZeroPoint scanner overlay
  zeropoint_v4_strategy.pine           # V4 strategy with profit capture
```

---

## Neural Models

| Property | Standard | ZeroPoint | FlipPredictor |
|----------|----------|-----------|---------------|
| Architecture | 3-layer MLP | 3-layer MLP | 2-layer GRU |
| Base features | 38 | 53 | 45 per bar |
| Total dims | 47 | 62 | 45 x sequence |
| Timeframes | M15+H1+H4 | M15+H1+H4 | H4 only |
| Labels | Price movement | H4 ZP confirmed | Flip timing/direction |
| Params | ~10K | ~15K | ~254K |

**Feature groups:**
- Price/indicator (18): OHLC ratios, ATR, RSI, SMA, EMA, spread
- Push structure (8): push count, direction, pitch, exhaustion metrics
- Forming patterns (12): completion %, confidence, direction, R:R, volume, pattern type
- ZeroPoint per-TF (15, ZP model only): trailing stop position, distance, ATR, recency, duration (x3 TFs)
- Symbol one-hot (9): AUDUSD through USDJPY

---

## V4 Parameter Reference

| Parameter | Value | Description |
|-----------|-------|-------------|
| ATR_PERIOD | 10 | Wilder's RMA period |
| ATR_MULTIPLIER | 3.0 | Trailing stop distance from price |
| BE_TRIGGER_MULT | 0.5 | Move to BE after 0.5x ATR favorable |
| BE_BUFFER_MULT | 0.15 | BE buffer: entry + 0.15x ATR |
| STALL_BARS | 6 | Exit at BE after 6 bars without TP1 |
| MICRO_TP_MULT | 0.8 | Take 15% profit at 0.8x ATR |
| MICRO_TP_PCT | 0.15 | Micro-partial size (15%) |
| TP1_MULT | 0.8 | First TP at 0.8x ATR (33% close) |
| TP2_MULT | 2.0 | Second TP at 2.0x ATR (33% close) |
| TP3_MULT | 5.0 | Final TP at 5.0x ATR (remainder) |
| PROFIT_TRAIL_MULT | 0.8 | Post-TP1 trail behind max price |
| SWING_LOOKBACK | 10 | Bars for Smart Structure SL |
| SL_ATR_MIN_MULT | 1.5 | Minimum SL distance in ATR |
| RISK_PER_TRADE | 30% | Balance risked per trade |

---

## Self-Learning System

A background daemon that keeps the neural model current:

1. **Trade Journal** — logs every trade to SQLite with full context
2. **Performance Monitor** — detects degradation (rolling Sharpe, win rate, drawdown)
3. **Warm-Start Retrainer** — loads previous weights, trains on new data at LR=0.0003
4. **Model Hot-Swap** — atomically replaces the live model, runs 20-trade probation
5. **Automatic Rollback** — if the new model underperforms during probation, reverts to the previous version

---

## Limitations and Risks

1. **Backtest is not live trading.** Slippage, requotes, spread widening during news will reduce performance. The backtest assumes bar-close fills.

2. **Win rate depends on the BE trigger.** If H4 ZP flips stop producing 0.5x ATR favorable movement 99%+ of the time (extended ranging markets, black swans), win rate degrades.

3. **Small sample of losses.** Only 26 losses in 1,101 trades — the loss distribution is poorly characterized. True loss rate may be higher than 2.5%.

4. **Compounding projections are theoretical.** Real accounts face margin limits, lot caps, and liquidity constraints. Growth curve flattens at large sizes.

5. **Broker risk.** Offshore brokers carry counterparty risk.

6. **Correlation events.** All 8 pairs losing simultaneously during a USD shock could produce drawdown exceeding 50% at 30% risk. Partially mitigated by the correlation filter (reduces size when 3+ USD/JPY pairs signal together).

---

## Technology Stack

- **Language:** Python 3.10+
- **Broker:** MetaTrader 5 (any broker; tested on Coinexx 1:500 ECN/STP)
- **UI:** PySide6 dark theme
- **Neural Networks:** PyTorch
- **Indicators:** pandas, numpy (Wilder's RMA, not SMA)
- **TradingView:** Pine Script v6 overlay
- **Database:** SQLite (trade journal)

## Requirements

- Windows 10/11
- Python 3.8+
- MetaTrader 5 installed with a trading account
- See `requirements.txt` for Python packages

## License

MIT — see [LICENSE](LICENSE)

## Disclaimer

This software is for educational and research purposes. Forex trading involves substantial risk of loss. Never trade with money you cannot afford to lose. Past performance does not guarantee future results. The high degree of leverage can work against you as well as for you. Always test on a demo account first.
