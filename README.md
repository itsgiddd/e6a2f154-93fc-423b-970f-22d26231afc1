# Neural Forex Trader

Automated forex trading system powered by a neural network and ZeroPoint ATR trailing stop signals. Trades 9 currency pairs on MetaTrader 5 with a dark-themed PySide6 desktop app.

## What It Does

- Trains a neural model on M15 price data with multi-timeframe features (M15 / H1 / H4)
- Generates BUY/SELL signals using ZeroPoint ATR trailing stops on H4
- Executes trades automatically on MT5 with configurable lot sizing
- Manages open positions with trailing stops, break-even moves, and max-loss cutoffs
- Closes all trades when a dollar profit target is hit
- Self-learning: monitors performance, retrains on new data, hot-swaps models

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Launch the trading app
python trading_app.py
```

1. Click **Connect MT5** (auto-detects your account)
2. Click **Load Model** (loads the ZeroPoint neural model)
3. Check **ZeroPoint Pure Mode**, set your lot size
4. Click **Start**

## Supported Pairs

EURUSD, GBPUSD, USDJPY, AUDUSD, USDCAD, NZDUSD, EURJPY, GBPJPY, BTCUSD

## Training a Model

```bash
# Standard neural trainer (38 base features + symbol one-hot)
python simple_neural_trainer.py

# ZeroPoint trainer (53 base features + symbol one-hot = 62 dims)
python zeropoint_neural_trainer.py
```

Models are saved as `.pth` files in the project root. The app auto-detects `zeropoint_neural_model.pth` first.

## Project Structure

```
trading_app.py                  # Main app — PySide6 dark UI
app/
  trading_engine.py             # Core trading loop + position management
  zeropoint_signal.py           # ZeroPoint ATR trailing stop signals
  model_manager.py              # Neural model load / inference
  mt5_connector.py              # MetaTrader 5 connection
  config_manager.py             # YAML config handling
simple_neural_trainer.py        # Standard neural model trainer
zeropoint_neural_trainer.py     # ZeroPoint-aware trainer (62-dim)
zeropoint_backtest.py           # Backtest ZP signals historically
zp_trade_now.py                 # Quick CLI: scan + place best ZP trade
zp_scan.py                      # Scan all pairs for ZP signals
pattern_recognition.py          # Chart pattern detection (forming + completed)
push_structure_analyzer.py      # Swing detection + push exhaustion
ai_brain.py                     # Rule-based trade decision engine
trade_validator.py              # Trade validation rules
adaptive_risk.py                # Dynamic risk adjustment
agentic_orchestrator.py         # Self-learning daemon
adaptive_performance_monitor.py # Degradation detection
warm_start_retrainer.py         # Warm-start model retraining
model_hot_swap.py               # Atomic model swap + probation
trade_journal.py                # SQLite trade logging
launch_agentic_trader.py        # Headless launcher (no UI)
```

## Features

### ZeroPoint Pure Mode
Bypasses the neural confidence gate and trades directly on H4 ATR trailing stop flips. Fixed lot sizing, no correlation filters — just clean ZP signals.

### Trade Monitor
- **Max loss cutoff** — auto-close if a position loses more than $X
- **Break-even SL** — move stop to entry after Y pips of profit
- **ZP trailing stop** — trail SL using the ATR stop level
- **ZP flip exit** — close when ZeroPoint flips against you

### Dollar Profit Target
Set a total P/L goal (e.g. $120). When all open positions combined reach that profit, everything closes automatically.

### Live Settings Sync
Change lot size, pairs, risk, or any setting while trading — updates apply immediately without restarting.

### Per-Trade Timers
Every open position shows a live duration timer (e.g. `2h 34m`) so you can see how long each trade has been running.

### Self-Learning System
Background daemon that monitors trade performance, detects model degradation, retrains with warm-start on fresh data, and hot-swaps the model with a 20-trade probation period + automatic rollback.

## Neural Model

| Property | Standard | ZeroPoint |
|----------|----------|-----------|
| Base features | 38 | 53 |
| Total dims (9 symbols) | 47 | 62 |
| Timeframes | M15 + H1 + H4 | M15 + H1 + H4 |
| Labels | Price movement | H4 ZP confirmed |
| Architecture | 3-layer MLP | 3-layer MLP |

**Feature groups**: price/indicator (18) + push structure (8) + forming patterns (12) + ZeroPoint per-TF (15, ZP model only) + symbol one-hot (9)

## Requirements

- Windows 10/11
- Python 3.8+
- MetaTrader 5 installed with a trading account
- See `requirements.txt` for Python packages

## License

MIT — see [LICENSE](LICENSE)

## Disclaimer

This software is for educational and research purposes. Forex trading involves substantial risk of loss. Never trade with money you cannot afford to lose. Always test on a demo account first.
