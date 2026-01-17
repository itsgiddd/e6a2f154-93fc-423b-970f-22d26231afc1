# MT5 Windows Setup Notes

This project uses the **MetaTrader5 Python API**, which requires a local MT5 terminal running on Windows.

## 1) Install prerequisites
1. Install the MetaTrader 5 terminal from your broker (or MetaQuotes demo).
2. Install Python 3.9+ (64-bit) on Windows.
3. Install dependencies:
   ```bash
   pip install MetaTrader5 numpy pandas scipy joblib
   ```

## 2) Configure MT5 terminal
1. Launch MT5 and log in to your trading account.
2. Keep the MT5 terminal running (the Python API attaches to the running terminal).
3. Enable algorithmic trading if your broker requires it.

## 3) Run the bot
```bash
python live_trader.py
```

## 4) Notes on pattern detection & safety
- Pattern detection is **heuristic** and should be treated as **experimental**.
- The bot enforces:
  - Push-count exhaustion (skips setups after 4 pushes),
  - Minimum **1:2** risk-reward before placing trades,
  - Conservative risk sizing (2–4% baseline).

**Trading involves risk. No system can guarantee returns or specific account growth.**
