# Fix Model Loading and Feature Scaling in Live Trader

## User Review Required
None. This is a bug fix for runtime initialization.

## Proposed Changes
### Trading Bot
#### [MODIFY] [live_trader.py](file:///C:/Users/Shadow/.gemini/antigravity/brain/e6a2f154-93fc-423b-970f-22d26231afc1/live_trader.py)
- Update `load_model` to check if loaded object is a `dict`.
- If `dict`, extract `model` and `scaler`.
- Update `run` loop to apply `scaler.transform()` to features before passing to `model.predict()`.

## Verification Plan
### Automated Tests
- Run `python live_trader.py` and verify it connects and enters the trading loop without crashing.
