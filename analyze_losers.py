#!/usr/bin/env python3
"""
LOSER DNA PROFILER — Find what makes the 2.5% losers different from winners.
Extracts every possible feature at the moment of entry for both losers and winners,
then runs statistical comparisons to find discriminating characteristics.

Goal: Find filters that can skip "dead on arrival" trades BEFORE entering.
"""

import sys, os
import numpy as np
import pandas as pd
from collections import defaultdict
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "app"))

import MetaTrader5 as mt5
from app.zeropoint_signal import (
    compute_zeropoint_state, ATR_PERIOD, ATR_MULTIPLIER,
    BE_TRIGGER_MULT, BE_BUFFER_MULT, PROFIT_TRAIL_DISTANCE_MULT,
    STALL_BARS, MICRO_TP_MULT, MICRO_TP_PCT,
    TP1_MULT_AGG, TP2_MULT_AGG, TP3_MULT_AGG,
    SL_BUFFER_PCT, SL_ATR_MIN_MULT, SWING_LOOKBACK,
)

SYMBOLS = ["AUDUSD", "EURJPY", "EURUSD", "GBPJPY", "GBPUSD", "NZDUSD", "USDCAD", "USDJPY"]
FETCH_BARS = 5000
WARMUP_BARS = 50
LOT_SIZE = 1.0


def contract_size(sym):
    return 1.0 if "BTC" in sym.upper() else 100_000.0


def pip_value(sym):
    return 0.01 if "JPY" in sym.upper() else 0.0001


def resolve_symbol(ticker):
    for c in [ticker, ticker + ".raw", ticker + "m", ticker + ".a", ticker + ".e"]:
        info = mt5.symbol_info(c)
        if info is not None:
            mt5.symbol_select(c, True)
            return c
    return None


def compute_smart_sl(df, bar_idx, direction, atr_val):
    lookback_start = max(0, bar_idx - SWING_LOOKBACK + 1)
    cur_close = float(df["close"].iloc[bar_idx])
    if direction == "BUY":
        recent_low = float(df["low"].iloc[lookback_start:bar_idx + 1].min())
        buffer = recent_low * SL_BUFFER_PCT
        structural = recent_low - buffer
        min_sl = cur_close - atr_val * SL_ATR_MIN_MULT
        return min_sl if structural > min_sl else structural
    else:
        recent_high = float(df["high"].iloc[lookback_start:bar_idx + 1].max())
        buffer = recent_high * SL_BUFFER_PCT
        structural = recent_high + buffer
        min_sl = cur_close + atr_val * SL_ATR_MIN_MULT
        return min_sl if structural < min_sl else structural


# ═══════════════════════════════════════════════════════════════
# V4 Position (same as backtest)
# ═══════════════════════════════════════════════════════════════
class V4Position:
    def __init__(self, sym, direction, entry, sl, atr_val, lot, cont_sz, entry_time=None):
        self.sym = sym
        self.direction = direction
        self.entry = entry
        self.sl = sl
        self.original_sl = sl
        self.atr_val = atr_val
        self.total_lot = lot
        self.remaining_lot = lot
        self.contract_size = cont_sz
        self.entry_time = entry_time
        self.exit_time = None
        sign = 1 if direction == "BUY" else -1
        self.tp1 = entry + sign * TP1_MULT_AGG * atr_val
        self.tp2 = entry + sign * TP2_MULT_AGG * atr_val
        self.tp3 = entry + sign * TP3_MULT_AGG * atr_val
        self.tp1_hit = False
        self.tp2_hit = False
        self.tp3_hit = False
        self.closed = False
        self.partials = []
        self.bars_in_trade = 0
        self.max_profit_reached = 0.0
        self.max_favorable_price = entry
        self.be_activated = False
        self.profit_lock_sl = None
        self.profit_lock_active = False
        self.micro_tp_hit = False
        self.stall_be_activated = False
        self.exit_type = None

    def partial_lot(self):
        return max(0.01, round(self.total_lot / 3, 2))

    def pnl_for_price(self, price, lot):
        if self.direction == 'BUY':
            return (price - self.entry) * lot * self.contract_size
        else:
            return (self.entry - price) * lot * self.contract_size

    def check_bar(self, high, low, close, confirmed_pos):
        if self.closed:
            return
        self.bars_in_trade += 1
        is_buy = self.direction == 'BUY'
        atr = self.atr_val

        if is_buy:
            if high > self.max_favorable_price:
                self.max_favorable_price = high
            cur_profit = high - self.entry
        else:
            if low < self.max_favorable_price:
                self.max_favorable_price = low
            cur_profit = self.entry - low
        if cur_profit > self.max_profit_reached:
            self.max_profit_reached = cur_profit

        # Post-TP1 trail
        if self.tp1_hit:
            trail_dist = PROFIT_TRAIL_DISTANCE_MULT * atr
            if is_buy:
                new_lock = self.max_favorable_price - trail_dist
                if new_lock > self.entry and (self.profit_lock_sl is None or new_lock > self.profit_lock_sl):
                    self.profit_lock_sl = new_lock
                    self.profit_lock_active = True
            else:
                new_lock = self.max_favorable_price + trail_dist
                if new_lock < self.entry and (self.profit_lock_sl is None or new_lock < self.profit_lock_sl):
                    self.profit_lock_sl = new_lock
                    self.profit_lock_active = True

        # Early BE
        if not self.be_activated:
            if self.max_profit_reached >= BE_TRIGGER_MULT * atr:
                be_buf = BE_BUFFER_MULT * atr
                if is_buy:
                    new_sl = self.entry + be_buf
                    if new_sl > self.sl:
                        self.sl = new_sl
                        self.be_activated = True
                else:
                    new_sl = self.entry - be_buf
                    if new_sl < self.sl:
                        self.sl = new_sl
                        self.be_activated = True

        # Stall exit
        if not self.tp1_hit and not self.stall_be_activated:
            if self.bars_in_trade >= STALL_BARS:
                be_buf = BE_BUFFER_MULT * atr
                if is_buy:
                    new_sl = self.entry + be_buf
                    if new_sl > self.sl:
                        self.sl = new_sl
                        self.stall_be_activated = True
                        self.be_activated = True
                else:
                    new_sl = self.entry - be_buf
                    if new_sl < self.sl:
                        self.sl = new_sl
                        self.stall_be_activated = True
                        self.be_activated = True

        # Micro-partial
        if not self.micro_tp_hit and not self.tp1_hit:
            micro_price = (self.entry + MICRO_TP_MULT * atr) if is_buy else (self.entry - MICRO_TP_MULT * atr)
            micro_triggered = (is_buy and high >= micro_price) or (not is_buy and low <= micro_price)
            if micro_triggered:
                self.micro_tp_hit = True
                micro_lot = round(self.total_lot * MICRO_TP_PCT, 2)
                micro_lot = max(0.01, min(micro_lot, self.remaining_lot))
                self.partials.append(self.pnl_for_price(micro_price, micro_lot))
                self.remaining_lot = round(self.remaining_lot - micro_lot, 2)
                if self.remaining_lot <= 0:
                    self.closed = True
                    self.exit_type = 'MICRO_TP'
                    return

        # TP1
        if not self.tp1_hit:
            if (is_buy and high >= self.tp1) or (not is_buy and low <= self.tp1):
                self.tp1_hit = True
                partial = min(self.partial_lot(), self.remaining_lot)
                self.partials.append(self.pnl_for_price(self.tp1, partial))
                self.remaining_lot = round(self.remaining_lot - partial, 2)
                if self.remaining_lot <= 0:
                    self.closed = True
                    self.exit_type = 'TP1'
                    return

        # TP2
        if self.tp1_hit and not self.tp2_hit:
            if (is_buy and high >= self.tp2) or (not is_buy and low <= self.tp2):
                self.tp2_hit = True
                self.sl = self.entry
                self.be_activated = True
                partial = min(self.partial_lot(), self.remaining_lot)
                self.partials.append(self.pnl_for_price(self.tp2, partial))
                self.remaining_lot = round(self.remaining_lot - partial, 2)
                if self.remaining_lot <= 0:
                    self.closed = True
                    self.exit_type = 'TP2'
                    return

        # TP3
        if self.tp2_hit and not self.tp3_hit:
            if (is_buy and high >= self.tp3) or (not is_buy and low <= self.tp3):
                self.tp3_hit = True
                self.partials.append(self.pnl_for_price(self.tp3, self.remaining_lot))
                self.remaining_lot = 0
                self.closed = True
                self.exit_type = 'TP3'
                return

        # Profit lock
        if self.profit_lock_active and self.profit_lock_sl is not None:
            lock_hit = (is_buy and low <= self.profit_lock_sl) or (not is_buy and high >= self.profit_lock_sl)
            if lock_hit:
                self.partials.append(self.pnl_for_price(self.profit_lock_sl, self.remaining_lot))
                self.remaining_lot = 0
                self.closed = True
                self.exit_type = 'PROFIT_LOCK'
                return

        # SL
        if (is_buy and low <= self.sl) or (not is_buy and high >= self.sl):
            self.partials.append(self.pnl_for_price(self.sl, self.remaining_lot))
            self.remaining_lot = 0
            self.closed = True
            if self.stall_be_activated:
                self.exit_type = 'SL_STALL'
            elif self.be_activated:
                self.exit_type = 'SL_BE'
            else:
                self.exit_type = 'SL'
            return

        # ZP Flip
        if confirmed_pos != 0:
            if (is_buy and confirmed_pos == -1) or (not is_buy and confirmed_pos == 1):
                self.partials.append(self.pnl_for_price(close, self.remaining_lot))
                self.remaining_lot = 0
                self.closed = True
                self.exit_type = 'ZP_FLIP'
                return

    def force_close(self, price):
        if self.closed or self.remaining_lot <= 0:
            return
        self.partials.append(self.pnl_for_price(price, self.remaining_lot))
        self.remaining_lot = 0
        self.closed = True
        self.exit_type = 'END'

    @property
    def total_pnl(self):
        return sum(self.partials)


# ═══════════════════════════════════════════════════════════════
# ENTRY FEATURE EXTRACTOR — captures everything about the bar
# ═══════════════════════════════════════════════════════════════
def extract_entry_features(df, idx, direction, atr_val):
    """Extract every possible feature at the moment of signal, BEFORE entry."""
    row = df.iloc[idx]
    close = float(row["close"])
    high = float(row["high"])
    low = float(row["low"])
    opn = float(row["open"])
    vol = float(row.get("volume", row.get("tick_volume", 0)))

    features = {}

    # ── 1. ATR-based features ──
    features["atr"] = atr_val
    atr_median = float(row.get("atr_median_100", atr_val))
    features["atr_vs_median"] = atr_val / atr_median if atr_median > 0 else 1.0  # >1 = high vol

    # ATR trend: is ATR expanding or contracting?
    if idx >= 5:
        atr_5ago = float(df["atr"].iloc[idx - 5])
        features["atr_change_5bar"] = (atr_val - atr_5ago) / atr_5ago if atr_5ago > 0 else 0
    else:
        features["atr_change_5bar"] = 0

    if idx >= 10:
        atr_10ago = float(df["atr"].iloc[idx - 10])
        features["atr_change_10bar"] = (atr_val - atr_10ago) / atr_10ago if atr_10ago > 0 else 0
    else:
        features["atr_change_10bar"] = 0

    # ── 2. Signal bar characteristics ──
    bar_range = high - low
    features["bar_range_vs_atr"] = bar_range / atr_val if atr_val > 0 else 1.0  # big bar = strong flip
    features["bar_body_vs_range"] = abs(close - opn) / bar_range if bar_range > 0 else 0  # body ratio

    # Is the signal bar bullish or bearish?
    bar_bullish = close > opn
    features["signal_bar_agrees"] = 1 if ((direction == "BUY" and bar_bullish) or
                                           (direction == "SELL" and not bar_bullish)) else 0

    # Upper/lower wick ratios
    if bar_range > 0:
        if close >= opn:  # bullish bar
            features["upper_wick_pct"] = (high - close) / bar_range
            features["lower_wick_pct"] = (opn - low) / bar_range
        else:  # bearish bar
            features["upper_wick_pct"] = (high - opn) / bar_range
            features["lower_wick_pct"] = (close - low) / bar_range
    else:
        features["upper_wick_pct"] = 0
        features["lower_wick_pct"] = 0

    # ── 3. Volume features ──
    if idx >= 10:
        vol_sma10 = float(df["volume"].iloc[idx-10:idx].mean()) if "volume" in df else 0
        features["vol_vs_avg"] = vol / vol_sma10 if vol_sma10 > 0 else 1.0
    else:
        features["vol_vs_avg"] = 1.0

    # ── 4. Bars since flip / bars in position ──
    features["bars_in_position"] = int(row.get("bars_in_position", 1))
    features["bars_since_flip"] = int(row.get("bars_since_flip", 0))

    # ── 5. Previous bars momentum ──
    # How much did price move in the signal direction over the last N bars?
    for lookback in [3, 5, 10]:
        if idx >= lookback:
            prev_close = float(df["close"].iloc[idx - lookback])
            move = (close - prev_close) / atr_val if atr_val > 0 else 0
            if direction == "SELL":
                move = -move  # normalize: positive = signal direction
            features[f"momentum_{lookback}bar_atr"] = move
        else:
            features[f"momentum_{lookback}bar_atr"] = 0

    # ── 6. SL distance ──
    smart_sl = compute_smart_sl(df, idx, direction, atr_val)
    sl_dist = abs(close - smart_sl)
    features["sl_dist_atr"] = sl_dist / atr_val if atr_val > 0 else 0
    features["sl_dist_pips"] = sl_dist / pip_value(df.attrs.get("symbol", "EURUSD")) if hasattr(df, 'attrs') else sl_dist

    # ── 7. Trailing stop distance ──
    trailing_stop = float(row.get("xATRTrailingStop", close))
    features["ts_dist_atr"] = abs(close - trailing_stop) / atr_val if atr_val > 0 else 0

    # ── 8. Price position relative to recent range ──
    if idx >= 20:
        high_20 = float(df["high"].iloc[idx-20:idx].max())
        low_20 = float(df["low"].iloc[idx-20:idx].min())
        range_20 = high_20 - low_20
        if range_20 > 0:
            features["price_position_20"] = (close - low_20) / range_20  # 0=at low, 1=at high
        else:
            features["price_position_20"] = 0.5
    else:
        features["price_position_20"] = 0.5

    if idx >= 50:
        high_50 = float(df["high"].iloc[idx-50:idx].max())
        low_50 = float(df["low"].iloc[idx-50:idx].min())
        range_50 = high_50 - low_50
        if range_50 > 0:
            features["price_position_50"] = (close - low_50) / range_50
        else:
            features["price_position_50"] = 0.5
    else:
        features["price_position_50"] = 0.5

    # ── 9. Consecutive signal direction ──
    # How many recent signals were in the same direction? (whipsaw detection)
    same_dir_count = 0
    if idx >= 2:
        for j in range(idx - 1, max(idx - 20, 0), -1):
            r = df.iloc[j]
            if direction == "BUY" and bool(r.get("buy_signal", False)):
                same_dir_count += 1
            elif direction == "SELL" and bool(r.get("sell_signal", False)):
                same_dir_count += 1
            elif bool(r.get("buy_signal", False)) or bool(r.get("sell_signal", False)):
                break  # hit a signal in opposite direction
    features["recent_same_dir_signals"] = same_dir_count

    # ── 10. Time features ──
    t = row["time"]
    if hasattr(t, 'hour'):
        features["hour"] = t.hour
        features["day_of_week"] = t.dayofweek  # 0=Mon, 4=Fri
        features["is_friday"] = 1 if t.dayofweek == 4 else 0
        features["is_monday"] = 1 if t.dayofweek == 0 else 0
        # Session: 0-8 Asian, 8-16 London, 16-24 NY
        if t.hour < 8:
            features["session"] = 0  # Asian
        elif t.hour < 16:
            features["session"] = 1  # London
        else:
            features["session"] = 2  # NY
    else:
        features["hour"] = 0
        features["day_of_week"] = 0
        features["is_friday"] = 0
        features["is_monday"] = 0
        features["session"] = 0

    # ── 11. Recent price action patterns ──
    # Count how many of last 5 bars were in the same direction as signal
    if idx >= 5:
        bullish_count = 0
        for j in range(idx - 5, idx):
            if float(df["close"].iloc[j]) > float(df["open"].iloc[j]):
                bullish_count += 1
        if direction == "BUY":
            features["recent_bar_agreement"] = bullish_count / 5
        else:
            features["recent_bar_agreement"] = (5 - bullish_count) / 5
    else:
        features["recent_bar_agreement"] = 0.5

    # ── 12. Gap at signal ──
    # Was there a gap between previous close and current open?
    if idx >= 1:
        prev_close = float(df["close"].iloc[idx - 1])
        gap = abs(opn - prev_close) / atr_val if atr_val > 0 else 0
        features["gap_atr"] = gap
    else:
        features["gap_atr"] = 0

    return features


def main():
    print("=" * 100)
    print("  LOSER DNA PROFILER - V4 ZeroPoint")
    print("  Analyzing what makes the 2.5% losers different from winners")
    print("=" * 100)

    if not mt5.initialize():
        print("ERROR: Could not initialize MT5")
        return

    # Load data
    symbol_data = {}
    print("\nLoading H4 data...")
    for sym in SYMBOLS:
        resolved = resolve_symbol(sym)
        if resolved is None:
            continue
        rates = mt5.copy_rates_from_pos(resolved, mt5.TIMEFRAME_H4, 0, FETCH_BARS)
        if rates is None or len(rates) < 100:
            continue
        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        df.rename(columns={"tick_volume": "volume"}, inplace=True)
        df_zp = compute_zeropoint_state(df)
        if df_zp is not None and len(df_zp) >= WARMUP_BARS:
            df_zp.attrs["symbol"] = sym
            symbol_data[sym] = df_zp
            print(f"  {sym}: {len(df_zp)} bars")
    mt5.shutdown()

    if not symbol_data:
        print("No data!")
        return

    # ═══════════════════════════════════════════════════════════════
    # RUN SIMULATION + COLLECT FEATURES AT ENTRY
    # ═══════════════════════════════════════════════════════════════
    print("\nSimulating trades and extracting entry features...")

    trade_records = []  # list of {features..., outcome, pnl, exit_type, mfe, ...}

    for sym, df in symbol_data.items():
        cont_sz = contract_size(sym)
        n = len(df)
        pos_obj = None

        for i in range(WARMUP_BARS, n):
            row = df.iloc[i]
            high = float(row["high"])
            low = float(row["low"])
            close = float(row["close"])
            atr_val = float(row["atr"])
            confirmed_pos = int(row.get("pos", 0))
            buy_sig = bool(row.get("buy_signal", False))
            sell_sig = bool(row.get("sell_signal", False))

            if np.isnan(atr_val) or atr_val <= 0:
                continue

            # Check active position
            if pos_obj is not None and not pos_obj.closed:
                pos_obj.check_bar(high, low, close, confirmed_pos)
                if pos_obj.closed:
                    pos_obj.exit_time = row["time"]
                    # Record this completed trade
                    rec = pos_obj._entry_features.copy()
                    rec["symbol"] = pos_obj.sym
                    rec["direction"] = pos_obj.direction
                    rec["pnl"] = pos_obj.total_pnl
                    rec["exit_type"] = pos_obj.exit_type
                    rec["bars_in_trade"] = pos_obj.bars_in_trade
                    rec["mfe_atr"] = pos_obj.max_profit_reached / pos_obj.atr_val if pos_obj.atr_val > 0 else 0
                    rec["outcome"] = "WIN" if pos_obj.total_pnl > 0 else "LOSS"
                    rec["is_loser"] = 1 if pos_obj.total_pnl <= 0 else 0
                    rec["entry_time"] = pos_obj.entry_time
                    rec["exit_time_val"] = pos_obj.exit_time
                    trade_records.append(rec)
                    pos_obj = None

            # Open new position on signal
            if buy_sig or sell_sig:
                direction = "BUY" if buy_sig else "SELL"

                # Force close existing
                if pos_obj is not None and not pos_obj.closed:
                    pos_obj.force_close(close)
                    pos_obj.exit_time = row["time"]
                    rec = pos_obj._entry_features.copy()
                    rec["symbol"] = pos_obj.sym
                    rec["direction"] = pos_obj.direction
                    rec["pnl"] = pos_obj.total_pnl
                    rec["exit_type"] = pos_obj.exit_type
                    rec["bars_in_trade"] = pos_obj.bars_in_trade
                    rec["mfe_atr"] = pos_obj.max_profit_reached / pos_obj.atr_val if pos_obj.atr_val > 0 else 0
                    rec["outcome"] = "WIN" if pos_obj.total_pnl > 0 else "LOSS"
                    rec["is_loser"] = 1 if pos_obj.total_pnl <= 0 else 0
                    rec["entry_time"] = pos_obj.entry_time
                    rec["exit_time_val"] = pos_obj.exit_time
                    trade_records.append(rec)

                smart_sl = compute_smart_sl(df, i, direction, atr_val)
                pos_obj = V4Position(
                    sym=sym, direction=direction, entry=close, sl=smart_sl,
                    atr_val=atr_val, lot=LOT_SIZE, cont_sz=cont_sz, entry_time=row["time"],
                )

                # Extract features AT ENTRY (before we know the outcome)
                pos_obj._entry_features = extract_entry_features(df, i, direction, atr_val)

        # Close remaining
        if pos_obj is not None and not pos_obj.closed:
            pos_obj.force_close(float(df.iloc[-1]["close"]))
            pos_obj.exit_time = df.iloc[-1]["time"]
            rec = pos_obj._entry_features.copy()
            rec["symbol"] = pos_obj.sym
            rec["direction"] = pos_obj.direction
            rec["pnl"] = pos_obj.total_pnl
            rec["exit_type"] = pos_obj.exit_type
            rec["bars_in_trade"] = pos_obj.bars_in_trade
            rec["mfe_atr"] = pos_obj.max_profit_reached / pos_obj.atr_val if pos_obj.atr_val > 0 else 0
            rec["outcome"] = "WIN" if pos_obj.total_pnl > 0 else "LOSS"
            rec["is_loser"] = 1 if pos_obj.total_pnl <= 0 else 0
            rec["entry_time"] = pos_obj.entry_time
            rec["exit_time_val"] = pos_obj.exit_time
            trade_records.append(rec)

    # ═══════════════════════════════════════════════════════════════
    # ANALYSIS
    # ═══════════════════════════════════════════════════════════════
    df_trades = pd.DataFrame(trade_records)
    total = len(df_trades)
    losers = df_trades[df_trades["is_loser"] == 1]
    winners = df_trades[df_trades["is_loser"] == 0]
    n_losers = len(losers)
    n_winners = len(winners)

    print(f"\n{'=' * 100}")
    print(f"  RESULTS: {total} trades | {n_winners} winners ({n_winners/total*100:.1f}%) | {n_losers} losers ({n_losers/total*100:.1f}%)")
    print(f"{'=' * 100}")

    # ── Feature comparison: Losers vs Winners ──
    feature_cols = [c for c in df_trades.columns if c not in [
        "symbol", "direction", "pnl", "exit_type", "bars_in_trade", "mfe_atr",
        "outcome", "is_loser", "entry_time", "exit_time_val"
    ]]

    print(f"\n  FEATURE COMPARISON: Losers vs Winners")
    print(f"  {'Feature':<30} | {'Loser Avg':>12} | {'Winner Avg':>12} | {'Diff':>10} | {'Loser Med':>12} | {'Winner Med':>12} | {'Separation':>12}")
    print(f"  " + "-" * 115)

    separations = []
    for feat in sorted(feature_cols):
        try:
            l_vals = losers[feat].astype(float)
            w_vals = winners[feat].astype(float)
            l_mean = l_vals.mean()
            w_mean = w_vals.mean()
            l_med = l_vals.median()
            w_med = w_vals.median()
            diff = l_mean - w_mean

            # Cohen's d effect size
            pooled_std = np.sqrt((l_vals.std()**2 + w_vals.std()**2) / 2)
            cohens_d = abs(diff) / pooled_std if pooled_std > 0 else 0

            separations.append((feat, cohens_d, l_mean, w_mean, l_med, w_med, diff))
            print(f"  {feat:<30} | {l_mean:>12.4f} | {w_mean:>12.4f} | {diff:>+10.4f} | {l_med:>12.4f} | {w_med:>12.4f} | d={cohens_d:>8.3f}")
        except Exception:
            pass

    # ── Top discriminating features ──
    separations.sort(key=lambda x: -x[1])
    print(f"\n  TOP DISCRIMINATING FEATURES (by Cohen's d effect size):")
    print(f"  {'Rank':>4} | {'Feature':<30} | {'Cohen d':>8} | {'Loser':>12} | {'Winner':>12} | {'Direction':<20}")
    print(f"  " + "-" * 100)
    for rank, (feat, d, lm, wm, lmed, wmed, diff) in enumerate(separations[:15], 1):
        direction = "LOSER HIGHER" if diff > 0 else "LOSER LOWER"
        stars = "***" if d >= 0.8 else "**" if d >= 0.5 else "*" if d >= 0.2 else ""
        print(f"  {rank:>4} | {feat:<30} | {d:>7.3f}{stars:>1} | {lm:>12.4f} | {wm:>12.4f} | {direction}")

    # ── Symbol breakdown ──
    print(f"\n  SYMBOL LOSS RATES:")
    print(f"  {'Symbol':<10} | {'Total':>6} | {'Losses':>6} | {'Loss Rate':>10} | {'Avg Loser PnL':>14} | {'Avg Winner PnL':>15}")
    print(f"  " + "-" * 75)
    for sym in sorted(df_trades["symbol"].unique()):
        sym_df = df_trades[df_trades["symbol"] == sym]
        sym_losses = sym_df[sym_df["is_loser"] == 1]
        sym_wins = sym_df[sym_df["is_loser"] == 0]
        loss_rate = len(sym_losses) / len(sym_df) * 100 if len(sym_df) > 0 else 0
        avg_loss_pnl = sym_losses["pnl"].mean() if len(sym_losses) > 0 else 0
        avg_win_pnl = sym_wins["pnl"].mean() if len(sym_wins) > 0 else 0
        print(f"  {sym:<10} | {len(sym_df):>6} | {len(sym_losses):>6} | {loss_rate:>8.1f}% | ${avg_loss_pnl:>12,.2f} | ${avg_win_pnl:>13,.2f}")

    # ── Direction breakdown ──
    print(f"\n  DIRECTION LOSS RATES:")
    for d in ["BUY", "SELL"]:
        d_df = df_trades[df_trades["direction"] == d]
        d_losses = d_df[d_df["is_loser"] == 1]
        loss_rate = len(d_losses) / len(d_df) * 100 if len(d_df) > 0 else 0
        print(f"  {d}: {len(d_df)} trades, {len(d_losses)} losses ({loss_rate:.1f}%)")

    # ── Session breakdown ──
    print(f"\n  SESSION LOSS RATES:")
    session_names = {0: "Asian (00-08 UTC)", 1: "London (08-16 UTC)", 2: "New York (16-24 UTC)"}
    for s in [0, 1, 2]:
        s_df = df_trades[df_trades["session"] == s]
        if len(s_df) == 0:
            continue
        s_losses = s_df[s_df["is_loser"] == 1]
        loss_rate = len(s_losses) / len(s_df) * 100
        print(f"  {session_names[s]}: {len(s_df)} trades, {len(s_losses)} losses ({loss_rate:.1f}%)")

    # ── Day of week breakdown ──
    print(f"\n  DAY OF WEEK LOSS RATES:")
    day_names = {0: "Monday", 1: "Tuesday", 2: "Wednesday", 3: "Thursday", 4: "Friday"}
    for d in range(5):
        d_df = df_trades[df_trades["day_of_week"] == d]
        if len(d_df) == 0:
            continue
        d_losses = d_df[d_df["is_loser"] == 1]
        loss_rate = len(d_losses) / len(d_df) * 100
        print(f"  {day_names[d]:>12}: {len(d_df):>4} trades, {len(d_losses):>3} losses ({loss_rate:.1f}%)")

    # ── Individual loser details ──
    print(f"\n  ALL {n_losers} INDIVIDUAL LOSERS:")
    print(f"  {'#':>3} | {'Symbol':<8} | {'Dir':<4} | {'Exit':>10} | {'PnL':>10} | {'MFE ATR':>8} | {'Bars':>5} | {'ATR vs Med':>10} | {'BarRange/ATR':>12} | {'SigBarAgree':>11} | {'Mom3bar':>8} | {'Day':>3} | {'Hour':>4}")
    print(f"  " + "-" * 130)
    for idx, row in losers.iterrows():
        print(f"  {idx:>3} | {row['symbol']:<8} | {row['direction']:<4} | {row['exit_type']:>10} | ${row['pnl']:>8,.2f} | {row['mfe_atr']:>7.3f} | {row['bars_in_trade']:>5} | {row.get('atr_vs_median', 0):>10.3f} | {row.get('bar_range_vs_atr', 0):>12.3f} | {row.get('signal_bar_agrees', 0):>11} | {row.get('momentum_3bar_atr', 0):>+7.3f} | {row.get('day_of_week', 0):>3} | {row.get('hour', 0):>4}")

    # ═══════════════════════════════════════════════════════════════
    # FILTER DISCOVERY: Can we find a filter that removes losers
    # without removing too many winners?
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'=' * 100}")
    print(f"  FILTER DISCOVERY: Testing potential loser filters")
    print(f"{'=' * 100}")

    # Test single-feature filters
    print(f"\n  {'Filter':<55} | {'Losers Removed':>14} | {'Winners Removed':>15} | {'Net Trades':>10} | {'New WR':>8} | {'Worth it?':>10}")
    print(f"  " + "-" * 120)

    filters_tested = []

    # Test various thresholds for each discriminating feature
    for feat, d_val, lm, wm, lmed, wmed, diff in separations[:10]:
        if d_val < 0.15:
            continue

        # Try multiple thresholds
        for pct in [10, 25, 50, 75, 90]:
            try:
                threshold = np.percentile(losers[feat].astype(float), pct)

                if diff > 0:  # losers have HIGHER values → filter out high values
                    mask = df_trades[feat].astype(float) <= threshold
                    direction = f"{feat} <= {threshold:.4f} (P{pct} of losers)"
                else:  # losers have LOWER values → filter out low values
                    mask = df_trades[feat].astype(float) >= threshold
                    direction = f"{feat} >= {threshold:.4f} (P{pct} of losers)"

                filtered = df_trades[mask]
                f_losers = filtered[filtered["is_loser"] == 1]
                f_winners = filtered[filtered["is_loser"] == 0]
                losers_removed = n_losers - len(f_losers)
                winners_removed = n_winners - len(f_winners)
                new_wr = len(f_winners) / len(filtered) * 100 if len(filtered) > 0 else 0

                # Worth it if we remove more losers (proportionally) than winners
                loser_pct_removed = losers_removed / n_losers * 100 if n_losers > 0 else 0
                winner_pct_removed = winners_removed / n_winners * 100 if n_winners > 0 else 0
                worth = "YES" if loser_pct_removed > winner_pct_removed * 2 else "maybe" if loser_pct_removed > winner_pct_removed else "NO"

                if losers_removed > 0:
                    filters_tested.append((direction, losers_removed, winners_removed, len(filtered), new_wr, worth, loser_pct_removed, winner_pct_removed))
                    print(f"  {direction:<55} | {losers_removed:>3}/{n_losers} ({loser_pct_removed:>4.0f}%) | {winners_removed:>4}/{n_winners} ({winner_pct_removed:>4.1f}%) | {len(filtered):>10} | {new_wr:>6.1f}% | {worth:>10}")
            except Exception:
                pass

    # ── Best combined filters ──
    print(f"\n  BEST FILTERS (remove most losers, fewest winners):")
    filters_tested.sort(key=lambda x: (-x[1], x[2]))  # most losers removed, least winners removed
    for i, (desc, lr, wr, nt, nwr, worth, lpct, wpct) in enumerate(filters_tested[:10], 1):
        ratio = lpct / wpct if wpct > 0 else float('inf')
        print(f"  {i:>2}. {desc}")
        print(f"      Removes {lr} losers ({lpct:.0f}%), {wr} winners ({wpct:.1f}%) | Selectivity ratio: {ratio:.1f}x | New WR: {nwr:.1f}%")

    print(f"\n{'=' * 100}")


if __name__ == "__main__":
    main()
