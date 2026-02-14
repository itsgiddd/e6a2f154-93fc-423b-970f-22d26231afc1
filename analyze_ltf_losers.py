#!/usr/bin/env python3
"""
LOWER-TIMEFRAME LOSER PROFILER — V4 ZeroPoint
═══════════════════════════════════════════════
At each H4 signal, pull M15 and H1 data around that moment.
Extract microstructure features from lower timeframes.
Find a SINGLE distinct feature that cleanly separates the 2.5% losers
WITHOUT losing significant profit from winners.

Key idea: The H4 signal is correct 97.5% of the time. The 2.5% losers
are "dead on arrival" — price never moves 0.5x ATR favorably.
Something visible on M15/H1 at the signal moment should reveal this.
"""

import sys, os
import numpy as np
import pandas as pd
from collections import defaultdict
from datetime import datetime, timedelta

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
FETCH_BARS_H4 = 5000
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
# V4 Position (same as analyze_losers.py / backtest)
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

        if self.tp2_hit and not self.tp3_hit:
            if (is_buy and high >= self.tp3) or (not is_buy and low <= self.tp3):
                self.tp3_hit = True
                self.partials.append(self.pnl_for_price(self.tp3, self.remaining_lot))
                self.remaining_lot = 0
                self.closed = True
                self.exit_type = 'TP3'
                return

        if self.profit_lock_active and self.profit_lock_sl is not None:
            lock_hit = (is_buy and low <= self.profit_lock_sl) or (not is_buy and high >= self.profit_lock_sl)
            if lock_hit:
                self.partials.append(self.pnl_for_price(self.profit_lock_sl, self.remaining_lot))
                self.remaining_lot = 0
                self.closed = True
                self.exit_type = 'PROFIT_LOCK'
                return

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
# LOWER-TIMEFRAME FEATURE EXTRACTION
# ═══════════════════════════════════════════════════════════════

def compute_rsi(closes, period=14):
    """Compute RSI from a series of closes."""
    if len(closes) < period + 1:
        return 50.0  # neutral default
    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    # Wilder's smoothing
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])
    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def compute_atr_series(df_ltf, period=14):
    """Compute ATR using Wilder's RMA."""
    highs = df_ltf["high"].values.astype(float)
    lows = df_ltf["low"].values.astype(float)
    closes = df_ltf["close"].values.astype(float)
    tr = np.zeros(len(df_ltf))
    tr[0] = highs[0] - lows[0]
    for i in range(1, len(df_ltf)):
        tr[i] = max(highs[i] - lows[i], abs(highs[i] - closes[i-1]), abs(lows[i] - closes[i-1]))
    # Wilder's RMA
    alpha = 1.0 / period
    atr = np.zeros(len(df_ltf))
    atr[0] = tr[0]
    for i in range(1, len(df_ltf)):
        atr[i] = alpha * tr[i] + (1 - alpha) * atr[i-1]
    return atr


def extract_ltf_features(df_ltf, h4_time, h4_atr, direction, timeframe_label):
    """
    Extract features from lower-timeframe data at the moment of H4 signal.

    df_ltf: M15 or H1 DataFrame
    h4_time: timestamp of H4 signal bar
    h4_atr: H4 ATR value (for normalizing LTF moves)
    direction: "BUY" or "SELL"
    timeframe_label: "M15" or "H1"

    Returns dict of features or None if not enough data.
    """
    prefix = timeframe_label.lower()
    features = {}

    # Find bars BEFORE the H4 signal time (no look-ahead)
    # H4 bar at time T covers T to T+4h. LTF bars before T are safe.
    mask = df_ltf["time"] < h4_time
    if mask.sum() < 30:
        return None  # not enough data

    df_before = df_ltf[mask].copy()

    closes = df_before["close"].values.astype(float)
    highs = df_before["high"].values.astype(float)
    lows = df_before["low"].values.astype(float)
    opens = df_before["open"].values.astype(float)
    vols = df_before["volume"].values.astype(float) if "volume" in df_before else np.ones(len(df_before))

    cur_close = closes[-1]

    # ── 1. RSI at signal time ──
    rsi = compute_rsi(closes[-30:], period=14)
    features[f"{prefix}_rsi"] = rsi

    # RSI extreme (overbought BUY or oversold SELL = counter-signal)
    if direction == "BUY":
        features[f"{prefix}_rsi_extreme"] = 1 if rsi > 70 else (0 if rsi < 30 else 0.5)
        # Counter: buying when RSI says overbought
        features[f"{prefix}_rsi_counter"] = max(0, (rsi - 50) / 50)  # 0=neutral, 1=max overbought
    else:
        features[f"{prefix}_rsi_extreme"] = 1 if rsi < 30 else (0 if rsi > 70 else 0.5)
        features[f"{prefix}_rsi_counter"] = max(0, (50 - rsi) / 50)

    # ── 2. Momentum (price change over last N bars, normalized to H4 ATR) ──
    for n_bars in [4, 8, 16]:
        if len(closes) > n_bars:
            move = (closes[-1] - closes[-1 - n_bars])
            if direction == "SELL":
                move = -move
            features[f"{prefix}_momentum_{n_bars}bar"] = move / h4_atr if h4_atr > 0 else 0
        else:
            features[f"{prefix}_momentum_{n_bars}bar"] = 0

    # ── 3. Candle pattern on last LTF bar ──
    last_bar_range = highs[-1] - lows[-1]
    last_bar_body = abs(closes[-1] - opens[-1])
    features[f"{prefix}_last_bar_body_ratio"] = last_bar_body / last_bar_range if last_bar_range > 0 else 0

    # Is last LTF bar agreeing with H4 signal direction?
    last_bullish = closes[-1] > opens[-1]
    features[f"{prefix}_last_bar_agrees"] = 1 if ((direction == "BUY" and last_bullish) or
                                                   (direction == "SELL" and not last_bullish)) else 0

    # ── 4. Consecutive bars in signal direction (LTF momentum) ──
    consec = 0
    for k in range(len(closes) - 1, max(len(closes) - 20, 0), -1):
        bar_bull = closes[k] > opens[k]
        if (direction == "BUY" and bar_bull) or (direction == "SELL" and not bar_bull):
            consec += 1
        else:
            break
    features[f"{prefix}_consec_agree_bars"] = consec

    # ── 5. Consecutive bars AGAINST signal direction (counter-momentum) ──
    consec_against = 0
    for k in range(len(closes) - 1, max(len(closes) - 20, 0), -1):
        bar_bull = closes[k] > opens[k]
        if (direction == "BUY" and not bar_bull) or (direction == "SELL" and bar_bull):
            consec_against += 1
        else:
            break
    features[f"{prefix}_consec_against_bars"] = consec_against

    # ── 6. Volume spike at signal time ──
    if len(vols) >= 10:
        vol_avg = np.mean(vols[-10:])
        vol_last = vols[-1]
        features[f"{prefix}_vol_spike"] = vol_last / vol_avg if vol_avg > 0 else 1.0
    else:
        features[f"{prefix}_vol_spike"] = 1.0

    # Volume trend (increasing = conviction)
    if len(vols) >= 6:
        vol_first_half = np.mean(vols[-6:-3])
        vol_second_half = np.mean(vols[-3:])
        features[f"{prefix}_vol_trend"] = vol_second_half / vol_first_half if vol_first_half > 0 else 1.0
    else:
        features[f"{prefix}_vol_trend"] = 1.0

    # ── 7. ATR on LTF (volatility regime at signal time) ──
    atr_ltf = compute_atr_series(df_before.iloc[-30:], period=14)
    if len(atr_ltf) >= 2:
        features[f"{prefix}_atr_current"] = atr_ltf[-1]
        features[f"{prefix}_atr_vs_h4"] = atr_ltf[-1] / h4_atr if h4_atr > 0 else 0
        # ATR expanding or contracting on LTF?
        if len(atr_ltf) >= 10:
            atr_old = np.mean(atr_ltf[:5])
            atr_new = np.mean(atr_ltf[-5:])
            features[f"{prefix}_atr_expansion"] = atr_new / atr_old if atr_old > 0 else 1.0
        else:
            features[f"{prefix}_atr_expansion"] = 1.0
    else:
        features[f"{prefix}_atr_current"] = 0
        features[f"{prefix}_atr_vs_h4"] = 0
        features[f"{prefix}_atr_expansion"] = 1.0

    # ── 8. Price position on LTF range (are we at a micro S/R?) ──
    if len(closes) >= 20:
        h20 = np.max(highs[-20:])
        l20 = np.min(lows[-20:])
        rng = h20 - l20
        if rng > 0:
            features[f"{prefix}_price_pos_20"] = (cur_close - l20) / rng
        else:
            features[f"{prefix}_price_pos_20"] = 0.5
    else:
        features[f"{prefix}_price_pos_20"] = 0.5

    # ── 9. Wicks analysis — are recent LTF bars showing rejection? ──
    # Average upper/lower wick ratio of last 4 bars
    if len(closes) >= 4:
        upper_wicks = []
        lower_wicks = []
        for k in range(-4, 0):
            br = highs[k] - lows[k]
            if br > 0:
                if closes[k] >= opens[k]:  # bullish
                    upper_wicks.append((highs[k] - closes[k]) / br)
                    lower_wicks.append((opens[k] - lows[k]) / br)
                else:  # bearish
                    upper_wicks.append((highs[k] - opens[k]) / br)
                    lower_wicks.append((closes[k] - lows[k]) / br)
            else:
                upper_wicks.append(0)
                lower_wicks.append(0)
        features[f"{prefix}_avg_upper_wick"] = np.mean(upper_wicks)
        features[f"{prefix}_avg_lower_wick"] = np.mean(lower_wicks)

        # Rejection wicks: for BUY, upper wicks = sellers pushing back
        if direction == "BUY":
            features[f"{prefix}_rejection_wick"] = np.mean(upper_wicks)  # higher = more selling pressure
        else:
            features[f"{prefix}_rejection_wick"] = np.mean(lower_wicks)  # higher = more buying pressure
    else:
        features[f"{prefix}_avg_upper_wick"] = 0
        features[f"{prefix}_avg_lower_wick"] = 0
        features[f"{prefix}_rejection_wick"] = 0

    # ── 10. Engulfing / doji on last bar ──
    if len(closes) >= 2:
        prev_body = abs(closes[-2] - opens[-2])
        cur_body = abs(closes[-1] - opens[-1])
        features[f"{prefix}_engulfing"] = 1 if cur_body > prev_body * 1.5 else 0
        features[f"{prefix}_doji"] = 1 if (last_bar_range > 0 and last_bar_body / last_bar_range < 0.1) else 0
    else:
        features[f"{prefix}_engulfing"] = 0
        features[f"{prefix}_doji"] = 0

    # ── 11. Price distance from LTF EMAs ──
    if len(closes) >= 20:
        ema_fast = pd.Series(closes).ewm(span=8, adjust=False).mean().iloc[-1]
        ema_slow = pd.Series(closes).ewm(span=20, adjust=False).mean().iloc[-1]
        features[f"{prefix}_price_vs_ema8"] = (cur_close - ema_fast) / h4_atr if h4_atr > 0 else 0
        features[f"{prefix}_price_vs_ema20"] = (cur_close - ema_slow) / h4_atr if h4_atr > 0 else 0
        features[f"{prefix}_ema_cross"] = 1 if ema_fast > ema_slow else 0  # fast above slow

        # For signal alignment
        if direction == "BUY":
            features[f"{prefix}_ema_agrees"] = 1 if ema_fast > ema_slow else 0
        else:
            features[f"{prefix}_ema_agrees"] = 1 if ema_fast < ema_slow else 0
    else:
        features[f"{prefix}_price_vs_ema8"] = 0
        features[f"{prefix}_price_vs_ema20"] = 0
        features[f"{prefix}_ema_cross"] = 0.5
        features[f"{prefix}_ema_agrees"] = 0.5

    # ── 12. Spike-then-retrace detection (false breakout) ──
    # Did price spike in signal direction then retrace in last few LTF bars?
    if len(closes) >= 8:
        # Max favorable in last 8 bars
        if direction == "BUY":
            peak = np.max(highs[-8:])
            spike_size = (peak - closes[-1]) / h4_atr if h4_atr > 0 else 0
        else:
            trough = np.min(lows[-8:])
            spike_size = (closes[-1] - trough) / h4_atr if h4_atr > 0 else 0
        features[f"{prefix}_retrace_from_peak"] = spike_size  # high = price spiked then pulled back
    else:
        features[f"{prefix}_retrace_from_peak"] = 0

    # ── 13. Divergence: price making new high/low but momentum weakening ──
    if len(closes) >= 16:
        # Compare first 8-bar block vs second 8-bar block
        first_move = abs(closes[-16] - closes[-8])
        second_move = abs(closes[-8] - closes[-1])
        if first_move > 0:
            features[f"{prefix}_momentum_ratio"] = second_move / first_move
        else:
            features[f"{prefix}_momentum_ratio"] = 1.0
    else:
        features[f"{prefix}_momentum_ratio"] = 1.0

    # ── 14. Bollinger Band position (squeeze = low vol -> breakout potential) ──
    if len(closes) >= 20:
        sma20 = np.mean(closes[-20:])
        std20 = np.std(closes[-20:])
        upper_bb = sma20 + 2 * std20
        lower_bb = sma20 - 2 * std20
        bb_width = (upper_bb - lower_bb) / h4_atr if h4_atr > 0 else 0
        features[f"{prefix}_bb_width"] = bb_width  # narrow = squeeze

        if upper_bb != lower_bb:
            features[f"{prefix}_bb_position"] = (cur_close - lower_bb) / (upper_bb - lower_bb)
        else:
            features[f"{prefix}_bb_position"] = 0.5
    else:
        features[f"{prefix}_bb_width"] = 0
        features[f"{prefix}_bb_position"] = 0.5

    # ── 15. Recent high/low break (is LTF confirming the H4 breakout?) ──
    if len(closes) >= 16:
        prev_high_16 = np.max(highs[-16:-4])  # High from 16 to 4 bars ago
        prev_low_16 = np.min(lows[-16:-4])
        recent_high_4 = np.max(highs[-4:])     # High of last 4 bars
        recent_low_4 = np.min(lows[-4:])

        if direction == "BUY":
            features[f"{prefix}_breakout_confirmed"] = 1 if recent_high_4 > prev_high_16 else 0
            features[f"{prefix}_breakdown_counter"] = 1 if recent_low_4 < prev_low_16 else 0
        else:
            features[f"{prefix}_breakout_confirmed"] = 1 if recent_low_4 < prev_low_16 else 0
            features[f"{prefix}_breakdown_counter"] = 1 if recent_high_4 > prev_high_16 else 0
    else:
        features[f"{prefix}_breakout_confirmed"] = 0
        features[f"{prefix}_breakdown_counter"] = 0

    # ── 16. Speed of approach: how fast was price moving INTO the signal? ──
    if len(closes) >= 4:
        speed = abs(closes[-1] - closes[-4]) / 3  # avg per-bar move
        speed_norm = speed / h4_atr if h4_atr > 0 else 0
        features[f"{prefix}_approach_speed"] = speed_norm
    else:
        features[f"{prefix}_approach_speed"] = 0

    return features


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    print("=" * 110)
    print("  LOWER-TIMEFRAME LOSER PROFILER — V4 ZeroPoint")
    print("  Analyzing M15 + H1 microstructure at the moment of each H4 signal")
    print("=" * 110)

    if not mt5.initialize():
        print("ERROR: Could not initialize MT5")
        return

    # ── Load H4 data ──
    h4_data = {}
    print("\n[1/3] Loading H4 data + ZeroPoint signals...")
    for sym in SYMBOLS:
        resolved = resolve_symbol(sym)
        if resolved is None:
            print(f"  {sym}: SKIP (not found)")
            continue
        rates = mt5.copy_rates_from_pos(resolved, mt5.TIMEFRAME_H4, 0, FETCH_BARS_H4)
        if rates is None or len(rates) < 100:
            continue
        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        df.rename(columns={"tick_volume": "volume"}, inplace=True)
        df_zp = compute_zeropoint_state(df)
        if df_zp is not None and len(df_zp) >= WARMUP_BARS:
            df_zp.attrs["symbol"] = sym
            h4_data[sym] = (df_zp, resolved)
            print(f"  {sym}: {len(df_zp)} H4 bars")

    # ── Load M15 + H1 data ──
    print("\n[2/3] Loading M15 + H1 data...")
    ltf_data = {}  # {sym: {"M15": df, "H1": df}}
    for sym in SYMBOLS:
        if sym not in h4_data:
            continue
        _, resolved = h4_data[sym]
        ltf_data[sym] = {}

        # M15 — fetch more bars since 1 H4 bar = 16 M15 bars
        rates_m15 = mt5.copy_rates_from_pos(resolved, mt5.TIMEFRAME_M15, 0, min(FETCH_BARS_H4 * 16, 50000))
        if rates_m15 is not None and len(rates_m15) >= 100:
            df_m15 = pd.DataFrame(rates_m15)
            df_m15["time"] = pd.to_datetime(df_m15["time"], unit="s")
            df_m15.rename(columns={"tick_volume": "volume"}, inplace=True)
            ltf_data[sym]["M15"] = df_m15
            print(f"  {sym} M15: {len(df_m15)} bars")

        # H1 — fetch more bars since 1 H4 bar = 4 H1 bars
        rates_h1 = mt5.copy_rates_from_pos(resolved, mt5.TIMEFRAME_H1, 0, min(FETCH_BARS_H4 * 4, 20000))
        if rates_h1 is not None and len(rates_h1) >= 100:
            df_h1 = pd.DataFrame(rates_h1)
            df_h1["time"] = pd.to_datetime(df_h1["time"], unit="s")
            df_h1.rename(columns={"tick_volume": "volume"}, inplace=True)
            ltf_data[sym]["H1"] = df_h1
            print(f"  {sym} H1:  {len(df_h1)} bars")

    mt5.shutdown()

    if not h4_data:
        print("No data!")
        return

    # ── Run H4 simulation + extract LTF features at each signal ──
    print("\n[3/3] Simulating V4 trades + extracting LTF features at each signal...")

    trade_records = []

    for sym, (df, resolved) in h4_data.items():
        cont_sz = contract_size(sym)
        n = len(df)
        pos_obj = None
        sym_ltf = ltf_data.get(sym, {})

        def record_trade(pos):
            """Record a completed trade with its LTF features."""
            rec = {
                "symbol": pos.sym,
                "direction": pos.direction,
                "entry_price": pos.entry,
                "entry_time": pos.entry_time,
                "exit_time": pos.exit_time,
                "pnl": pos.total_pnl,
                "exit_type": pos.exit_type,
                "bars_in_trade": pos.bars_in_trade,
                "mfe_atr": pos.max_profit_reached / pos.atr_val if pos.atr_val > 0 else 0,
                "outcome": "WIN" if pos.total_pnl > 0 else "LOSS",
                "is_loser": 1 if pos.total_pnl <= 0 else 0,
                "h4_atr": pos.atr_val,
            }
            # Add LTF features that were captured at entry
            if hasattr(pos, '_ltf_features') and pos._ltf_features:
                rec.update(pos._ltf_features)
            trade_records.append(rec)

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
                    record_trade(pos_obj)
                    pos_obj = None

            # Open new position on signal
            if buy_sig or sell_sig:
                direction = "BUY" if buy_sig else "SELL"

                # Force close existing
                if pos_obj is not None and not pos_obj.closed:
                    pos_obj.force_close(close)
                    pos_obj.exit_time = row["time"]
                    record_trade(pos_obj)

                smart_sl = compute_smart_sl(df, i, direction, atr_val)
                pos_obj = V4Position(
                    sym=sym, direction=direction, entry=close, sl=smart_sl,
                    atr_val=atr_val, lot=LOT_SIZE, cont_sz=cont_sz, entry_time=row["time"],
                )

                # ── EXTRACT LTF FEATURES ──
                h4_time = row["time"]
                ltf_features = {}

                for tf_key, tf_label in [("M15", "M15"), ("H1", "H1")]:
                    if tf_key in sym_ltf:
                        feats = extract_ltf_features(
                            sym_ltf[tf_key], h4_time, atr_val, direction, tf_label
                        )
                        if feats:
                            ltf_features.update(feats)

                pos_obj._ltf_features = ltf_features

        # Close remaining
        if pos_obj is not None and not pos_obj.closed:
            pos_obj.force_close(float(df.iloc[-1]["close"]))
            pos_obj.exit_time = df.iloc[-1]["time"]
            record_trade(pos_obj)

    # ═══════════════════════════════════════════════════════════════
    # ANALYSIS
    # ═══════════════════════════════════════════════════════════════
    df_trades = pd.DataFrame(trade_records)
    total = len(df_trades)
    losers = df_trades[df_trades["is_loser"] == 1]
    winners = df_trades[df_trades["is_loser"] == 0]

    print(f"\n{'='*110}")
    print(f"  TRADE SUMMARY")
    print(f"{'='*110}")
    print(f"  Total trades:   {total}")
    print(f"  Winners:        {len(winners)} ({100*len(winners)/total:.1f}%)")
    print(f"  Losers:         {len(losers)} ({100*len(losers)/total:.1f}%)")
    print(f"  Net PnL:        ${df_trades['pnl'].sum():,.0f}")
    print(f"  Gross Loss:     ${losers['pnl'].sum():,.0f}")

    # Get LTF feature columns
    ltf_cols = [c for c in df_trades.columns if c.startswith("m15_") or c.startswith("h1_")]
    if not ltf_cols:
        print("\nERROR: No LTF features extracted! Check that M15/H1 data covers the H4 signal period.")
        return

    print(f"\n  LTF features extracted: {len(ltf_cols)}")

    # ── STATISTICAL COMPARISON: Losers vs Winners ──
    print(f"\n{'='*110}")
    print(f"  LOSER vs WINNER — Lower Timeframe Feature Comparison")
    print(f"  Cohen's d: effect size (|d| > 0.5 = medium, |d| > 0.8 = large)")
    print(f"{'='*110}")
    print(f"  {'Feature':<40} {'Loser Mean':>12} {'Winner Mean':>12} {'Cohen d':>10} {'Direction':>12}")
    print(f"  {'-'*40} {'-'*12} {'-'*12} {'-'*10} {'-'*12}")

    results = []
    for col in sorted(ltf_cols):
        l_vals = losers[col].dropna()
        w_vals = winners[col].dropna()
        if len(l_vals) < 3 or len(w_vals) < 10:
            continue
        l_mean = l_vals.mean()
        w_mean = w_vals.mean()
        pooled_std = np.sqrt(((len(l_vals)-1)*l_vals.std()**2 + (len(w_vals)-1)*w_vals.std()**2) /
                             (len(l_vals) + len(w_vals) - 2))
        if pooled_std > 0:
            d = (l_mean - w_mean) / pooled_std
        else:
            d = 0

        # Direction: higher on losers means "loser-like"
        if abs(d) > 0.1:
            direction = "LOSER HIGHER" if d > 0 else "LOSER LOWER"
        else:
            direction = "~same"

        results.append({
            "feature": col,
            "loser_mean": l_mean,
            "winner_mean": w_mean,
            "cohen_d": d,
            "abs_d": abs(d),
            "direction": direction,
        })

    results.sort(key=lambda x: x["abs_d"], reverse=True)

    for r in results:
        marker = "***" if r["abs_d"] >= 0.5 else "** " if r["abs_d"] >= 0.3 else "   "
        print(f"  {marker}{r['feature']:<37} {r['loser_mean']:>12.4f} {r['winner_mean']:>12.4f} {r['cohen_d']:>10.3f} {r['direction']:>12}")

    # ═══════════════════════════════════════════════════════════════
    # FILTER DISCOVERY — Test each promising feature as a binary filter
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*110}")
    print(f"  FILTER DISCOVERY — Testing each LTF feature as a trade/skip filter")
    print(f"  Goal: Skip losers WITHOUT losing significant winner profit")
    print(f"{'='*110}")

    baseline_pnl = df_trades["pnl"].sum()
    baseline_losses = losers["pnl"].sum()
    total_losers = len(losers)

    # Only test features with |d| >= 0.3 (at least small-medium effect)
    promising = [r for r in results if r["abs_d"] >= 0.25]

    if not promising:
        print("\n  No features with |d| >= 0.25 found. The 2.5% losers are not distinguishable on LTF.")
        print("  This means the losers look identical to winners at entry time on M15/H1.")
        return

    print(f"\n  Testing {len(promising)} promising features (|d| >= 0.25)...")
    print(f"  Baseline: {total} trades, PnL ${baseline_pnl:,.0f}, Losses ${baseline_losses:,.0f}\n")

    best_filters = []

    for r in promising:
        col = r["feature"]
        l_vals = losers[col].dropna()
        w_vals = winners[col].dropna()

        # Test multiple thresholds
        all_vals = pd.concat([l_vals, w_vals])
        percentiles = [10, 15, 20, 25, 30, 70, 75, 80, 85, 90]

        for pct in percentiles:
            threshold = np.percentile(all_vals, pct)

            # Test: skip trades where feature < threshold
            for op, op_name in [("lt", "<"), ("gt", ">")]:
                if op == "lt":
                    skipped = df_trades[df_trades[col] < threshold]
                    kept = df_trades[df_trades[col] >= threshold]
                else:
                    skipped = df_trades[df_trades[col] > threshold]
                    kept = df_trades[df_trades[col] <= threshold]

                if len(skipped) == 0 or len(kept) == 0:
                    continue

                losers_skipped = len(skipped[skipped["is_loser"] == 1])
                winners_skipped = len(skipped[skipped["is_loser"] == 0])
                total_skipped = len(skipped)

                kept_pnl = kept["pnl"].sum()
                pnl_change = kept_pnl - baseline_pnl
                pnl_change_pct = 100 * pnl_change / baseline_pnl if baseline_pnl != 0 else 0

                loss_reduction = losers_skipped / total_losers * 100 if total_losers > 0 else 0

                # Calculate "efficiency": losers skipped per winner lost
                if winners_skipped > 0:
                    efficiency = losers_skipped / winners_skipped
                else:
                    efficiency = float('inf') if losers_skipped > 0 else 0

                # We want: high loser skip rate, low winner collateral, minimal profit loss
                # Filter quality = (% losers caught) * (1 - % profit lost)
                # A perfect filter catches 100% losers with 0% profit cost
                if losers_skipped >= 2 and pnl_change_pct > -5:  # at least 2 losers caught, <5% profit loss
                    quality = (loss_reduction / 100) * (1 + pnl_change_pct / 100)

                    kept_losers = len(kept[kept["is_loser"] == 1])
                    kept_winners = len(kept[kept["is_loser"] == 0])
                    new_wr = 100 * kept_winners / len(kept) if len(kept) > 0 else 0

                    best_filters.append({
                        "feature": col,
                        "op": op_name,
                        "threshold": threshold,
                        "pct": pct,
                        "losers_skipped": losers_skipped,
                        "winners_skipped": winners_skipped,
                        "total_skipped": total_skipped,
                        "kept_trades": len(kept),
                        "kept_pnl": kept_pnl,
                        "pnl_change": pnl_change,
                        "pnl_change_pct": pnl_change_pct,
                        "loss_reduction": loss_reduction,
                        "efficiency": efficiency,
                        "quality": quality,
                        "new_wr": new_wr,
                    })

    if not best_filters:
        print("  No filters found that catch >= 2 losers with < 5% profit cost.")
        print("  The losers are indistinguishable from winners on lower timeframes.")

        # Still show top features
        print(f"\n  Top 10 distinguishing features (for reference):")
        for i, r in enumerate(results[:10]):
            print(f"    {i+1}. {r['feature']}: Cohen's d = {r['cohen_d']:.3f} ({r['direction']})")
        return

    # Sort by quality (high = better)
    best_filters.sort(key=lambda x: x["quality"], reverse=True)

    print(f"\n  TOP 20 FILTERS (sorted by quality = loser_catch * profit_retention):")
    print(f"  {'Feature':<35} {'Rule':<15} {'Skip':>5} {'L_Skip':>6} {'W_Skip':>6} {'PnL Chg':>10} {'%Chg':>6} {'LossCut':>8} {'WR':>6} {'Quality':>8}")
    print(f"  {'-'*35} {'-'*15} {'-'*5} {'-'*6} {'-'*6} {'-'*10} {'-'*6} {'-'*8} {'-'*6} {'-'*8}")

    for f in best_filters[:20]:
        rule = f"{f['op']} {f['threshold']:.4f}"
        print(f"  {f['feature']:<35} {rule:<15} {f['total_skipped']:>5} {f['losers_skipped']:>6} {f['winners_skipped']:>6} ${f['pnl_change']:>9,.0f} {f['pnl_change_pct']:>5.1f}% {f['loss_reduction']:>7.1f}% {f['new_wr']:>5.1f}% {f['quality']:>8.3f}")

    # ═══════════════════════════════════════════════════════════════
    # DEEP DIVE on best filter
    # ═══════════════════════════════════════════════════════════════
    best = best_filters[0]
    print(f"\n{'='*110}")
    print(f"  BEST FILTER DEEP DIVE")
    print(f"{'='*110}")
    print(f"  Feature:          {best['feature']}")
    print(f"  Rule:             Skip if {best['feature']} {best['op']} {best['threshold']:.4f}")
    print(f"  Losers caught:    {best['losers_skipped']}/{total_losers} ({best['loss_reduction']:.1f}%)")
    print(f"  Winners lost:     {best['winners_skipped']}")
    print(f"  PnL change:       ${best['pnl_change']:,.0f} ({best['pnl_change_pct']:.1f}%)")
    print(f"  New win rate:     {best['new_wr']:.1f}%")
    print(f"  Quality score:    {best['quality']:.3f}")

    # Show the losers that would still remain
    if best["op"] == "<":
        remaining_trades = df_trades[df_trades[best['feature']] >= best['threshold']]
    else:
        remaining_trades = df_trades[df_trades[best['feature']] <= best['threshold']]

    remaining_losers = remaining_trades[remaining_trades["is_loser"] == 1]
    if len(remaining_losers) > 0:
        print(f"\n  Remaining losers ({len(remaining_losers)}):")
        for _, rl in remaining_losers.iterrows():
            print(f"    {rl['symbol']} {rl['direction']} {rl['entry_time']} "
                  f"PnL=${rl['pnl']:,.0f} MFE={rl['mfe_atr']:.3f}x ATR "
                  f"{best['feature']}={rl.get(best['feature'], 'N/A')}")

    # Show the caught losers
    if best["op"] == "<":
        caught = df_trades[(df_trades[best['feature']] < best['threshold']) & (df_trades["is_loser"] == 1)]
    else:
        caught = df_trades[(df_trades[best['feature']] > best['threshold']) & (df_trades["is_loser"] == 1)]

    if len(caught) > 0:
        print(f"\n  Caught losers ({len(caught)}):")
        for _, cl in caught.iterrows():
            print(f"    {cl['symbol']} {cl['direction']} {cl['entry_time']} "
                  f"PnL=${cl['pnl']:,.0f} MFE={cl['mfe_atr']:.3f}x ATR "
                  f"{best['feature']}={cl.get(best['feature'], 'N/A')}")

    # ═══════════════════════════════════════════════════════════════
    # COMBINATION FILTER TEST
    # ═══════════════════════════════════════════════════════════════
    # Test combining top 2-3 features for an even cleaner filter
    if len(best_filters) >= 2:
        print(f"\n{'='*110}")
        print(f"  COMBINATION FILTER TEST — Top 2-3 features combined")
        print(f"{'='*110}")

        # Get unique top features (different features, not same feature with different thresholds)
        seen_features = set()
        unique_top = []
        for f in best_filters:
            if f["feature"] not in seen_features:
                seen_features.add(f["feature"])
                unique_top.append(f)
                if len(unique_top) >= 5:
                    break

        # Test pairwise combinations
        for i in range(len(unique_top)):
            for j in range(i + 1, len(unique_top)):
                f1 = unique_top[i]
                f2 = unique_top[j]

                # Skip if feature is OR: trade is skipped if EITHER condition met
                mask1 = df_trades[f1["feature"]] < f1["threshold"] if f1["op"] == "<" else df_trades[f1["feature"]] > f1["threshold"]
                mask2 = df_trades[f2["feature"]] < f2["threshold"] if f2["op"] == "<" else df_trades[f2["feature"]] > f2["threshold"]

                # OR combination (skip if either triggers)
                skip_or = mask1 | mask2
                kept_or = df_trades[~skip_or]
                if len(kept_or) > 0:
                    or_losers_caught = len(df_trades[skip_or & (df_trades["is_loser"] == 1)])
                    or_winners_lost = len(df_trades[skip_or & (df_trades["is_loser"] == 0)])
                    or_pnl = kept_or["pnl"].sum()
                    or_pnl_chg = 100 * (or_pnl - baseline_pnl) / baseline_pnl if baseline_pnl != 0 else 0
                    or_wr = 100 * len(kept_or[kept_or["is_loser"] == 0]) / len(kept_or)

                    if or_losers_caught >= 2 and or_pnl_chg > -5:
                        print(f"\n  OR: ({f1['feature']} {f1['op']} {f1['threshold']:.4f}) OR ({f2['feature']} {f2['op']} {f2['threshold']:.4f})")
                        print(f"       Losers caught: {or_losers_caught}/{total_losers} | Winners lost: {or_winners_lost} | PnL: ${or_pnl:,.0f} ({or_pnl_chg:+.1f}%) | WR: {or_wr:.1f}%")

                # AND combination (skip only if BOTH trigger)
                skip_and = mask1 & mask2
                kept_and = df_trades[~skip_and]
                if len(kept_and) > 0:
                    and_losers_caught = len(df_trades[skip_and & (df_trades["is_loser"] == 1)])
                    and_winners_lost = len(df_trades[skip_and & (df_trades["is_loser"] == 0)])
                    and_pnl = kept_and["pnl"].sum()
                    and_pnl_chg = 100 * (and_pnl - baseline_pnl) / baseline_pnl if baseline_pnl != 0 else 0
                    and_wr = 100 * len(kept_and[kept_and["is_loser"] == 0]) / len(kept_and)

                    if and_losers_caught >= 2 and and_pnl_chg > -3:
                        print(f"\n  AND: ({f1['feature']} {f1['op']} {f1['threshold']:.4f}) AND ({f2['feature']} {f2['op']} {f2['threshold']:.4f})")
                        print(f"       Losers caught: {and_losers_caught}/{total_losers} | Winners lost: {and_winners_lost} | PnL: ${and_pnl:,.0f} ({and_pnl_chg:+.1f}%) | WR: {and_wr:.1f}%")

    # ═══════════════════════════════════════════════════════════════
    # LOSER DETAIL TABLE
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*110}")
    print(f"  ALL LOSERS — LTF Feature Values at Entry")
    print(f"{'='*110}")

    # Show top 5 discriminating features for each loser
    top5_features = [r["feature"] for r in results[:5]] if len(results) >= 5 else [r["feature"] for r in results]

    if top5_features:
        header = f"  {'Symbol':<8} {'Dir':<5} {'Entry Time':<20} {'PnL':>10} {'MFE':>6}"
        for feat in top5_features:
            short_name = feat.replace("m15_", "").replace("h1_", "")[:12]
            header += f" {short_name:>12}"
        print(header)
        print(f"  {'-'*8} {'-'*5} {'-'*20} {'-'*10} {'-'*6}" + f" {'-'*12}" * len(top5_features))

        for _, row in losers.iterrows():
            line = f"  {row['symbol']:<8} {row['direction']:<5} {str(row['entry_time']):<20} ${row['pnl']:>9,.0f} {row['mfe_atr']:>5.3f}"
            for feat in top5_features:
                val = row.get(feat, None)
                if val is not None and not (isinstance(val, float) and np.isnan(val)):
                    line += f" {val:>12.4f}"
                else:
                    line += f" {'N/A':>12}"
            print(line)

    print(f"\n{'='*110}")
    print(f"  ANALYSIS COMPLETE")
    print(f"{'='*110}")


if __name__ == "__main__":
    main()
