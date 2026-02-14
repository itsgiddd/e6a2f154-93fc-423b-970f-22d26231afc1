#!/usr/bin/env python3
"""
LOWER-TIMEFRAME LOSER PROFILER V2 -- V4 ZeroPoint
===================================================
V2 fixes V1's key problem: M15 data only covers ~520 days (50K bars),
but H4 data covers ~2860 days (5000 bars). Many older trades have no
M15 features, crippling the statistical analysis.

V2 approach:
  1. Extract H4 BAR-LEVEL "micro-features" for ALL trades (full coverage)
  2. Also extract LTF features (M15/H1) where available
  3. Run Cohen's d analysis on both feature sets
  4. Filter discovery with "cheap" (<2% cost) and "aggressive" (<10% cost) tables
  5. Combination filter section (AND logic on top feature pairs)

H4 Micro-Features (12 features, available for ALL 1101+ trades):
  a) h4_bar_rejection_wick
  b) h4_body_alignment
  c) h4_close_position
  d) h4_prev_bar_against
  e) h4_2bar_reversal
  f) h4_gap_against
  g) h4_vol_ratio
  h) h4_range_vs_atr
  i) h4_close_vs_trailing_stop
  j) h4_prev_3bar_range
  k) h4_consecutive_against
  l) h4_signal_bar_vs_prev_range
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


# ===================================================================
# V4 Position (EXACT copy from V1 -- do not modify trade management)
# ===================================================================
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


# ===================================================================
# H4 MICRO-FEATURE EXTRACTION (available for ALL trades)
# ===================================================================

def extract_h4_micro_features(df, bar_idx, direction, atr_val):
    """
    Extract 12 micro-features from the H4 signal bar and its immediate context.
    These features are available for EVERY trade since they use only H4 data.

    df:        H4 DataFrame with OHLCV columns
    bar_idx:   Index of the signal bar in df
    direction: "BUY" or "SELL"
    atr_val:   H4 ATR value at signal bar

    Returns dict of features.
    """
    features = {}
    is_buy = direction == "BUY"

    sig_open = float(df["open"].iloc[bar_idx])
    sig_high = float(df["high"].iloc[bar_idx])
    sig_low = float(df["low"].iloc[bar_idx])
    sig_close = float(df["close"].iloc[bar_idx])
    sig_range = sig_high - sig_low
    sig_body = sig_close - sig_open  # positive = bullish

    # (a) h4_bar_rejection_wick: wick AGAINST signal direction / bar range
    #     BUY with big upper wick = selling pressure
    if sig_range > 0:
        if is_buy:
            # Upper wick for a bullish bar = high - max(open,close)
            rejection_wick = (sig_high - max(sig_open, sig_close)) / sig_range
        else:
            # Lower wick for a bearish bar = min(open,close) - low
            rejection_wick = (min(sig_open, sig_close) - sig_low) / sig_range
    else:
        rejection_wick = 0.0
    features["h4_bar_rejection_wick"] = rejection_wick

    # (b) h4_body_alignment: Is the bar body in the signal direction?
    #     BUY: body > 0 means bullish = aligned. Normalize to [-1, 1].
    if sig_range > 0:
        body_ratio = sig_body / sig_range  # -1 to +1
        if is_buy:
            features["h4_body_alignment"] = body_ratio  # positive = aligned
        else:
            features["h4_body_alignment"] = -body_ratio  # flip: negative body = aligned for SELL
    else:
        features["h4_body_alignment"] = 0.0

    # (c) h4_close_position: Where close sits in bar range. (close-low)/(high-low).
    #     BUY near bar low = weak.
    if sig_range > 0:
        close_pos = (sig_close - sig_low) / sig_range
    else:
        close_pos = 0.5
    features["h4_close_position"] = close_pos

    # (d) h4_prev_bar_against: Was the previous H4 bar against signal direction?
    if bar_idx >= 1:
        prev_open = float(df["open"].iloc[bar_idx - 1])
        prev_close = float(df["close"].iloc[bar_idx - 1])
        prev_bullish = prev_close > prev_open
        if is_buy:
            features["h4_prev_bar_against"] = 0.0 if prev_bullish else 1.0
        else:
            features["h4_prev_bar_against"] = 1.0 if prev_bullish else 0.0
    else:
        features["h4_prev_bar_against"] = 0.5

    # (e) h4_2bar_reversal: Did the signal bar reverse a large prior bar?
    #     Big bearish bar then flip to BUY = potentially weak flip.
    #     Measured as: prev bar body (against direction) / ATR
    if bar_idx >= 1 and atr_val > 0:
        prev_open = float(df["open"].iloc[bar_idx - 1])
        prev_close = float(df["close"].iloc[bar_idx - 1])
        prev_body = prev_close - prev_open  # positive = bullish
        if is_buy:
            # BUY signal: prev bar bearish body magnitude
            reversal_strength = max(0, -prev_body) / atr_val
        else:
            # SELL signal: prev bar bullish body magnitude
            reversal_strength = max(0, prev_body) / atr_val
        features["h4_2bar_reversal"] = reversal_strength
    else:
        features["h4_2bar_reversal"] = 0.0

    # (f) h4_gap_against: Gap between prev close and signal open, against direction.
    #     Normalized to ATR.
    if bar_idx >= 1 and atr_val > 0:
        prev_close_val = float(df["close"].iloc[bar_idx - 1])
        gap = sig_open - prev_close_val  # positive = gap up
        if is_buy:
            # Gap DOWN against a BUY = negative gap is against
            features["h4_gap_against"] = max(0, -gap) / atr_val
        else:
            # Gap UP against a SELL = positive gap is against
            features["h4_gap_against"] = max(0, gap) / atr_val
    else:
        features["h4_gap_against"] = 0.0

    # (g) h4_vol_ratio: Volume on signal bar vs 10-bar avg.
    if "volume" in df.columns and bar_idx >= 10:
        vols = df["volume"].iloc[bar_idx - 10:bar_idx].values.astype(float)
        vol_avg = np.mean(vols)
        vol_sig = float(df["volume"].iloc[bar_idx])
        features["h4_vol_ratio"] = vol_sig / vol_avg if vol_avg > 0 else 1.0
    else:
        features["h4_vol_ratio"] = 1.0

    # (h) h4_range_vs_atr: Signal bar range / ATR. Small = weak conviction.
    if atr_val > 0:
        features["h4_range_vs_atr"] = sig_range / atr_val
    else:
        features["h4_range_vs_atr"] = 1.0

    # (i) h4_close_vs_trailing_stop: Distance from close to ATR trailing stop, normalized to ATR.
    #     Uses the trailing_stop column from compute_zeropoint_state if available.
    if "trailing_stop" in df.columns and atr_val > 0:
        ts = float(df["trailing_stop"].iloc[bar_idx])
        if not np.isnan(ts):
            dist = sig_close - ts  # positive = above stop (bullish)
            features["h4_close_vs_trailing_stop"] = dist / atr_val
        else:
            features["h4_close_vs_trailing_stop"] = 0.0
    else:
        features["h4_close_vs_trailing_stop"] = 0.0

    # (j) h4_prev_3bar_range: Range of last 3 bars / ATR. Tight = consolidation.
    if bar_idx >= 3 and atr_val > 0:
        recent_high = float(df["high"].iloc[bar_idx - 2:bar_idx + 1].max())
        recent_low = float(df["low"].iloc[bar_idx - 2:bar_idx + 1].min())
        features["h4_prev_3bar_range"] = (recent_high - recent_low) / atr_val
    else:
        features["h4_prev_3bar_range"] = 1.0

    # (k) h4_consecutive_against: Count of H4 bars AGAINST signal direction before the flip.
    consec = 0
    for k in range(bar_idx - 1, max(bar_idx - 20, -1), -1):
        if k < 0:
            break
        k_open = float(df["open"].iloc[k])
        k_close = float(df["close"].iloc[k])
        k_bullish = k_close > k_open
        if is_buy and not k_bullish:
            consec += 1
        elif not is_buy and k_bullish:
            consec += 1
        else:
            break
    features["h4_consecutive_against"] = float(consec)

    # (l) h4_signal_bar_vs_prev_range: Signal bar range / previous bar range.
    #     Smaller = weaker flip.
    if bar_idx >= 1:
        prev_high = float(df["high"].iloc[bar_idx - 1])
        prev_low = float(df["low"].iloc[bar_idx - 1])
        prev_range = prev_high - prev_low
        if prev_range > 0:
            features["h4_signal_bar_vs_prev_range"] = sig_range / prev_range
        else:
            features["h4_signal_bar_vs_prev_range"] = 1.0
    else:
        features["h4_signal_bar_vs_prev_range"] = 1.0

    return features


# ===================================================================
# LOWER-TIMEFRAME FEATURE EXTRACTION (same as V1, where data exists)
# ===================================================================

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

    # -- 1. RSI at signal time --
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

    # -- 2. Momentum (price change over last N bars, normalized to H4 ATR) --
    for n_bars in [4, 8, 16]:
        if len(closes) > n_bars:
            move = (closes[-1] - closes[-1 - n_bars])
            if direction == "SELL":
                move = -move
            features[f"{prefix}_momentum_{n_bars}bar"] = move / h4_atr if h4_atr > 0 else 0
        else:
            features[f"{prefix}_momentum_{n_bars}bar"] = 0

    # -- 3. Candle pattern on last LTF bar --
    last_bar_range = highs[-1] - lows[-1]
    last_bar_body = abs(closes[-1] - opens[-1])
    features[f"{prefix}_last_bar_body_ratio"] = last_bar_body / last_bar_range if last_bar_range > 0 else 0

    # Is last LTF bar agreeing with H4 signal direction?
    last_bullish = closes[-1] > opens[-1]
    features[f"{prefix}_last_bar_agrees"] = 1 if ((direction == "BUY" and last_bullish) or
                                                   (direction == "SELL" and not last_bullish)) else 0

    # -- 4. Consecutive bars in signal direction (LTF momentum) --
    consec = 0
    for k in range(len(closes) - 1, max(len(closes) - 20, 0), -1):
        bar_bull = closes[k] > opens[k]
        if (direction == "BUY" and bar_bull) or (direction == "SELL" and not bar_bull):
            consec += 1
        else:
            break
    features[f"{prefix}_consec_agree_bars"] = consec

    # -- 5. Consecutive bars AGAINST signal direction (counter-momentum) --
    consec_against = 0
    for k in range(len(closes) - 1, max(len(closes) - 20, 0), -1):
        bar_bull = closes[k] > opens[k]
        if (direction == "BUY" and not bar_bull) or (direction == "SELL" and bar_bull):
            consec_against += 1
        else:
            break
    features[f"{prefix}_consec_against_bars"] = consec_against

    # -- 6. Volume spike at signal time --
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

    # -- 7. ATR on LTF (volatility regime at signal time) --
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

    # -- 8. Price position on LTF range (are we at a micro S/R?) --
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

    # -- 9. Wicks analysis -- are recent LTF bars showing rejection? --
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
            features[f"{prefix}_rejection_wick"] = np.mean(upper_wicks)
        else:
            features[f"{prefix}_rejection_wick"] = np.mean(lower_wicks)
    else:
        features[f"{prefix}_avg_upper_wick"] = 0
        features[f"{prefix}_avg_lower_wick"] = 0
        features[f"{prefix}_rejection_wick"] = 0

    # -- 10. Engulfing / doji on last bar --
    if len(closes) >= 2:
        prev_body = abs(closes[-2] - opens[-2])
        cur_body = abs(closes[-1] - opens[-1])
        features[f"{prefix}_engulfing"] = 1 if cur_body > prev_body * 1.5 else 0
        features[f"{prefix}_doji"] = 1 if (last_bar_range > 0 and last_bar_body / last_bar_range < 0.1) else 0
    else:
        features[f"{prefix}_engulfing"] = 0
        features[f"{prefix}_doji"] = 0

    # -- 11. Price distance from LTF EMAs --
    if len(closes) >= 20:
        ema_fast = pd.Series(closes).ewm(span=8, adjust=False).mean().iloc[-1]
        ema_slow = pd.Series(closes).ewm(span=20, adjust=False).mean().iloc[-1]
        features[f"{prefix}_price_vs_ema8"] = (cur_close - ema_fast) / h4_atr if h4_atr > 0 else 0
        features[f"{prefix}_price_vs_ema20"] = (cur_close - ema_slow) / h4_atr if h4_atr > 0 else 0
        features[f"{prefix}_ema_cross"] = 1 if ema_fast > ema_slow else 0

        if direction == "BUY":
            features[f"{prefix}_ema_agrees"] = 1 if ema_fast > ema_slow else 0
        else:
            features[f"{prefix}_ema_agrees"] = 1 if ema_fast < ema_slow else 0
    else:
        features[f"{prefix}_price_vs_ema8"] = 0
        features[f"{prefix}_price_vs_ema20"] = 0
        features[f"{prefix}_ema_cross"] = 0.5
        features[f"{prefix}_ema_agrees"] = 0.5

    # -- 12. Spike-then-retrace detection (false breakout) --
    if len(closes) >= 8:
        if direction == "BUY":
            peak = np.max(highs[-8:])
            spike_size = (peak - closes[-1]) / h4_atr if h4_atr > 0 else 0
        else:
            trough = np.min(lows[-8:])
            spike_size = (closes[-1] - trough) / h4_atr if h4_atr > 0 else 0
        features[f"{prefix}_retrace_from_peak"] = spike_size
    else:
        features[f"{prefix}_retrace_from_peak"] = 0

    # -- 13. Divergence: price making new high/low but momentum weakening --
    if len(closes) >= 16:
        first_move = abs(closes[-16] - closes[-8])
        second_move = abs(closes[-8] - closes[-1])
        if first_move > 0:
            features[f"{prefix}_momentum_ratio"] = second_move / first_move
        else:
            features[f"{prefix}_momentum_ratio"] = 1.0
    else:
        features[f"{prefix}_momentum_ratio"] = 1.0

    # -- 14. Bollinger Band position --
    if len(closes) >= 20:
        sma20 = np.mean(closes[-20:])
        std20 = np.std(closes[-20:])
        upper_bb = sma20 + 2 * std20
        lower_bb = sma20 - 2 * std20
        bb_width = (upper_bb - lower_bb) / h4_atr if h4_atr > 0 else 0
        features[f"{prefix}_bb_width"] = bb_width

        if upper_bb != lower_bb:
            features[f"{prefix}_bb_position"] = (cur_close - lower_bb) / (upper_bb - lower_bb)
        else:
            features[f"{prefix}_bb_position"] = 0.5
    else:
        features[f"{prefix}_bb_width"] = 0
        features[f"{prefix}_bb_position"] = 0.5

    # -- 15. Recent high/low break --
    if len(closes) >= 16:
        prev_high_16 = np.max(highs[-16:-4])
        prev_low_16 = np.min(lows[-16:-4])
        recent_high_4 = np.max(highs[-4:])
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

    # -- 16. Speed of approach --
    if len(closes) >= 4:
        speed = abs(closes[-1] - closes[-4]) / 3
        speed_norm = speed / h4_atr if h4_atr > 0 else 0
        features[f"{prefix}_approach_speed"] = speed_norm
    else:
        features[f"{prefix}_approach_speed"] = 0

    return features


# ===================================================================
# MAIN
# ===================================================================

def main():
    print("=" * 110)
    print("  LOWER-TIMEFRAME LOSER PROFILER V2 -- V4 ZeroPoint")
    print("  H4 micro-features (ALL trades) + LTF features (where available)")
    print("=" * 110)

    if not mt5.initialize():
        print("ERROR: Could not initialize MT5")
        return

    # -- Load H4 data --
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

    # -- Load M15 + H1 data --
    print("\n[2/3] Loading M15 + H1 data...")
    ltf_data = {}  # {sym: {"M15": df, "H1": df}}
    for sym in SYMBOLS:
        if sym not in h4_data:
            continue
        _, resolved = h4_data[sym]
        ltf_data[sym] = {}

        # M15 -- fetch as much as broker provides (up to 50K bars ~520 days)
        rates_m15 = mt5.copy_rates_from_pos(resolved, mt5.TIMEFRAME_M15, 0, min(FETCH_BARS_H4 * 16, 50000))
        if rates_m15 is not None and len(rates_m15) >= 100:
            df_m15 = pd.DataFrame(rates_m15)
            df_m15["time"] = pd.to_datetime(df_m15["time"], unit="s")
            df_m15.rename(columns={"tick_volume": "volume"}, inplace=True)
            ltf_data[sym]["M15"] = df_m15
            m15_start = df_m15["time"].iloc[0]
            m15_end = df_m15["time"].iloc[-1]
            print(f"  {sym} M15: {len(df_m15)} bars ({m15_start.date()} to {m15_end.date()})")

        # H1 -- fetch more bars since 1 H4 bar = 4 H1 bars
        rates_h1 = mt5.copy_rates_from_pos(resolved, mt5.TIMEFRAME_H1, 0, min(FETCH_BARS_H4 * 4, 20000))
        if rates_h1 is not None and len(rates_h1) >= 100:
            df_h1 = pd.DataFrame(rates_h1)
            df_h1["time"] = pd.to_datetime(df_h1["time"], unit="s")
            df_h1.rename(columns={"tick_volume": "volume"}, inplace=True)
            ltf_data[sym]["H1"] = df_h1
            h1_start = df_h1["time"].iloc[0]
            h1_end = df_h1["time"].iloc[-1]
            print(f"  {sym} H1:  {len(df_h1)} bars ({h1_start.date()} to {h1_end.date()})")

    mt5.shutdown()

    if not h4_data:
        print("No data!")
        return

    # -- Run H4 simulation + extract features at each signal --
    print("\n[3/3] Simulating V4 trades + extracting H4 micro + LTF features...")

    trade_records = []
    ltf_coverage_count = 0
    total_signals = 0

    for sym, (df, resolved) in h4_data.items():
        cont_sz = contract_size(sym)
        n = len(df)
        pos_obj = None
        sym_ltf = ltf_data.get(sym, {})

        def record_trade(pos):
            """Record a completed trade with its features."""
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
            # Add H4 micro-features (always present)
            if hasattr(pos, '_h4_micro_features') and pos._h4_micro_features:
                rec.update(pos._h4_micro_features)
            # Add LTF features (where available)
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
                total_signals += 1

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

                # -- EXTRACT H4 MICRO-FEATURES (always available) --
                h4_micro = extract_h4_micro_features(df, i, direction, atr_val)
                pos_obj._h4_micro_features = h4_micro

                # -- EXTRACT LTF FEATURES (where data available) --
                h4_time = row["time"]
                ltf_features = {}
                has_ltf = False

                for tf_key, tf_label in [("M15", "M15"), ("H1", "H1")]:
                    if tf_key in sym_ltf:
                        feats = extract_ltf_features(
                            sym_ltf[tf_key], h4_time, atr_val, direction, tf_label
                        )
                        if feats:
                            ltf_features.update(feats)
                            has_ltf = True

                pos_obj._ltf_features = ltf_features
                if has_ltf:
                    ltf_coverage_count += 1

        # Close remaining
        if pos_obj is not None and not pos_obj.closed:
            pos_obj.force_close(float(df.iloc[-1]["close"]))
            pos_obj.exit_time = df.iloc[-1]["time"]
            record_trade(pos_obj)

    # ===================================================================
    # ANALYSIS
    # ===================================================================
    df_trades = pd.DataFrame(trade_records)
    total = len(df_trades)
    losers = df_trades[df_trades["is_loser"] == 1]
    winners = df_trades[df_trades["is_loser"] == 0]

    print(f"\n{'='*110}")
    print(f"  TRADE SUMMARY")
    print(f"{'='*110}")
    print(f"  Total trades:        {total}")
    print(f"  Winners:             {len(winners)} ({100*len(winners)/total:.1f}%)")
    print(f"  Losers:              {len(losers)} ({100*len(losers)/total:.1f}%)")
    print(f"  Net PnL:             ${df_trades['pnl'].sum():,.0f}")
    print(f"  Gross Loss:          ${losers['pnl'].sum():,.0f}")
    print(f"  LTF coverage:        {ltf_coverage_count}/{total_signals} signals ({100*ltf_coverage_count/total_signals:.1f}%)")

    # Identify feature columns
    h4_micro_cols = [c for c in df_trades.columns if c.startswith("h4_") and c not in ("h4_atr",)]
    ltf_cols = [c for c in df_trades.columns if c.startswith("m15_") or c.startswith("h1_")]
    all_feature_cols = h4_micro_cols + ltf_cols

    print(f"  H4 micro-features:   {len(h4_micro_cols)}")
    print(f"  LTF features:        {len(ltf_cols)}")

    if not h4_micro_cols:
        print("\nERROR: No H4 micro-features extracted!")
        return

    # ===================================================================
    # SECTION 1: H4 MICRO-FEATURES (full coverage -- ALL trades)
    # ===================================================================
    print(f"\n{'='*110}")
    print(f"  SECTION 1: H4 MICRO-FEATURES -- Cohen's d Analysis (ALL {total} trades)")
    print(f"  These features cover ALL trades including ALL {len(losers)} losers")
    print(f"  Cohen's d: |d| > 0.5 = medium, |d| > 0.8 = large effect")
    print(f"{'='*110}")
    print(f"  {'Feature':<40} {'Loser Mean':>12} {'Winner Mean':>12} {'Cohen d':>10} {'Loser N':>8} {'Winner N':>8} {'Direction':>14}")
    print(f"  {'-'*40} {'-'*12} {'-'*12} {'-'*10} {'-'*8} {'-'*8} {'-'*14}")

    h4_results = []
    for col in sorted(h4_micro_cols):
        l_vals = losers[col].dropna()
        w_vals = winners[col].dropna()
        if len(l_vals) < 2 or len(w_vals) < 10:
            continue
        l_mean = l_vals.mean()
        w_mean = w_vals.mean()
        pooled_std = np.sqrt(((len(l_vals)-1)*l_vals.std()**2 + (len(w_vals)-1)*w_vals.std()**2) /
                             (len(l_vals) + len(w_vals) - 2))
        if pooled_std > 0:
            d = (l_mean - w_mean) / pooled_std
        else:
            d = 0

        if abs(d) > 0.1:
            dir_label = "LOSER HIGHER" if d > 0 else "LOSER LOWER"
        else:
            dir_label = "~same"

        h4_results.append({
            "feature": col,
            "loser_mean": l_mean,
            "winner_mean": w_mean,
            "cohen_d": d,
            "abs_d": abs(d),
            "direction": dir_label,
            "loser_n": len(l_vals),
            "winner_n": len(w_vals),
        })

    h4_results.sort(key=lambda x: x["abs_d"], reverse=True)

    for r in h4_results:
        marker = "***" if r["abs_d"] >= 0.5 else "** " if r["abs_d"] >= 0.3 else "   "
        print(f"  {marker}{r['feature']:<37} {r['loser_mean']:>12.4f} {r['winner_mean']:>12.4f} "
              f"{r['cohen_d']:>10.3f} {r['loser_n']:>8} {r['winner_n']:>8} {r['direction']:>14}")

    # ===================================================================
    # SECTION 2: LTF FEATURES (partial coverage)
    # ===================================================================
    if ltf_cols:
        print(f"\n{'='*110}")
        print(f"  SECTION 2: LTF FEATURES -- Cohen's d Analysis (partial coverage)")
        print(f"{'='*110}")

        # Count LTF coverage for losers
        ltf_loser_coverage = 0
        for _, row in losers.iterrows():
            has_any_ltf = False
            for col in ltf_cols:
                val = row.get(col, None)
                if val is not None and not (isinstance(val, float) and np.isnan(val)):
                    has_any_ltf = True
                    break
            if has_any_ltf:
                ltf_loser_coverage += 1

        print(f"  Losers with LTF data: {ltf_loser_coverage}/{len(losers)}")

        print(f"\n  {'Feature':<40} {'Loser Mean':>12} {'Winner Mean':>12} {'Cohen d':>10} {'Loser N':>8} {'Winner N':>8} {'Direction':>14}")
        print(f"  {'-'*40} {'-'*12} {'-'*12} {'-'*10} {'-'*8} {'-'*8} {'-'*14}")

        ltf_results = []
        for col in sorted(ltf_cols):
            l_vals = losers[col].dropna()
            w_vals = winners[col].dropna()
            if len(l_vals) < 2 or len(w_vals) < 10:
                continue
            l_mean = l_vals.mean()
            w_mean = w_vals.mean()
            pooled_std = np.sqrt(((len(l_vals)-1)*l_vals.std()**2 + (len(w_vals)-1)*w_vals.std()**2) /
                                 (len(l_vals) + len(w_vals) - 2))
            if pooled_std > 0:
                d = (l_mean - w_mean) / pooled_std
            else:
                d = 0

            if abs(d) > 0.1:
                dir_label = "LOSER HIGHER" if d > 0 else "LOSER LOWER"
            else:
                dir_label = "~same"

            ltf_results.append({
                "feature": col,
                "loser_mean": l_mean,
                "winner_mean": w_mean,
                "cohen_d": d,
                "abs_d": abs(d),
                "direction": dir_label,
                "loser_n": len(l_vals),
                "winner_n": len(w_vals),
            })

        ltf_results.sort(key=lambda x: x["abs_d"], reverse=True)

        for r in ltf_results:
            marker = "***" if r["abs_d"] >= 0.5 else "** " if r["abs_d"] >= 0.3 else "   "
            print(f"  {marker}{r['feature']:<37} {r['loser_mean']:>12.4f} {r['winner_mean']:>12.4f} "
                  f"{r['cohen_d']:>10.3f} {r['loser_n']:>8} {r['winner_n']:>8} {r['direction']:>14}")
    else:
        ltf_results = []

    # Combine all results for filter discovery
    all_results = h4_results + ltf_results

    # ===================================================================
    # FILTER DISCOVERY -- H4 micro-features (full coverage)
    # ===================================================================
    print(f"\n{'='*110}")
    print(f"  FILTER DISCOVERY -- Testing H4 micro-features as trade/skip filters")
    print(f"  Goal: Skip losers WITHOUT losing significant winner profit")
    print(f"{'='*110}")

    baseline_pnl = df_trades["pnl"].sum()
    baseline_losses = losers["pnl"].sum()
    total_losers = len(losers)

    # Test H4 micro-features with |d| >= 0.15 (lower threshold since we have full coverage)
    promising_h4 = [r for r in h4_results if r["abs_d"] >= 0.15]

    if not promising_h4:
        print("\n  No H4 micro-features with |d| >= 0.15 found.")
    else:
        print(f"\n  Testing {len(promising_h4)} H4 micro-features (|d| >= 0.15)...")
        print(f"  Baseline: {total} trades, PnL ${baseline_pnl:,.0f}, Losses ${baseline_losses:,.0f}\n")

    all_filters = []  # collect all filters (both H4 and LTF)
    percentiles = [5, 10, 15, 20, 25, 30, 70, 75, 80, 85, 90, 95]

    # Test H4 micro-features
    for r in promising_h4:
        col = r["feature"]
        l_vals = losers[col].dropna()
        w_vals = winners[col].dropna()

        all_vals = pd.concat([l_vals, w_vals])

        for pct in percentiles:
            threshold = np.percentile(all_vals, pct)

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

                if losers_skipped < 2:
                    continue

                kept_pnl = kept["pnl"].sum()
                pnl_change = kept_pnl - baseline_pnl
                pnl_change_pct = 100 * pnl_change / baseline_pnl if baseline_pnl != 0 else 0

                loss_reduction = losers_skipped / total_losers * 100 if total_losers > 0 else 0

                if winners_skipped > 0:
                    efficiency = losers_skipped / winners_skipped
                else:
                    efficiency = float('inf') if losers_skipped > 0 else 0

                kept_losers = len(kept[kept["is_loser"] == 1])
                kept_winners = len(kept[kept["is_loser"] == 0])
                new_wr = 100 * kept_winners / len(kept) if len(kept) > 0 else 0

                profit_cost_pct = -pnl_change_pct if pnl_change_pct < 0 else 0
                quality = (loss_reduction / 100) * max(0, 1 + pnl_change_pct / 100)

                all_filters.append({
                    "feature": col,
                    "source": "H4",
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
                    "profit_cost_pct": profit_cost_pct,
                    "loss_reduction": loss_reduction,
                    "efficiency": efficiency,
                    "quality": quality,
                    "new_wr": new_wr,
                })

    # Also test LTF features (partial coverage -- only useful for recent trades)
    promising_ltf = [r for r in ltf_results if r["abs_d"] >= 0.25]
    for r in promising_ltf:
        col = r["feature"]
        # Only test on trades that HAVE this feature (not NaN)
        valid_mask = df_trades[col].notna()
        df_valid = df_trades[valid_mask]
        if len(df_valid) < 50:
            continue

        l_vals = df_valid[df_valid["is_loser"] == 1][col].dropna()
        w_vals = df_valid[df_valid["is_loser"] == 0][col].dropna()
        if len(l_vals) < 2:
            continue

        all_vals = pd.concat([l_vals, w_vals])

        for pct in percentiles:
            threshold = np.percentile(all_vals, pct)

            for op, op_name in [("lt", "<"), ("gt", ">")]:
                if op == "lt":
                    skipped = df_valid[df_valid[col] < threshold]
                    kept = df_valid[df_valid[col] >= threshold]
                else:
                    skipped = df_valid[df_valid[col] > threshold]
                    kept = df_valid[df_valid[col] <= threshold]

                if len(skipped) == 0 or len(kept) == 0:
                    continue

                losers_skipped = len(skipped[skipped["is_loser"] == 1])
                winners_skipped = len(skipped[skipped["is_loser"] == 0])

                if losers_skipped < 2:
                    continue

                kept_pnl = kept["pnl"].sum()
                valid_baseline = df_valid["pnl"].sum()
                pnl_change = kept_pnl - valid_baseline
                pnl_change_pct = 100 * pnl_change / valid_baseline if valid_baseline != 0 else 0

                valid_losers = len(df_valid[df_valid["is_loser"] == 1])
                loss_reduction = losers_skipped / valid_losers * 100 if valid_losers > 0 else 0

                if winners_skipped > 0:
                    efficiency = losers_skipped / winners_skipped
                else:
                    efficiency = float('inf') if losers_skipped > 0 else 0

                kept_winners = len(kept[kept["is_loser"] == 0])
                new_wr = 100 * kept_winners / len(kept) if len(kept) > 0 else 0

                profit_cost_pct = -pnl_change_pct if pnl_change_pct < 0 else 0
                quality = (loss_reduction / 100) * max(0, 1 + pnl_change_pct / 100)

                all_filters.append({
                    "feature": col,
                    "source": "LTF",
                    "op": op_name,
                    "threshold": threshold,
                    "pct": pct,
                    "losers_skipped": losers_skipped,
                    "winners_skipped": winners_skipped,
                    "total_skipped": len(skipped),
                    "kept_trades": len(kept),
                    "kept_pnl": kept_pnl,
                    "pnl_change": pnl_change,
                    "pnl_change_pct": pnl_change_pct,
                    "profit_cost_pct": profit_cost_pct,
                    "loss_reduction": loss_reduction,
                    "efficiency": efficiency,
                    "quality": quality,
                    "new_wr": new_wr,
                })

    if not all_filters:
        print("\n  No filters found that catch >= 2 losers.")
        print("  The losers are indistinguishable from winners on these features.")
        return

    # ===================================================================
    # CHEAP FILTERS (< 2% profit cost)
    # ===================================================================
    cheap_filters = [f for f in all_filters if f["profit_cost_pct"] < 2.0]
    cheap_filters.sort(key=lambda x: x["losers_skipped"], reverse=True)

    print(f"\n{'='*110}")
    print(f"  CHEAP FILTERS (< 2% profit cost) -- sorted by losers caught")
    print(f"  These filters barely hurt profitability while removing some losers")
    print(f"{'='*110}")

    if cheap_filters:
        print(f"  {'Feature':<35} {'Src':>3} {'Rule':<15} {'Skip':>5} {'L_Skip':>6} {'W_Skip':>6} "
              f"{'PnL Chg':>10} {'%Cost':>6} {'LossCut':>8} {'WR':>6} {'Quality':>8}")
        print(f"  {'-'*35} {'-'*3} {'-'*15} {'-'*5} {'-'*6} {'-'*6} "
              f"{'-'*10} {'-'*6} {'-'*8} {'-'*6} {'-'*8}")

        for f in cheap_filters[:25]:
            rule = f"{f['op']} {f['threshold']:.4f}"
            print(f"  {f['feature']:<35} {f['source']:>3} {rule:<15} {f['total_skipped']:>5} "
                  f"{f['losers_skipped']:>6} {f['winners_skipped']:>6} "
                  f"${f['pnl_change']:>9,.0f} {f['profit_cost_pct']:>5.1f}% "
                  f"{f['loss_reduction']:>7.1f}% {f['new_wr']:>5.1f}% {f['quality']:>8.3f}")
    else:
        print("  No filters with < 2% profit cost found.")

    # ===================================================================
    # AGGRESSIVE FILTERS (< 10% profit cost)
    # ===================================================================
    aggressive_filters = [f for f in all_filters if f["profit_cost_pct"] < 10.0]
    aggressive_filters.sort(key=lambda x: x["losers_skipped"], reverse=True)

    print(f"\n{'='*110}")
    print(f"  AGGRESSIVE FILTERS (< 10% profit cost) -- sorted by losers caught")
    print(f"  These filters catch more losers but may cost some profit")
    print(f"{'='*110}")

    if aggressive_filters:
        print(f"  {'Feature':<35} {'Src':>3} {'Rule':<15} {'Skip':>5} {'L_Skip':>6} {'W_Skip':>6} "
              f"{'PnL Chg':>10} {'%Cost':>6} {'LossCut':>8} {'WR':>6} {'Quality':>8}")
        print(f"  {'-'*35} {'-'*3} {'-'*15} {'-'*5} {'-'*6} {'-'*6} "
              f"{'-'*10} {'-'*6} {'-'*8} {'-'*6} {'-'*8}")

        for f in aggressive_filters[:25]:
            rule = f"{f['op']} {f['threshold']:.4f}"
            print(f"  {f['feature']:<35} {f['source']:>3} {rule:<15} {f['total_skipped']:>5} "
                  f"{f['losers_skipped']:>6} {f['winners_skipped']:>6} "
                  f"${f['pnl_change']:>9,.0f} {f['profit_cost_pct']:>5.1f}% "
                  f"{f['loss_reduction']:>7.1f}% {f['new_wr']:>5.1f}% {f['quality']:>8.3f}")
    else:
        print("  No filters with < 10% profit cost found.")

    # ===================================================================
    # BEST FILTER DEEP DIVE
    # ===================================================================
    # Pick the best filter by quality from all filters with < 10% cost
    if aggressive_filters:
        best_by_quality = sorted(aggressive_filters, key=lambda x: x["quality"], reverse=True)
        best = best_by_quality[0]

        print(f"\n{'='*110}")
        print(f"  BEST FILTER DEEP DIVE (highest quality score)")
        print(f"{'='*110}")
        print(f"  Feature:          {best['feature']} ({best['source']})")
        print(f"  Rule:             Skip if {best['feature']} {best['op']} {best['threshold']:.6f}")
        print(f"  Losers caught:    {best['losers_skipped']}/{total_losers} ({best['loss_reduction']:.1f}%)")
        print(f"  Winners lost:     {best['winners_skipped']}")
        print(f"  PnL change:       ${best['pnl_change']:,.0f} ({best['pnl_change_pct']:+.1f}%)")
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
                feat_val = rl.get(best['feature'], None)
                feat_str = f"{feat_val:.4f}" if feat_val is not None and not (isinstance(feat_val, float) and np.isnan(feat_val)) else "N/A"
                print(f"    {rl['symbol']} {rl['direction']} {rl['entry_time']} "
                      f"PnL=${rl['pnl']:,.0f} MFE={rl['mfe_atr']:.3f}x ATR "
                      f"{best['feature']}={feat_str}")

        # Show the caught losers
        if best["op"] == "<":
            caught = df_trades[(df_trades[best['feature']] < best['threshold']) & (df_trades["is_loser"] == 1)]
        else:
            caught = df_trades[(df_trades[best['feature']] > best['threshold']) & (df_trades["is_loser"] == 1)]

        if len(caught) > 0:
            print(f"\n  Caught losers ({len(caught)}):")
            for _, cl in caught.iterrows():
                feat_val = cl.get(best['feature'], None)
                feat_str = f"{feat_val:.4f}" if feat_val is not None and not (isinstance(feat_val, float) and np.isnan(feat_val)) else "N/A"
                print(f"    {cl['symbol']} {cl['direction']} {cl['entry_time']} "
                      f"PnL=${cl['pnl']:,.0f} MFE={cl['mfe_atr']:.3f}x ATR "
                      f"{best['feature']}={feat_str}")

    # ===================================================================
    # COMBINATION FILTER (AND logic on top feature pairs)
    # ===================================================================
    print(f"\n{'='*110}")
    print(f"  COMBINATION FILTERS -- AND logic (skip only when BOTH features trigger)")
    print(f"  More precise: fewer false positives (winners wrongly skipped)")
    print(f"{'='*110}")

    # Get unique top features from H4 micro-features (sorted by |d|)
    seen_features = set()
    unique_top_h4 = []
    for r in h4_results:
        if r["feature"] not in seen_features and r["abs_d"] >= 0.15:
            seen_features.add(r["feature"])
            unique_top_h4.append(r["feature"])
            if len(unique_top_h4) >= 5:
                break

    if len(unique_top_h4) < 2:
        print("  Not enough discriminating H4 features for combination testing.")
    else:
        print(f"  Testing AND combinations of top {len(unique_top_h4)} H4 features: {unique_top_h4}")

        # For each feature, find the best threshold (the one from aggressive_filters with most losers)
        best_threshold_for_feature = {}
        for f in aggressive_filters:
            feat = f["feature"]
            if feat in unique_top_h4:
                key = feat
                if key not in best_threshold_for_feature or f["losers_skipped"] > best_threshold_for_feature[key]["losers_skipped"]:
                    best_threshold_for_feature[key] = f

        combo_results = []

        for i in range(len(unique_top_h4)):
            for j in range(i + 1, len(unique_top_h4)):
                feat1 = unique_top_h4[i]
                feat2 = unique_top_h4[j]

                if feat1 not in best_threshold_for_feature or feat2 not in best_threshold_for_feature:
                    continue

                f1 = best_threshold_for_feature[feat1]
                f2 = best_threshold_for_feature[feat2]

                # Build masks
                if f1["op"] == "<":
                    mask1 = df_trades[feat1] < f1["threshold"]
                else:
                    mask1 = df_trades[feat1] > f1["threshold"]

                if f2["op"] == "<":
                    mask2 = df_trades[feat2] < f2["threshold"]
                else:
                    mask2 = df_trades[feat2] > f2["threshold"]

                # AND combination: skip only when BOTH trigger
                skip_and = mask1 & mask2
                # Handle NaN: NaN comparisons return False, which means NaN rows won't be skipped
                # That's the correct behavior -- don't skip if we can't evaluate
                kept_and = df_trades[~skip_and]

                if len(kept_and) == 0 or skip_and.sum() == 0:
                    continue

                and_losers_caught = len(df_trades[skip_and & (df_trades["is_loser"] == 1)])
                and_winners_lost = len(df_trades[skip_and & (df_trades["is_loser"] == 0)])

                if and_losers_caught < 2:
                    continue

                and_pnl = kept_and["pnl"].sum()
                and_pnl_chg = 100 * (and_pnl - baseline_pnl) / baseline_pnl if baseline_pnl != 0 else 0
                and_wr = 100 * len(kept_and[kept_and["is_loser"] == 0]) / len(kept_and)
                and_loss_reduction = and_losers_caught / total_losers * 100
                and_cost = -and_pnl_chg if and_pnl_chg < 0 else 0
                and_quality = (and_loss_reduction / 100) * max(0, 1 + and_pnl_chg / 100)

                combo_results.append({
                    "feat1": feat1,
                    "op1": f1["op"],
                    "thresh1": f1["threshold"],
                    "feat2": feat2,
                    "op2": f2["op"],
                    "thresh2": f2["threshold"],
                    "losers_caught": and_losers_caught,
                    "winners_lost": and_winners_lost,
                    "total_skipped": int(skip_and.sum()),
                    "kept_trades": len(kept_and),
                    "pnl": and_pnl,
                    "pnl_chg_pct": and_pnl_chg,
                    "profit_cost_pct": and_cost,
                    "loss_reduction": and_loss_reduction,
                    "new_wr": and_wr,
                    "quality": and_quality,
                })

        if combo_results:
            combo_results.sort(key=lambda x: x["quality"], reverse=True)

            print(f"\n  {'Combo':<70} {'Skip':>5} {'L_Catch':>7} {'W_Lost':>6} "
                  f"{'PnL Chg':>10} {'%Cost':>6} {'LossCut':>8} {'WR':>6} {'Quality':>8}")
            print(f"  {'-'*70} {'-'*5} {'-'*7} {'-'*6} "
                  f"{'-'*10} {'-'*6} {'-'*8} {'-'*6} {'-'*8}")

            for c in combo_results:
                combo_desc = f"({c['feat1']} {c['op1']} {c['thresh1']:.4f}) AND ({c['feat2']} {c['op2']} {c['thresh2']:.4f})"
                print(f"  {combo_desc:<70} {c['total_skipped']:>5} {c['losers_caught']:>7} "
                      f"{c['winners_lost']:>6} ${c['pnl'] - baseline_pnl:>9,.0f} "
                      f"{c['profit_cost_pct']:>5.1f}% {c['loss_reduction']:>7.1f}% "
                      f"{c['new_wr']:>5.1f}% {c['quality']:>8.3f}")
        else:
            print("  No AND combinations found that catch >= 2 losers.")

        # Also test with all H4 percentile thresholds for more combos
        print(f"\n  --- Exhaustive pairwise threshold search ---")

        exhaustive_combos = []
        for i in range(len(unique_top_h4)):
            for j in range(i + 1, len(unique_top_h4)):
                feat1 = unique_top_h4[i]
                feat2 = unique_top_h4[j]

                # Get all single-feature filters for feat1 and feat2
                f1_filters = [f for f in all_filters if f["feature"] == feat1 and f["source"] == "H4"]
                f2_filters = [f for f in all_filters if f["feature"] == feat2 and f["source"] == "H4"]

                for f1 in f1_filters:
                    for f2 in f2_filters:
                        if f1["op"] == "<":
                            mask1 = df_trades[feat1] < f1["threshold"]
                        else:
                            mask1 = df_trades[feat1] > f1["threshold"]

                        if f2["op"] == "<":
                            mask2 = df_trades[feat2] < f2["threshold"]
                        else:
                            mask2 = df_trades[feat2] > f2["threshold"]

                        skip_and = mask1 & mask2
                        kept_and = df_trades[~skip_and]

                        if len(kept_and) == 0 or skip_and.sum() == 0:
                            continue

                        and_losers = len(df_trades[skip_and & (df_trades["is_loser"] == 1)])
                        and_winners = len(df_trades[skip_and & (df_trades["is_loser"] == 0)])

                        if and_losers < 2:
                            continue

                        and_pnl = kept_and["pnl"].sum()
                        and_pnl_chg = 100 * (and_pnl - baseline_pnl) / baseline_pnl if baseline_pnl != 0 else 0
                        and_cost = -and_pnl_chg if and_pnl_chg < 0 else 0

                        if and_cost > 10:
                            continue  # skip if too expensive

                        and_wr = 100 * len(kept_and[kept_and["is_loser"] == 0]) / len(kept_and)
                        and_loss_reduction = and_losers / total_losers * 100
                        and_quality = (and_loss_reduction / 100) * max(0, 1 + and_pnl_chg / 100)

                        exhaustive_combos.append({
                            "feat1": feat1,
                            "op1": f1["op"],
                            "thresh1": f1["threshold"],
                            "pct1": f1["pct"],
                            "feat2": feat2,
                            "op2": f2["op"],
                            "thresh2": f2["threshold"],
                            "pct2": f2["pct"],
                            "losers_caught": and_losers,
                            "winners_lost": and_winners,
                            "total_skipped": int(skip_and.sum()),
                            "pnl_chg_pct": and_pnl_chg,
                            "profit_cost_pct": and_cost,
                            "loss_reduction": and_loss_reduction,
                            "new_wr": and_wr,
                            "quality": and_quality,
                        })

        if exhaustive_combos:
            # Show top 15 by quality
            exhaustive_combos.sort(key=lambda x: x["quality"], reverse=True)

            print(f"\n  Top 15 AND-combos (exhaustive search, sorted by quality):")
            print(f"  {'Feature 1':<25} {'Op':>2} {'P%':>3} {'Feature 2':<25} {'Op':>2} {'P%':>3} "
                  f"{'Skip':>5} {'L':>3} {'W':>4} {'%Cost':>6} {'LossCut':>8} {'WR':>6} {'Quality':>8}")
            print(f"  {'-'*25} {'-'*2} {'-'*3} {'-'*25} {'-'*2} {'-'*3} "
                  f"{'-'*5} {'-'*3} {'-'*4} {'-'*6} {'-'*8} {'-'*6} {'-'*8}")

            for c in exhaustive_combos[:15]:
                print(f"  {c['feat1']:<25} {c['op1']:>2} {c['pct1']:>3} "
                      f"{c['feat2']:<25} {c['op2']:>2} {c['pct2']:>3} "
                      f"{c['total_skipped']:>5} {c['losers_caught']:>3} {c['winners_lost']:>4} "
                      f"{c['profit_cost_pct']:>5.1f}% {c['loss_reduction']:>7.1f}% "
                      f"{c['new_wr']:>5.1f}% {c['quality']:>8.3f}")

            # Also show cheapest combos (< 2% cost)
            cheap_combos = [c for c in exhaustive_combos if c["profit_cost_pct"] < 2.0]
            if cheap_combos:
                cheap_combos.sort(key=lambda x: x["losers_caught"], reverse=True)
                print(f"\n  Cheapest AND-combos (< 2% profit cost, sorted by losers caught):")
                for c in cheap_combos[:10]:
                    print(f"    ({c['feat1']} {c['op1']} {c['thresh1']:.4f} [p{c['pct1']}]) "
                          f"AND ({c['feat2']} {c['op2']} {c['thresh2']:.4f} [p{c['pct2']}]) "
                          f"-> L={c['losers_caught']}, W={c['winners_lost']}, "
                          f"cost={c['profit_cost_pct']:.1f}%, WR={c['new_wr']:.1f}%")
        else:
            print("  No exhaustive AND combos found with >= 2 losers and < 10% cost.")

    # ===================================================================
    # LOSER DETAIL TABLE -- H4 Micro-Feature Values
    # ===================================================================
    print(f"\n{'='*110}")
    print(f"  ALL LOSERS -- H4 Micro-Feature Values at Entry")
    print(f"{'='*110}")

    top_h4_features = [r["feature"] for r in h4_results[:6]] if len(h4_results) >= 6 else [r["feature"] for r in h4_results]

    if top_h4_features:
        header = f"  {'Symbol':<8} {'Dir':<5} {'Entry Time':<22} {'PnL':>10} {'MFE':>6}"
        for feat in top_h4_features:
            short_name = feat.replace("h4_", "")[:14]
            header += f" {short_name:>14}"
        print(header)
        print(f"  {'-'*8} {'-'*5} {'-'*22} {'-'*10} {'-'*6}" + f" {'-'*14}" * len(top_h4_features))

        for _, row in losers.iterrows():
            line = f"  {row['symbol']:<8} {row['direction']:<5} {str(row['entry_time']):<22} ${row['pnl']:>9,.0f} {row['mfe_atr']:>5.3f}"
            for feat in top_h4_features:
                val = row.get(feat, None)
                if val is not None and not (isinstance(val, float) and np.isnan(val)):
                    line += f" {val:>14.4f}"
                else:
                    line += f" {'N/A':>14}"
            print(line)

    # ===================================================================
    # LOSER DETAIL TABLE -- Winner Comparison Stats
    # ===================================================================
    print(f"\n{'='*110}")
    print(f"  LOSER vs WINNER PERCENTILE ANALYSIS")
    print(f"  For each H4 micro-feature: where does each loser fall in the winner distribution?")
    print(f"{'='*110}")

    if top_h4_features and len(winners) > 0:
        header = f"  {'Symbol':<8} {'Dir':<5} {'Entry Time':<22}"
        for feat in top_h4_features:
            short_name = feat.replace("h4_", "")[:14]
            header += f" {short_name:>14}"
        print(header)
        print(f"  {'-'*8} {'-'*5} {'-'*22}" + f" {'-'*14}" * len(top_h4_features))

        for _, row in losers.iterrows():
            line = f"  {row['symbol']:<8} {row['direction']:<5} {str(row['entry_time']):<22}"
            for feat in top_h4_features:
                val = row.get(feat, None)
                if val is not None and not (isinstance(val, float) and np.isnan(val)):
                    w_vals = winners[feat].dropna().values
                    if len(w_vals) > 0:
                        pctile = 100 * np.sum(w_vals <= val) / len(w_vals)
                        line += f" {'p'+str(int(pctile)):>14}"
                    else:
                        line += f" {'N/A':>14}"
                else:
                    line += f" {'N/A':>14}"
            print(line)

    # ===================================================================
    # SUMMARY
    # ===================================================================
    print(f"\n{'='*110}")
    print(f"  ANALYSIS COMPLETE -- V2 SUMMARY")
    print(f"{'='*110}")
    print(f"  Total trades analyzed:    {total}")
    print(f"  H4 micro-features:       {len(h4_micro_cols)} (full coverage for all {total} trades)")
    print(f"  LTF features:            {len(ltf_cols)} (coverage for {ltf_coverage_count}/{total} trades)")
    print(f"  Losers:                  {len(losers)} ({100*len(losers)/total:.1f}%)")

    if h4_results:
        print(f"\n  Top 5 H4 discriminating features:")
        for i, r in enumerate(h4_results[:5]):
            print(f"    {i+1}. {r['feature']}: d={r['cohen_d']:+.3f} ({r['direction']})")

    if cheap_filters:
        best_cheap = sorted(cheap_filters, key=lambda x: x["quality"], reverse=True)[0]
        print(f"\n  Best cheap filter (< 2% cost):")
        print(f"    {best_cheap['feature']} {best_cheap['op']} {best_cheap['threshold']:.4f}")
        print(f"    Catches {best_cheap['losers_skipped']}/{total_losers} losers, costs {best_cheap['profit_cost_pct']:.1f}% profit")

    if aggressive_filters:
        best_agg = sorted(aggressive_filters, key=lambda x: x["quality"], reverse=True)[0]
        print(f"\n  Best aggressive filter (< 10% cost):")
        print(f"    {best_agg['feature']} {best_agg['op']} {best_agg['threshold']:.4f}")
        print(f"    Catches {best_agg['losers_skipped']}/{total_losers} losers, costs {best_agg['profit_cost_pct']:.1f}% profit")

    print(f"\n{'='*110}")


if __name__ == "__main__":
    main()
