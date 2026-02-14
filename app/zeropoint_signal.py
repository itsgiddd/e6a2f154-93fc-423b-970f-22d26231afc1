#!/usr/bin/env python3
"""
ZeroPoint PRO Signal Generator
================================

Live signal generator based on the ZeroPoint PRO ATR trailing stop flip strategy.
Designed to be called from the trading engine as an additional signal source.

Backtest results (180 days, H4):
  - TP1 Exit: PF 1.24, +1,226 pips, 61% win rate
  - Partial TP: PF 1.25, +1,212 pips, 80% win rate
  - Best symbols: USDCAD (PF 5.45), GBPJPY (PF 2.00), USDJPY (PF 1.67)

Integration:
  - Runs on H4 data (primary) with H1 confirmation
  - Generates BUY/SELL signals with SL/TP levels
  - Used as confluence signal alongside neural model
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Strategy parameters (from Pine Script, validated by backtest)
# ---------------------------------------------------------------------------

ATR_PERIOD = 10
ATR_MULTIPLIER = 3.0
TP1_MULT = 2.0
TP2_MULT = 3.5
TP3_MULT = 5.0
SWING_LOOKBACK = 10
SL_BUFFER_PCT = 0.001   # 0.1%
SL_ATR_MIN_MULT = 1.5

# V4 Profit Capture constants (optimized — PF 5.40, 97.9% WR on 1101 trades)
BE_TRIGGER_MULT = 0.5           # Move SL to BE after 0.5x ATR favorable move (was 1.0x)
BE_BUFFER_MULT = 0.15           # BE buffer: entry + 0.15x ATR
PROFIT_TRAIL_DISTANCE_MULT = 0.8  # Post-TP1 trail: 0.8x ATR behind max price (was 1.5x)
STALL_BARS = 6                  # Close at BE if TP1 not hit after 6 bars (was 12)
MICRO_TP_MULT = 0.8             # Take micro-partial at 0.8x ATR (was 1.0x)
MICRO_TP_PCT = 0.15             # Micro-partial = 15% of lot
TP1_MULT_AGG = 0.8              # V4 tighter TP1 (was 1.5x, originally 2.0x)
TP2_MULT_AGG = 2.0              # V4 tighter TP2 (was 3.0x, originally 3.5x)
TP3_MULT_AGG = 5.0              # V4 TP3 (unchanged)

# Symbols that are profitable on H4 (PF > 1.0 with TP1 exit, backtest validated)
ZEROPOINT_ENABLED_SYMBOLS = {
    "USDCAD": {"pf": 5.45, "tier": "S"},   # PF 5.45, 87% win
    "GBPJPY": {"pf": 2.00, "tier": "A"},   # PF 2.00, 67% win
    "USDJPY": {"pf": 1.67, "tier": "A"},   # PF 1.67, 70% win
    "AUDUSD": {"pf": 1.38, "tier": "B"},   # PF 1.38, 67% win
    "EURJPY": {"pf": 1.30, "tier": "B"},   # PF 1.30, 65% win
    "NZDUSD": {"pf": 1.22, "tier": "B"},   # PF 1.22, 62% win
    "GBPUSD": {"pf": 1.00, "tier": "C"},   # PF 1.00, 65% win (borderline)
}

# Minimum bars needed for signal generation
MIN_BARS = ATR_PERIOD + 5

# Number of ZeroPoint features per timeframe (used by trainer + live engine)
ZEROPOINT_FEATURES_PER_TF = 5
ZEROPOINT_TOTAL_FEATURES = ZEROPOINT_FEATURES_PER_TF * 3  # M15 + H1 + H4 = 15


# ---------------------------------------------------------------------------
# Shared utility functions (used by both trainer and live engine)
# ---------------------------------------------------------------------------

def compute_zeropoint_state(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Compute ZeroPoint ATR trailing stop state on any OHLCV DataFrame.

    Matches the Pine Script ZeroPoint PRO indicator EXACTLY:
      - ATR trailing stop with 4-branch position logic
      - lastSignalDir filter to prevent duplicate signals in same direction
      - Smart Structure SL using swing low/high with min ATR distance

    Output columns:
      - atr: ATR(10)
      - xATRTrailingStop: trailing stop level
      - pos: +1 (bullish) or -1 (bearish)
      - buy_signal / sell_signal: filtered signal (matches Pine buySignal/sellSignal)
      - raw_flip_buy / raw_flip_sell: raw flips before lastSignalDir filter
      - bars_since_flip: bars since last signal flip
      - bars_in_position: bars within current trend direction
      - atr_median_100: rolling 100-bar median ATR for normalization
      - smart_sl: Smart Structure SL level (for BUY: below price, for SELL: above price)

    Args:
        df: OHLCV DataFrame with columns: open, high, low, close, volume
            Works with M15, H1, H4, or any timeframe.

    Returns:
        Enriched DataFrame with ZeroPoint state columns, or None on error.
    """
    try:
        df = df.copy()

        # ---------- ATR ----------
        prev_close = df["close"].shift(1)
        tr = pd.concat([
            (df["high"] - df["low"]).abs(),
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1.0 / ATR_PERIOD, adjust=False).mean()
        n_loss = ATR_MULTIPLIER * atr

        close = df["close"].values
        high_arr = df["high"].values
        low_arr = df["low"].values
        n = len(close)
        n_loss_arr = n_loss.values
        atr_arr = atr.values

        stop = np.zeros(n, dtype=np.float64)
        pos = np.zeros(n, dtype=np.int32)
        raw_flip_buy = np.zeros(n, dtype=bool)   # flippedToBuy (raw, before filter)
        raw_flip_sell = np.zeros(n, dtype=bool)   # flippedToSell (raw, before filter)
        buy_sig = np.zeros(n, dtype=bool)         # buySignal (filtered by lastSignalDir)
        sell_sig = np.zeros(n, dtype=bool)        # sellSignal (filtered by lastSignalDir)
        smart_sl = np.full(n, np.nan, dtype=np.float64)

        first_valid = ATR_PERIOD
        if first_valid >= n:
            return None

        stop[first_valid] = close[first_valid]
        pos[first_valid] = 1

        # Pine Script: var int lastSignalDir = 0
        last_signal_dir = 0

        for i in range(first_valid + 1, n):
            nl = n_loss_arr[i]
            if np.isnan(nl):
                stop[i] = stop[i - 1]
                pos[i] = pos[i - 1]
                continue

            prev_stop = stop[i - 1]
            prev_c = close[i - 1]
            cur_c = close[i]

            # Pine Script: 4-branch trailing stop logic (EXACT match)
            if cur_c > prev_stop and prev_c > prev_stop:
                stop[i] = max(prev_stop, cur_c - nl)
                pos[i] = 1
            elif cur_c < prev_stop and prev_c < prev_stop:
                stop[i] = min(prev_stop, cur_c + nl)
                pos[i] = -1
            elif cur_c > prev_stop:
                stop[i] = cur_c - nl
                pos[i] = 1
            else:
                stop[i] = cur_c + nl
                pos[i] = -1

            # Pine Script: flippedToBuy = currentlyBullish and not currentlyBullish[1]
            flipped_to_buy = (pos[i] == 1 and pos[i - 1] != 1)
            flipped_to_sell = (pos[i] == -1 and pos[i - 1] != -1)
            raw_flip_buy[i] = flipped_to_buy
            raw_flip_sell[i] = flipped_to_sell

            # Pine Script: buySignal = flippedToBuy and lastSignalDir != 1
            # Pine Script: sellSignal = flippedToSell and lastSignalDir != -1
            is_buy_signal = flipped_to_buy and last_signal_dir != 1
            is_sell_signal = flipped_to_sell and last_signal_dir != -1
            buy_sig[i] = is_buy_signal
            sell_sig[i] = is_sell_signal

            # Pine Script: if buySignal -> lastSignalDir := 1
            # Pine Script: if sellSignal -> lastSignalDir := -1
            if is_buy_signal:
                last_signal_dir = 1
            if is_sell_signal:
                last_signal_dir = -1

            # ---------- Smart Structure SL (Pine Script exact) ----------
            atr_val = atr_arr[i]
            if np.isnan(atr_val) or atr_val <= 0:
                continue

            if is_buy_signal:
                # Pine: recentSwingLow = ta.lowest(low, 10)
                lookback_start = max(0, i - SWING_LOOKBACK + 1)
                recent_swing_low = np.min(low_arr[lookback_start:i + 1])

                # Pine: buffer = recentSwingLow * (input_slBuffer / 100)
                buffer = recent_swing_low * SL_BUFFER_PCT
                structural_sl = recent_swing_low - buffer
                min_sl = cur_c - (atr_val * SL_ATR_MIN_MULT)

                # Pine: if structuralSL > minSL -> slLevel := minSL else slLevel := structuralSL
                if structural_sl > min_sl:
                    smart_sl[i] = min_sl
                else:
                    smart_sl[i] = structural_sl

            elif is_sell_signal:
                # Pine: recentSwingHigh = ta.highest(high, 10)
                lookback_start = max(0, i - SWING_LOOKBACK + 1)
                recent_swing_high = np.max(high_arr[lookback_start:i + 1])

                # Pine: buffer = recentSwingHigh * (input_slBuffer / 100)
                buffer = recent_swing_high * SL_BUFFER_PCT
                structural_sl = recent_swing_high + buffer
                min_sl = cur_c + (atr_val * SL_ATR_MIN_MULT)

                # Pine: if structuralSL < minSL -> slLevel := minSL else slLevel := structuralSL
                if structural_sl < min_sl:
                    smart_sl[i] = min_sl
                else:
                    smart_sl[i] = structural_sl

        # ---------- Base columns ----------
        df["atr"] = atr
        df["xATRTrailingStop"] = stop
        df["pos"] = pos
        df["raw_flip_buy"] = raw_flip_buy
        df["raw_flip_sell"] = raw_flip_sell
        df["buy_signal"] = buy_sig
        df["sell_signal"] = sell_sig
        df["smart_sl"] = smart_sl

        # ---------- bars_since_flip ----------
        # Count bars since the most recent buy_signal or sell_signal
        bars_since_flip = np.zeros(n, dtype=np.int32)
        last_flip = -1
        for i in range(n):
            if buy_sig[i] or sell_sig[i]:
                last_flip = i
            bars_since_flip[i] = (i - last_flip) if last_flip >= 0 else n
        df["bars_since_flip"] = bars_since_flip

        # ---------- bars_in_position ----------
        # Count consecutive bars in the same position direction
        bars_in_pos = np.zeros(n, dtype=np.int32)
        for i in range(first_valid, n):
            if i == first_valid or pos[i] != pos[i - 1]:
                bars_in_pos[i] = 1
            else:
                bars_in_pos[i] = bars_in_pos[i - 1] + 1
        df["bars_in_position"] = bars_in_pos

        # ---------- atr_median_100 ----------
        # Rolling 100-bar median ATR for normalization (robust to outliers)
        df["atr_median_100"] = atr.rolling(100, min_periods=20).median()

        return df

    except Exception as e:
        logger.error(f"compute_zeropoint_state error: {e}")
        return None


def extract_zeropoint_bar_features(row) -> np.ndarray:
    """
    Extract 5 neural features from a single row of a ZeroPoint-enriched DataFrame.

    Features (all clipped to bounded ranges for stable training):
      [0] trailing_stop_position: +1.0 (bullish) or -1.0 (bearish)
      [1] stop_distance_ratio:    (close - stop) / ATR, clipped [-3, 3]
      [2] atr_normalized:         ATR / atr_median_100, clipped [0.3, 3.0]
      [3] signal_recency:         bars_since_flip / 20, clipped [0, 5]
      [4] position_duration:      bars_in_position / 50, clipped [0, 5]

    Args:
        row: A pandas Series or dict-like with ZeroPoint state columns.
             Must have: close, pos, xATRTrailingStop, atr, atr_median_100,
                        bars_since_flip, bars_in_position

    Returns:
        np.ndarray of shape (5,) with float32 features.
    """
    feats = np.zeros(ZEROPOINT_FEATURES_PER_TF, dtype=np.float32)

    try:
        pos_val = float(row.get("pos", 0) if hasattr(row, "get") else row["pos"])
        close_val = float(row.get("close", 0) if hasattr(row, "get") else row["close"])
        stop_val = float(row.get("xATRTrailingStop", 0) if hasattr(row, "get") else row["xATRTrailingStop"])
        atr_val = float(row.get("atr", 0) if hasattr(row, "get") else row["atr"])
        atr_med = float(row.get("atr_median_100", 0) if hasattr(row, "get") else row["atr_median_100"])
        bars_flip = float(row.get("bars_since_flip", 0) if hasattr(row, "get") else row["bars_since_flip"])
        bars_pos = float(row.get("bars_in_position", 0) if hasattr(row, "get") else row["bars_in_position"])

        # Handle NaN values
        if np.isnan(pos_val) or np.isnan(close_val) or np.isnan(atr_val):
            return feats

        # [0] trailing_stop_position: +1 or -1
        feats[0] = 1.0 if pos_val > 0 else -1.0

        # [1] stop_distance_ratio: how far price is from trailing stop, in ATR units
        if atr_val > 1e-12:
            feats[1] = np.clip((close_val - stop_val) / atr_val, -3.0, 3.0)

        # [2] atr_normalized: current ATR vs median (volatility regime)
        if not np.isnan(atr_med) and atr_med > 1e-12:
            feats[2] = np.clip(atr_val / atr_med, 0.3, 3.0)
        else:
            feats[2] = 1.0  # Neutral default

        # [3] signal_recency: how recently did the last flip occur
        if not np.isnan(bars_flip):
            feats[3] = np.clip(bars_flip / 20.0, 0.0, 5.0)

        # [4] position_duration: how long in the current trend
        if not np.isnan(bars_pos):
            feats[4] = np.clip(bars_pos / 50.0, 0.0, 5.0)

    except (KeyError, TypeError, ValueError) as e:
        logger.debug(f"extract_zeropoint_bar_features error: {e}")

    return feats


@dataclass
class ZeroPointSignal:
    """A ZeroPoint PRO trading signal."""
    symbol: str
    direction: str            # "BUY" or "SELL"
    entry_price: float        # Signal bar close
    stop_loss: float          # Smart Structure SL
    tp1: float                # 2.0x ATR
    tp2: float                # 3.5x ATR
    tp3: float                # 5.0x ATR
    atr_value: float          # Current ATR for reference
    confidence: float         # 0.0-1.0 based on symbol tier + conditions
    signal_time: datetime
    timeframe: str
    tier: str                 # S/A/B/C quality tier
    trailing_stop: float      # Current trailing stop level
    risk_reward: float        # TP1 distance / SL distance


class ZeroPointEngine:
    """
    Live ZeroPoint PRO signal generator.

    Maintains internal state per symbol to track trailing stop position
    and detect signal flips.
    """

    def __init__(self):
        # Per-symbol state: {symbol: {"stop": float, "pos": int, "last_bar_time": datetime}}
        self._state: Dict[str, dict] = {}
        self._last_signals: Dict[str, Optional[ZeroPointSignal]] = {}
        logger.info(f"ZeroPoint engine initialized. Enabled symbols: {list(ZEROPOINT_ENABLED_SYMBOLS.keys())}")

    def is_symbol_enabled(self, symbol: str) -> bool:
        """Check if ZeroPoint signals are enabled for this symbol."""
        norm = symbol.upper().replace(".", "").replace("#", "")
        return norm in ZEROPOINT_ENABLED_SYMBOLS

    def get_symbol_tier(self, symbol: str) -> str:
        """Get the performance tier for a symbol."""
        norm = symbol.upper().replace(".", "").replace("#", "")
        info = ZEROPOINT_ENABLED_SYMBOLS.get(norm, {})
        return info.get("tier", "D")

    def generate_signal(
        self,
        symbol: str,
        df_h4: pd.DataFrame,
        df_h1: Optional[pd.DataFrame] = None,
    ) -> Optional[ZeroPointSignal]:
        """
        Generate a ZeroPoint signal from H4 OHLCV data.

        Args:
            symbol: The trading symbol
            df_h4: H4 OHLCV DataFrame with columns: open, high, low, close, volume
            df_h1: Optional H1 data for confirmation

        Returns:
            ZeroPointSignal if a new signal is generated, None otherwise
        """
        norm = symbol.upper().replace(".", "").replace("#", "")

        if norm not in ZEROPOINT_ENABLED_SYMBOLS:
            return None

        if df_h4 is None or len(df_h4) < MIN_BARS:
            return None

        # Compute ATR trailing stop signals
        df = self._compute_signals(df_h4)
        if df is None:
            return None

        # Get the latest bar
        last_idx = len(df) - 1
        last_bar = df.iloc[last_idx]

        if pd.isna(last_bar.get("atr", np.nan)) or last_bar["atr"] <= 0:
            return None

        # Check for active ZP position (BULL=1, BEAR=-1)
        pos = int(last_bar.get("pos", 0))
        if pos == 0:
            return None

        direction = "BUY" if pos == 1 else "SELL"

        # Check if this is a fresh flip (for confidence scoring)
        buy_signal = bool(last_bar.get("buy_signal", False))
        sell_signal = bool(last_bar.get("sell_signal", False))
        is_fresh_flip = buy_signal or sell_signal
        if not is_fresh_flip and last_idx >= 1:
            prev_bar = df.iloc[last_idx - 1]
            is_fresh_flip = bool(prev_bar.get("buy_signal", False)) or bool(prev_bar.get("sell_signal", False))

        # Count bars in current position
        bars_in_pos = 1
        for idx in range(last_idx - 1, -1, -1):
            if int(df.iloc[idx].get("pos", 0)) == pos:
                bars_in_pos += 1
            else:
                break
        entry_price = last_bar["close"]
        atr_val = last_bar["atr"]

        # Smart Structure SL — use pre-computed smart_sl from signal bar
        # Find the most recent signal bar's smart_sl
        sl = None
        if "smart_sl" in df.columns:
            for idx in range(last_idx, -1, -1):
                sl_val = df.iloc[idx].get("smart_sl", np.nan)
                if not pd.isna(sl_val) and sl_val > 0:
                    sl = float(sl_val)
                    break

        if sl is None:
            # Fallback: compute Smart SL inline (same as Pine Script)
            sl = self._compute_smart_sl(df, last_idx, direction, atr_val)

        # For ongoing positions, use the trailing stop as SL (tighter than flip SL)
        trailing_stop_val = float(last_bar.get("xATRTrailingStop", 0))
        if not is_fresh_flip and trailing_stop_val > 0:
            # Use trailing stop directly — it's the live ATR stop
            sl = trailing_stop_val
            buffer = atr_val * SL_BUFFER_PCT
            if direction == "BUY":
                sl = sl - buffer
            else:
                sl = sl + buffer

        # TP levels — V4 OPTIMIZED (tighter TPs = faster profit capture)
        if direction == "BUY":
            tp1 = entry_price + atr_val * TP1_MULT_AGG
            tp2 = entry_price + atr_val * TP2_MULT_AGG
            tp3 = entry_price + atr_val * TP3_MULT_AGG
            sl_distance = entry_price - sl
            tp1_distance = tp1 - entry_price
        else:
            tp1 = entry_price - atr_val * TP1_MULT_AGG
            tp2 = entry_price - atr_val * TP2_MULT_AGG
            tp3 = entry_price - atr_val * TP3_MULT_AGG
            sl_distance = sl - entry_price
            tp1_distance = entry_price - tp1

        # If price already past TP (no room to profit), skip
        if direction == "BUY" and entry_price >= tp1:
            return None
        if direction == "SELL" and entry_price <= tp1:
            return None

        # Risk:Reward ratio
        rr = tp1_distance / sl_distance if sl_distance > 0 else 0.0

        # H1 confirmation (optional boost)
        h1_confirms = False
        if df_h1 is not None and len(df_h1) >= MIN_BARS:
            h1_df = self._compute_signals(df_h1)
            if h1_df is not None:
                h1_last = h1_df.iloc[-1]
                h1_pos = h1_last.get("pos", 0)
                if direction == "BUY" and h1_pos == 1:
                    h1_confirms = True
                elif direction == "SELL" and h1_pos == -1:
                    h1_confirms = True

        # Compute confidence — reduce for older positions
        tier = ZEROPOINT_ENABLED_SYMBOLS[norm]["tier"]
        base_pf = ZEROPOINT_ENABLED_SYMBOLS[norm]["pf"]

        confidence = self._compute_confidence(tier, rr, h1_confirms, base_pf)
        if not is_fresh_flip:
            # Ongoing position penalty: -2% per bar in position, max -15%
            age_penalty = min(bars_in_pos * 0.02, 0.15)
            confidence = max(0.40, confidence - age_penalty)

        signal = ZeroPointSignal(
            symbol=symbol,
            direction=direction,
            entry_price=entry_price,
            stop_loss=sl,
            tp1=tp1,
            tp2=tp2,
            tp3=tp3,
            atr_value=atr_val,
            confidence=confidence,
            signal_time=df.index[last_idx] if hasattr(df.index[last_idx], 'strftime') else datetime.now(),
            timeframe="H4",
            tier=tier,
            trailing_stop=last_bar.get("xATRTrailingStop", entry_price),
            risk_reward=rr,
        )

        self._last_signals[norm] = signal

        logger.info(
            f"ZeroPoint {direction} signal: {symbol} @ {entry_price:.5f} "
            f"SL={sl:.5f} TP1={tp1:.5f} R:R={rr:.2f} "
            f"Conf={confidence:.2f} Tier={tier} H1={'Y' if h1_confirms else 'N'}"
        )

        return signal

    def get_current_position(self, symbol: str, df_h4: pd.DataFrame) -> Optional[int]:
        """
        Get the current trailing stop position direction.
        Returns: 1 (bullish), -1 (bearish), 0 (unknown), None (error)
        """
        if df_h4 is None or len(df_h4) < MIN_BARS:
            return None
        df = self._compute_signals(df_h4)
        if df is None:
            return None
        return int(df.iloc[-1].get("pos", 0))

    def get_trailing_stop(self, symbol: str, df_h4: pd.DataFrame) -> Optional[float]:
        """Get the current trailing stop level for a symbol."""
        if df_h4 is None or len(df_h4) < MIN_BARS:
            return None
        df = self._compute_signals(df_h4)
        if df is None:
            return None
        val = df.iloc[-1].get("xATRTrailingStop", None)
        return float(val) if val is not None and not pd.isna(val) else None

    def get_last_signal(self, symbol: str) -> Optional[ZeroPointSignal]:
        """Get the most recent signal for a symbol."""
        norm = symbol.upper().replace(".", "").replace("#", "")
        return self._last_signals.get(norm)

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _compute_signals(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Compute ATR trailing stop, position, and signals on OHLCV data.

        Matches Pine Script ZeroPoint PRO exactly:
        - 4-branch trailing stop logic
        - lastSignalDir filter to prevent duplicate signals in same direction
        """
        try:
            df = df.copy()

            # ATR
            prev_close = df["close"].shift(1)
            tr = pd.concat([
                (df["high"] - df["low"]).abs(),
                (df["high"] - prev_close).abs(),
                (df["low"] - prev_close).abs(),
            ], axis=1).max(axis=1)
            atr = tr.ewm(alpha=1.0 / ATR_PERIOD, adjust=False).mean()
            n_loss = ATR_MULTIPLIER * atr

            close = df["close"].values
            high_arr = df["high"].values
            low_arr = df["low"].values
            n = len(close)
            n_loss_arr = n_loss.values
            atr_arr = atr.values

            stop = np.zeros(n, dtype=np.float64)
            pos = np.zeros(n, dtype=np.int32)
            buy_sig = np.zeros(n, dtype=bool)
            sell_sig = np.zeros(n, dtype=bool)
            smart_sl = np.full(n, np.nan, dtype=np.float64)

            first_valid = ATR_PERIOD
            if first_valid >= n:
                return None

            stop[first_valid] = close[first_valid]
            pos[first_valid] = 1

            # Pine Script: var int lastSignalDir = 0
            last_signal_dir = 0

            for i in range(first_valid + 1, n):
                nl = n_loss_arr[i]
                if np.isnan(nl):
                    stop[i] = stop[i - 1]
                    pos[i] = pos[i - 1]
                    continue

                prev_stop = stop[i - 1]
                prev_c = close[i - 1]
                cur_c = close[i]

                if cur_c > prev_stop and prev_c > prev_stop:
                    stop[i] = max(prev_stop, cur_c - nl)
                    pos[i] = 1
                elif cur_c < prev_stop and prev_c < prev_stop:
                    stop[i] = min(prev_stop, cur_c + nl)
                    pos[i] = -1
                elif cur_c > prev_stop:
                    stop[i] = cur_c - nl
                    pos[i] = 1
                else:
                    stop[i] = cur_c + nl
                    pos[i] = -1

                # Pine Script: flippedToBuy/flippedToSell + lastSignalDir filter
                flipped_to_buy = (pos[i] == 1 and pos[i - 1] != 1)
                flipped_to_sell = (pos[i] == -1 and pos[i - 1] != -1)

                is_buy_signal = flipped_to_buy and last_signal_dir != 1
                is_sell_signal = flipped_to_sell and last_signal_dir != -1
                buy_sig[i] = is_buy_signal
                sell_sig[i] = is_sell_signal

                if is_buy_signal:
                    last_signal_dir = 1
                if is_sell_signal:
                    last_signal_dir = -1

                # Smart Structure SL on signal bars
                atr_val = atr_arr[i]
                if not np.isnan(atr_val) and atr_val > 0:
                    if is_buy_signal:
                        lookback_start = max(0, i - SWING_LOOKBACK + 1)
                        recent_swing_low = np.min(low_arr[lookback_start:i + 1])
                        buffer = recent_swing_low * SL_BUFFER_PCT
                        structural_sl = recent_swing_low - buffer
                        min_sl = cur_c - (atr_val * SL_ATR_MIN_MULT)
                        smart_sl[i] = min_sl if structural_sl > min_sl else structural_sl
                    elif is_sell_signal:
                        lookback_start = max(0, i - SWING_LOOKBACK + 1)
                        recent_swing_high = np.max(high_arr[lookback_start:i + 1])
                        buffer = recent_swing_high * SL_BUFFER_PCT
                        structural_sl = recent_swing_high + buffer
                        min_sl = cur_c + (atr_val * SL_ATR_MIN_MULT)
                        smart_sl[i] = min_sl if structural_sl < min_sl else structural_sl

            df["atr"] = atr
            df["xATRTrailingStop"] = stop
            df["pos"] = pos
            df["buy_signal"] = buy_sig
            df["sell_signal"] = sell_sig
            df["smart_sl"] = smart_sl
            return df

        except Exception as e:
            logger.error(f"ZeroPoint signal computation error: {e}")
            return None

    def _compute_smart_sl(
        self, df: pd.DataFrame, bar_idx: int, direction: str, atr_val: float
    ) -> float:
        """Smart Structure SL from Pine Script."""
        lookback_start = max(0, bar_idx - SWING_LOOKBACK + 1)
        cur_close = df["close"].iloc[bar_idx]

        if direction == "BUY":
            recent_swing_low = df["low"].iloc[lookback_start:bar_idx + 1].min()
            buffer = recent_swing_low * SL_BUFFER_PCT
            structural_sl = recent_swing_low - buffer
            atr_minimum_sl = cur_close - atr_val * SL_ATR_MIN_MULT
            return min(structural_sl, atr_minimum_sl)
        else:
            recent_swing_high = df["high"].iloc[lookback_start:bar_idx + 1].max()
            buffer = recent_swing_high * SL_BUFFER_PCT
            structural_sl = recent_swing_high + buffer
            atr_maximum_sl = cur_close + atr_val * SL_ATR_MIN_MULT
            return max(structural_sl, atr_maximum_sl)

    def _compute_confidence(
        self, tier: str, rr: float, h1_confirms: bool, base_pf: float
    ) -> float:
        """
        Compute signal confidence 0.0-1.0.

        Factors:
        - Symbol tier (S=0.85, A=0.75, B=0.65, C=0.55)
        - Risk:Reward ratio bonus
        - H1 confirmation bonus
        - Base profit factor scaling
        """
        tier_base = {"S": 0.85, "A": 0.75, "B": 0.65, "C": 0.55, "D": 0.40}
        conf = tier_base.get(tier, 0.40)

        # R:R bonus (good R:R = +0.05, great R:R = +0.10)
        if rr >= 2.0:
            conf += 0.10
        elif rr >= 1.5:
            conf += 0.05

        # H1 confirmation bonus
        if h1_confirms:
            conf += 0.08

        # PF scaling (PF > 3 = +0.05)
        if base_pf >= 3.0:
            conf += 0.05

        return min(conf, 0.98)

    def get_enabled_symbols_info(self) -> Dict[str, dict]:
        """Return info about all enabled symbols and their tiers."""
        return dict(ZEROPOINT_ENABLED_SYMBOLS)
