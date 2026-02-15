#!/usr/bin/env python3
"""
Velocity 4 — ZeroPoint Pro Trading App
Modern web-based UI powered by QWebEngineView + QWebChannel.
Backend: ScanEngine (signal detection) + TPManager (V4 profit capture).
"""

import sys, os, threading, json, time as _time, logging, math
import traceback as _tb
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure project root is on sys.path so "from app.X" / "from models.X" etc. work
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from PySide6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout
from PySide6.QtCore import Qt, QTimer, Signal, QObject, Slot, QUrl
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWebEngineCore import QWebEngineSettings
from PySide6.QtWebChannel import QWebChannel

import MetaTrader5 as mt5

from app.zeropoint_signal import compute_zeropoint_state, ATR_PERIOD, ATR_MULTIPLIER
from app.zeropoint_signal import TP1_MULT, TP2_MULT, TP3_MULT  # baseline (unused)
from app.zeropoint_signal import (
    TP1_MULT_AGG, TP2_MULT_AGG, TP3_MULT_AGG,       # V4 optimized TPs
    BE_TRIGGER_MULT, BE_BUFFER_MULT,                   # Early breakeven
    PROFIT_TRAIL_DISTANCE_MULT,                        # Post-TP1 trailing
    STALL_BARS,                                        # Stall exit
    MICRO_TP_MULT, MICRO_TP_PCT,                       # Micro-partial
)

# ---------------------------------------------------------------------------
# Crash handler
# ---------------------------------------------------------------------------
def _global_exception_hook(exc_type, exc_value, exc_tb):
    crash_msg = "".join(_tb.format_exception(exc_type, exc_value, exc_tb))
    try:
        with open("crash.log", "a") as f:
            f.write(f"\n{'='*60}\n{datetime.now()}\n{crash_msg}\n")
    except Exception:
        pass
    sys.__excepthook__(exc_type, exc_value, exc_tb)
sys.excepthook = _global_exception_hook

LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(LOG_DIR, "aci.log"), encoding="utf-8"),
    ],
)
log = logging.getLogger("aci")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ALL_PAIRS = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "NZDUSD", "EURJPY", "GBPJPY"]

TF_MAP = {
    "M1": mt5.TIMEFRAME_M1,
    "M5": mt5.TIMEFRAME_M5,
    "M15": mt5.TIMEFRAME_M15,
    "M30": mt5.TIMEFRAME_M30,
    "H1": mt5.TIMEFRAME_H1,
    "H4": mt5.TIMEFRAME_H4,
    "D1": mt5.TIMEFRAME_D1,
    "W1": mt5.TIMEFRAME_W1,
}
TIMEFRAMES = list(TF_MAP.keys())

MAGIC_NUMBER = 234567
SETTINGS_PATH = os.path.join(os.path.dirname(__file__), "aci_settings.json")

# V4 partial close splits (1/3 at TP1, 1/3 at TP2, close rest at TP3)
TP1_CLOSE_PCT = 0.33
TP2_CLOSE_PCT = 0.33   # of original
TP3_CLOSE_PCT = 1.00   # close remaining
MICRO_CLOSE_PCT = MICRO_TP_PCT  # 15% micro-partial at 0.8x ATR

# Margin safety — require at least this % free margin after trade
MIN_MARGIN_LEVEL_PCT = 150   # 150% margin level = safe zone

# HTML UI file path
HTML_UI_PATH = os.path.join(os.path.dirname(__file__), "v4_ui.html")




# ---------------------------------------------------------------------------
# Scan Engine (background thread)
# ---------------------------------------------------------------------------
class ScanEngine(QObject):
    """Background ZeroPoint scanner: fetches MT5 data, detects signals, places trades."""

    log_message = Signal(str)
    signal_detected = Signal(str, str, float, float, float, float, float, float)
    # symbol, direction, entry, sl, tp1, tp2, tp3, atr
    scan_complete = Signal(str, object)
    # symbol, enriched DataFrame

    MAX_SIGNAL_AGE = 6   # max bars old a signal can be to still enter

    def __init__(self):
        super().__init__()
        self._running = False
        self._auto_trade = False     # only place trades when True
        self._last_signal_dir = {}   # {symbol: 1 or -1}
        self._last_bar_time = {}     # {symbol: datetime} — dedup bar
        self._entered_signals = {}   # {symbol: bar_time} — signals we've already traded
        # V4 defaults (overridden by set_v4_params from UI)
        self._v4 = {
            "tp1_mult": TP1_MULT_AGG, "tp2_mult": TP2_MULT_AGG, "tp3_mult": TP3_MULT_AGG,
            "be_trigger": BE_TRIGGER_MULT, "be_buffer": BE_BUFFER_MULT,
            "micro_tp": MICRO_TP_MULT, "micro_pct": MICRO_TP_PCT,
            "stall_bars": STALL_BARS, "trail_dist": PROFIT_TRAIL_DISTANCE_MULT,
        }

    def set_v4_params(self, params: dict):
        """Update V4 profit capture parameters from UI settings."""
        self._v4.update(params)

    # ── Trade helpers (ported from webhook_bridge.py) ──

    def _resolve_symbol(self, ticker: str):
        candidates = [ticker, ticker + ".raw", ticker + "m",
                      ticker + ".a", ticker + ".e", ticker[:6]]
        for c in candidates:
            info = mt5.symbol_info(c)
            if info is not None:
                mt5.symbol_select(c, True)
                return c
        self.log_message.emit(f"Cannot resolve symbol: {ticker}")
        return None

    def _check_existing_position(self, symbol: str) -> bool:
        positions = mt5.positions_get()
        if not positions:
            return False
        norm = symbol.upper().replace(".", "").replace("#", "")
        for pos in positions:
            pos_norm = pos.symbol.upper().replace(".", "").replace("#", "")
            if pos_norm == norm:
                return True
        return False

    def _calc_lot_size(self, entry, sl, sym_info, risk_pct, default_lot):
        try:
            acct = mt5.account_info()
            if acct is None:
                return default_lot

            balance = acct.balance
            risk_amount = balance * risk_pct

            point = sym_info.point
            tick_size = sym_info.trade_tick_size or point
            tick_value = sym_info.trade_tick_value
            if tick_value <= 0:
                tick_value = sym_info.trade_contract_size * tick_size

            sl_distance = abs(entry - sl)
            sl_ticks = sl_distance / tick_size if tick_size > 0 else 0
            loss_per_lot = sl_ticks * tick_value

            if loss_per_lot <= 0:
                return default_lot

            lot = risk_amount / loss_per_lot
            vol_step = sym_info.volume_step
            lot = round(lot / vol_step) * vol_step
            lot = max(sym_info.volume_min, min(lot, sym_info.volume_max))

            cap_table = [
                (500, 0.10), (1000, 0.20), (3000, 0.50),
                (5000, 1.00), (10000, 2.00), (50000, 5.00),
                (float('inf'), 10.00),
            ]
            for threshold, max_lot in cap_table:
                if balance <= threshold:
                    lot = min(lot, max_lot)
                    break

            self.log_message.emit(f"  Lot: balance=${balance:.0f} risk=${risk_amount:.0f} lot={lot:.2f}")
            return lot
        except Exception as e:
            self.log_message.emit(f"  Lot calc error: {e}, using {default_lot}")
            return default_lot

    def _place_trade(self, symbol_resolved, action, lot, sl, tp1, sym_info):
        try:
            if action == "BUY":
                order_type = mt5.ORDER_TYPE_BUY
                price = sym_info.ask
            else:
                order_type = mt5.ORDER_TYPE_SELL
                price = sym_info.bid

            digits = sym_info.digits
            sl_r = round(sl, digits)
            tp_r = round(tp1, digits)

            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": sym_info.name,
                "volume": lot,
                "type": order_type,
                "price": price,
                "sl": sl_r,
                "tp": tp_r,
                "deviation": 20,
                "magic": MAGIC_NUMBER,
                "comment": f"ACi-ZP-{action}",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_FOK,
            }

            for fill_mode in [mt5.ORDER_FILLING_FOK, mt5.ORDER_FILLING_IOC, mt5.ORDER_FILLING_RETURN]:
                request["type_filling"] = fill_mode
                result = mt5.order_send(request)
                if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                    self.log_message.emit(
                        f"  TRADE PLACED: {action} {sym_info.name} "
                        f"{lot:.2f}L @ {price:.5f} | SL={sl_r} TP={tp_r}"
                    )
                    return True

            rc = result.retcode if result else "?"
            self.log_message.emit(f"  Trade FAILED: {sym_info.name} retcode={rc}")
            return False
        except Exception as e:
            self.log_message.emit(f"  Trade error: {e}")
            return False

    # ── Main scan loop ──

    def start_scanning(self, symbols, tf_key, poll_interval, risk_pct, default_lot, max_trades):
        self._symbols = symbols
        self._tf_key = tf_key
        self._tf_mt5 = TF_MAP.get(tf_key, mt5.TIMEFRAME_H4)
        self._poll = poll_interval
        self._risk_pct = risk_pct
        self._default_lot = default_lot
        self._max_trades = max_trades
        self._running = True

        t = threading.Thread(target=self._scan_loop, daemon=True)
        t.start()

    def stop(self):
        self._running = False

    def _execute_trade_intent(self, intent):
        """Execute a single trade intent. Called from ThreadPoolExecutor."""
        try:
            success = self._place_trade(
                intent["resolved"], intent["direction"], intent["lot"],
                intent["sl"], intent["tp1"], intent["sym_info"])
            if success:
                self._entered_signals[intent["symbol"]] = intent["bar_time"]
                self.log_message.emit(
                    f"[{intent['symbol']}] Trade executed: {intent['direction']} "
                    f"{intent['lot']:.2f}L{intent['age_str']}")
            return success
        except Exception as e:
            self.log_message.emit(f"[{intent['symbol']}] Async trade error: {e}")
            return False

    def _scan_loop(self):
        self.log_message.emit(f"Scanner started | {self._tf_key} | {len(self._symbols)} symbols | poll={self._poll}s")
        first_scan = True

        while self._running:
            scanned = 0
            trade_intents = []
            for symbol in self._symbols:
                if not self._running:
                    break
                try:
                    intent = self._scan_symbol(symbol)
                    if intent is not None:
                        trade_intents.append(intent)
                    scanned += 1
                except Exception as e:
                    self.log_message.emit(f"[{symbol}] Scan error: {e}")

            # Execute all trade intents concurrently via ThreadPoolExecutor
            if trade_intents:
                n = len(trade_intents)
                syms = ", ".join(t["symbol"] for t in trade_intents)
                self.log_message.emit(f"Batch executing {n} trades: {syms}")
                with ThreadPoolExecutor(max_workers=min(n, 4)) as executor:
                    futures = {executor.submit(self._execute_trade_intent, t): t
                               for t in trade_intents}
                    for fut in as_completed(futures):
                        try:
                            fut.result()
                        except Exception as e:
                            t = futures[fut]
                            self.log_message.emit(f"[{t['symbol']}] Batch exec error: {e}")

            if first_scan:
                self.log_message.emit(f"Scan complete: {scanned}/{len(self._symbols)} pairs checked")
                first_scan = False

            # Interruptible sleep
            for _ in range(self._poll):
                if not self._running:
                    break
                _time.sleep(1)

        self.log_message.emit("Scanner stopped.")

    def _scan_symbol(self, symbol):
        resolved = self._resolve_symbol(symbol)
        if resolved is None:
            return

        # Fetch OHLCV — 1000 bars for full historical view
        rates = mt5.copy_rates_from_pos(resolved, self._tf_mt5, 0, 1000)
        if rates is None or len(rates) < 20:
            return

        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        df.rename(columns={"tick_volume": "volume"}, inplace=True)

        # Compute ZeroPoint state
        df_zp = compute_zeropoint_state(df)
        if df_zp is None or len(df_zp) < 2:
            return

        # Emit data for chart (always, for every pair)
        self.scan_complete.emit(symbol, df_zp)

        # Current ZP direction from last bar
        last_pos = int(df_zp.iloc[-2]["pos"]) if len(df_zp) > 1 else 0
        pos_str = "BULL" if last_pos == 1 else "BEAR"

        # ── Find most recent signal within MAX_SIGNAL_AGE confirmed bars ──
        # iloc[-1] is the forming (current) bar, so confirmed bars are iloc[-2], iloc[-3], ...
        n = len(df_zp)
        signal_bar = None
        signal_age = 0   # how many bars ago (0 = last confirmed bar)

        for lookback in range(2, 2 + self.MAX_SIGNAL_AGE):
            idx = n - lookback
            if idx < 0:
                break
            row = df_zp.iloc[idx]
            if row.get("buy_signal", False) or row.get("sell_signal", False):
                signal_bar = row
                signal_age = lookback - 2  # 0 = most recent confirmed bar
                break

        if signal_bar is None:
            return

        bar_time = signal_bar["time"]

        # Already traded this exact signal?
        if self._entered_signals.get(symbol) == bar_time:
            return

        buy_sig = bool(signal_bar.get("buy_signal", False))
        sell_sig = bool(signal_bar.get("sell_signal", False))
        direction = "BUY" if buy_sig else "SELL"
        entry_price = float(signal_bar["close"])  # original signal price
        sl_val = float(signal_bar["smart_sl"]) if not np.isnan(signal_bar["smart_sl"]) else None
        atr_val = float(signal_bar["atr"]) if not np.isnan(signal_bar["atr"]) else None

        if sl_val is None or atr_val is None:
            return

        # V4 optimized TP levels (read from UI settings)
        _tp1m = self._v4["tp1_mult"]
        _tp2m = self._v4["tp2_mult"]
        _tp3m = self._v4["tp3_mult"]
        if direction == "BUY":
            tp1 = entry_price + atr_val * _tp1m
            tp2 = entry_price + atr_val * _tp2m
            tp3 = entry_price + atr_val * _tp3m
        else:
            tp1 = entry_price - atr_val * _tp1m
            tp2 = entry_price - atr_val * _tp2m
            tp3 = entry_price - atr_val * _tp3m

        # R:R check
        sl_dist = abs(entry_price - sl_val)
        tp_dist = abs(tp1 - entry_price)
        rr = tp_dist / sl_dist if sl_dist > 0 else 0
        if rr < 0.3:
            self.log_message.emit(f"[{symbol}] {direction} R:R too low ({rr:.2f}), skipping")
            return

        # Staleness check — current price vs signal price
        # If price has drifted more than 50% of SL distance, skip (too stale)
        sym_info = mt5.symbol_info(resolved)
        if sym_info is None:
            return
        current_price = sym_info.ask if direction == "BUY" else sym_info.bid
        drift = abs(current_price - entry_price)
        if sl_dist > 0 and drift > sl_dist * 0.50:
            if signal_age == 0:
                self.log_message.emit(f"[{symbol}] {direction} price drifted {drift:.5f} (>{sl_dist*0.50:.5f}), too stale")
            return

        # Check signal hasn't already been invalidated (price past SL)
        if direction == "BUY" and current_price < sl_val:
            return
        if direction == "SELL" and current_price > sl_val:
            return

        age_str = f" ({signal_age} bars ago)" if signal_age > 0 else ""
        self.log_message.emit(f"[{symbol}] ZP {direction}{age_str} | entry={entry_price:.5f} SL={sl_val:.5f} "
                              f"TP1={tp1:.5f} R:R={rr:.2f} ATR={atr_val:.5f}")

        # Always emit signal for chart drawing (even when trading is off)
        self.signal_detected.emit(symbol, direction, entry_price, sl_val, tp1, tp2, tp3, atr_val)

        # Only place trade if auto-trade is ON
        if not self._auto_trade:
            self.log_message.emit(f"[{symbol}] Trading OFF — signal detected but not executing")
            self._entered_signals[symbol] = bar_time  # mark so we don't spam logs
            return

        # Check existing position
        if self._check_existing_position(symbol):
            self.log_message.emit(f"[{symbol}] Already have position, skipping")
            self._entered_signals[symbol] = bar_time
            return

        # Check max concurrent
        positions = mt5.positions_get()
        if positions and len(positions) >= self._max_trades:
            self.log_message.emit(f"Max concurrent ({self._max_trades}) reached, skipping")
            return

        # Margin safety check
        acct = mt5.account_info()
        if acct and acct.margin_level > 0 and acct.margin_level < MIN_MARGIN_LEVEL_PCT:
            self.log_message.emit(f"[{symbol}] Margin level {acct.margin_level:.0f}% < {MIN_MARGIN_LEVEL_PCT}%, skipping")
            return

        # Check free margin would stay positive
        if acct and acct.margin_free is not None and acct.margin_free < acct.balance * 0.20:
            self.log_message.emit(f"[{symbol}] Free margin ${acct.margin_free:.0f} too low (<20% of balance), skipping")
            return

        # Place trade at CURRENT price (not signal price)
        lot = self._calc_lot_size(current_price, sl_val, sym_info, self._risk_pct, self._default_lot)
        if lot <= 0:
            return None

        # Return trade intent for async batch execution
        return {
            "symbol": symbol, "resolved": resolved, "direction": direction,
            "lot": lot, "sl": sl_val, "tp1": tp1, "sym_info": sym_info,
            "bar_time": bar_time, "age_str": age_str,
        }


# ---------------------------------------------------------------------------
# V4 TP Manager — Full 5-layer profit capture system
# Layers: Micro-Partial | Early BE | Tighter TPs | Post-TP1 Trail | Stall Exit
# ---------------------------------------------------------------------------
STATE_FILE = os.path.join(os.path.dirname(__file__), "tp_state.json")


class TPManager(QObject):
    """V4 Trade Manager: 5-layer profit capture matching backtested system."""

    log_message = Signal(str)

    def __init__(self):
        super().__init__()
        self._running = False
        self._trade_info = {}   # {ticket: {direction, entry, sl, atr, tp1-3, original_lot}}
        self._tp_state = {}     # {ticket: {tp1, tp2, tp3, micro_tp, be_activated, stall_be, ...}}
        # V4 defaults (overridden by set_v4_params from UI)
        self._v4 = {
            "tp1_mult": TP1_MULT_AGG, "tp2_mult": TP2_MULT_AGG, "tp3_mult": TP3_MULT_AGG,
            "be_trigger": BE_TRIGGER_MULT, "be_buffer": BE_BUFFER_MULT,
            "micro_tp": MICRO_TP_MULT, "micro_pct": MICRO_TP_PCT,
            "stall_bars": STALL_BARS, "trail_dist": PROFIT_TRAIL_DISTANCE_MULT,
        }
        self._load_state()

    def set_v4_params(self, params: dict):
        """Update V4 profit capture parameters from UI settings."""
        self._v4.update(params)

    # ── State persistence ──

    def _save_state(self):
        """Persist trade_info + tp_state to JSON for crash recovery."""
        try:
            data = {
                "trade_info": {str(k): v for k, v in self._trade_info.items()},
                "tp_state": {str(k): v for k, v in self._tp_state.items()},
            }
            tmp = STATE_FILE + ".tmp"
            with open(tmp, "w") as f:
                json.dump(data, f)
            os.replace(tmp, STATE_FILE)  # atomic on Windows
        except Exception as e:
            log.warning(f"State save error: {e}")

    def _load_state(self):
        """Reload persisted state on startup, pruning tickets no longer open in MT5."""
        if not os.path.exists(STATE_FILE):
            return
        try:
            with open(STATE_FILE, "r") as f:
                data = json.load(f)
            loaded_info = {int(k): v for k, v in data.get("trade_info", {}).items()}
            loaded_state = {int(k): v for k, v in data.get("tp_state", {}).items()}

            # Prune tickets that are no longer open in MT5
            if mt5.terminal_info() is not None:
                positions = mt5.positions_get()
                open_tickets = {p.ticket for p in positions} if positions else set()
                loaded_info = {k: v for k, v in loaded_info.items() if k in open_tickets}
                loaded_state = {k: v for k, v in loaded_state.items() if k in open_tickets}

            if loaded_info:
                self._trade_info.update(loaded_info)
                self._tp_state.update(loaded_state)
                log.info(f"Restored state for {len(loaded_info)} positions from tp_state.json")
        except Exception as e:
            log.warning(f"State load error: {e}")

    def register_trade(self, ticket, direction, entry, sl, tp1, tp2, tp3, atr_val, original_lot):
        """Register a new trade for V4 profit capture management."""
        self._trade_info[ticket] = {
            "direction": direction,
            "entry": entry,
            "sl": sl,
            "atr": atr_val,
            "tp1": tp1,
            "tp2": tp2,
            "tp3": tp3,
            "original_lot": original_lot,
        }
        self._tp_state[ticket] = {
            "tp1": False, "tp2": False, "tp3": False,
            "micro_tp": False,           # Micro-partial at micro_tp * ATR
            "be_activated": False,       # Early BE at be_trigger * ATR
            "stall_be": False,           # Stall exit after stall_bars checks
            "profit_trail_sl": None,     # Post-TP1 trailing SL
            "max_favorable": entry,      # Best price seen
            "checks": 0,                 # ~bars since entry (each check = 2s, H4 = 14400s)
        }
        v = self._v4
        self.log_message.emit(
            f"  V4 Manager: #{ticket} | TP1={tp1:.5f} TP2={tp2:.5f} TP3={tp3:.5f} | "
            f"BE@{v['be_trigger']}x Stall@{v['stall_bars']}bars Trail@{v['trail_dist']}x"
        )
        self._save_state()

    def start(self):
        self._running = True
        t = threading.Thread(target=self._monitor_loop, daemon=True)
        t.start()
        # Fast tick-polling thread for time-critical BE/Micro-TP triggers
        t2 = threading.Thread(target=self._tick_monitor_loop, daemon=True)
        t2.start()

    def stop(self):
        self._running = False

    def _monitor_loop(self):
        while self._running:
            try:
                self._check_positions()
            except Exception as e:
                self.log_message.emit(f"V4 Manager error: {e}")
            _time.sleep(2)

    def _tick_monitor_loop(self):
        """Fast tick-level monitor (0.5s) for Layer 1 Micro-TP and Layer 2 Early BE.

        Uses mt5.copy_ticks_from() to capture intra-bar price spikes that the
        2-second main loop might miss. Only checks positions that haven't yet
        triggered BE — once BE is activated, the main loop handles trailing.
        """
        while self._running:
            try:
                self._check_tick_triggers()
            except Exception as e:
                self.log_message.emit(f"Tick monitor error: {e}")
            _time.sleep(0.5)

    def _check_tick_triggers(self):
        """Fast tick check — only Micro-TP and Early BE on positions that need it."""
        positions = mt5.positions_get()
        if not positions:
            return

        for pos in positions:
            if pos.magic != MAGIC_NUMBER:
                continue
            ticket = pos.ticket
            if ticket not in self._trade_info:
                continue

            state = self._tp_state[ticket]
            # Skip if both micro-TP and BE are already activated
            if state["micro_tp"] and state["be_activated"]:
                continue

            info = self._trade_info[ticket]
            direction = info["direction"]
            entry = info["entry"]
            atr = info["atr"]
            is_buy = direction == "BUY"

            sym_info = mt5.symbol_info(pos.symbol)
            if not sym_info:
                continue

            # Get recent ticks (last 2 seconds) for high/low price discovery
            now = datetime.now()
            ticks = mt5.copy_ticks_from(pos.symbol, now, 100, mt5.COPY_TICKS_ALL)
            if ticks is not None and len(ticks) > 0:
                # Find the extreme favorable price across recent ticks
                if is_buy:
                    tick_best = max(t[1] for t in ticks)  # highest bid
                else:
                    tick_best = min(t[2] for t in ticks)  # lowest ask
            else:
                # Fallback to current price
                tick_best = sym_info.bid if is_buy else sym_info.ask

            # Update max_favorable from ticks
            if is_buy and tick_best > state["max_favorable"]:
                state["max_favorable"] = tick_best
            elif not is_buy and tick_best < state["max_favorable"]:
                state["max_favorable"] = tick_best

            max_profit_dist = abs(state["max_favorable"] - entry)
            _v = self._v4

            # ── Fast Micro-TP check ──
            if not state["micro_tp"] and not state["tp1"]:
                micro_level = entry + (_v["micro_tp"] * atr if is_buy else -_v["micro_tp"] * atr)
                micro_hit = (is_buy and tick_best >= micro_level) or (not is_buy and tick_best <= micro_level)
                if micro_hit:
                    original_lot = info["original_lot"]
                    close_lot = round(original_lot * _v["micro_pct"], 2)
                    vol_step = sym_info.volume_step
                    vol_min = sym_info.volume_min
                    close_lot = max(vol_min, round(close_lot / vol_step) * vol_step)
                    close_lot = min(close_lot, pos.volume)
                    if close_lot >= vol_min:
                        ok = self._partial_close(pos, close_lot, sym_info)
                        if ok:
                            state["micro_tp"] = True
                            self.log_message.emit(
                                f"  TICK MICRO-TP: {pos.symbol} closed {close_lot:.2f}L "
                                f"(tick={tick_best:.5f})")

            # ── Fast Early BE check ──
            if not state["be_activated"]:
                if max_profit_dist >= _v["be_trigger"] * atr:
                    be_buffer_price = _v["be_buffer"] * atr
                    be_level = entry + (be_buffer_price if is_buy else -be_buffer_price)
                    should_move = (is_buy and be_level > pos.sl) or (not is_buy and be_level < pos.sl)
                    if should_move and self._is_spread_safe(sym_info, be_buffer_price):
                        ok = self._move_sl(pos, be_level, sym_info)
                        if ok:
                            state["be_activated"] = True
                            self.log_message.emit(
                                f"  TICK BE: {pos.symbol} SL -> {be_level:.5f} "
                                f"(tick peak={tick_best:.5f})")

    def _check_positions(self):
        positions = mt5.positions_get()
        if not positions:
            return

        # Clean up closed trades
        open_tickets = {p.ticket for p in positions}
        closed = [t for t in self._trade_info if t not in open_tickets]
        if closed:
            for t in closed:
                self._trade_info.pop(t, None)
                self._tp_state.pop(t, None)
            self._save_state()

        for pos in positions:
            if pos.magic != MAGIC_NUMBER:
                continue
            ticket = pos.ticket
            if ticket not in self._trade_info:
                continue

            info = self._trade_info[ticket]
            state = self._tp_state[ticket]
            current = pos.price_current
            direction = info["direction"]
            entry = info["entry"]
            atr = info["atr"]
            original_lot = info["original_lot"]
            is_buy = direction == "BUY"

            sym_info = mt5.symbol_info(pos.symbol)
            if not sym_info:
                continue

            state["checks"] += 1

            # Track max favorable price
            if is_buy:
                if current > state["max_favorable"]:
                    state["max_favorable"] = current
                cur_profit_dist = current - entry
            else:
                if current < state["max_favorable"]:
                    state["max_favorable"] = current
                cur_profit_dist = entry - current

            max_profit_dist = abs(state["max_favorable"] - entry)

            # Local V4 params from UI settings
            _v = self._v4
            _micro_tp_mult = _v["micro_tp"]
            _micro_pct = _v["micro_pct"]
            _be_trigger = _v["be_trigger"]
            _be_buffer = _v["be_buffer"]
            _stall_bars = _v["stall_bars"]
            _trail_dist_mult = _v["trail_dist"]

            # ── Layer 1: Micro-Partial at micro_tp * ATR ──
            if not state["micro_tp"] and not state["tp1"]:
                micro_level = entry + (_micro_tp_mult * atr if is_buy else -_micro_tp_mult * atr)
                micro_hit = (is_buy and current >= micro_level) or (not is_buy and current <= micro_level)
                if micro_hit:
                    close_lot = round(original_lot * _micro_pct, 2)
                    vol_step = sym_info.volume_step
                    vol_min = sym_info.volume_min
                    close_lot = max(vol_min, round(close_lot / vol_step) * vol_step)
                    close_lot = min(close_lot, pos.volume)
                    if close_lot >= vol_min:
                        ok = self._partial_close(pos, close_lot, sym_info)
                        if ok:
                            state["micro_tp"] = True
                            self.log_message.emit(
                                f"  MICRO-TP: {pos.symbol} closed {close_lot:.2f}L "
                                f"@ {current:.5f} ({_micro_tp_mult}x ATR)")

            # ── Layer 2: Early BE at be_trigger * ATR ──
            if not state["be_activated"]:
                if max_profit_dist >= _be_trigger * atr:
                    be_buffer_price = _be_buffer * atr
                    be_level = entry + (be_buffer_price if is_buy else -be_buffer_price)
                    # Only move SL if it improves (tighter)
                    should_move = (is_buy and be_level > pos.sl) or (not is_buy and be_level < pos.sl)
                    if should_move and self._is_spread_safe(sym_info, be_buffer_price):
                        ok = self._move_sl(pos, be_level, sym_info)
                        if ok:
                            state["be_activated"] = True
                            self.log_message.emit(
                                f"  EARLY BE: {pos.symbol} SL -> {be_level:.5f} "
                                f"(entry+{_be_buffer}x ATR)")

            # ── Layer 3: Stall Exit — move to BE after stall_bars H4 bars w/o TP1 ──
            if not state["tp1"] and not state["stall_be"]:
                # Approximate bars: each check is 2s, H4 bar = 14400s
                approx_bars = state["checks"] * 2 / 14400
                if approx_bars >= _stall_bars:
                    be_buffer_price = _be_buffer * atr
                    be_level = entry + (be_buffer_price if is_buy else -be_buffer_price)
                    should_move = (is_buy and be_level > pos.sl) or (not is_buy and be_level < pos.sl)
                    if should_move and self._is_spread_safe(sym_info, be_buffer_price):
                        ok = self._move_sl(pos, be_level, sym_info)
                        if ok:
                            state["stall_be"] = True
                            state["be_activated"] = True
                            self.log_message.emit(
                                f"  STALL EXIT: {pos.symbol} SL -> BE after ~{approx_bars:.1f} bars w/o TP1")

            # ── Layer 4: TP1 Hit — partial close + activate profit trail ──
            if not state["tp1"]:
                hit = (is_buy and current >= info["tp1"]) or (not is_buy and current <= info["tp1"])
                if hit:
                    close_lot = round(original_lot * TP1_CLOSE_PCT, 2)
                    vol_step = sym_info.volume_step
                    vol_min = sym_info.volume_min
                    close_lot = max(vol_min, round(close_lot / vol_step) * vol_step)
                    # Refresh volume (micro-partial may have reduced it)
                    refreshed = mt5.positions_get(ticket=ticket)
                    cur_vol = refreshed[0].volume if refreshed and len(refreshed) > 0 else pos.volume
                    close_lot = min(close_lot, cur_vol)
                    if close_lot >= vol_min:
                        ok = self._partial_close(pos, close_lot, sym_info)
                        if ok:
                            state["tp1"] = True
                            # Move SL to BE if not already
                            if not state["be_activated"]:
                                be_buffer_price = _be_buffer * atr
                                be_level = entry + (be_buffer_price if is_buy else -be_buffer_price)
                                if self._is_spread_safe(sym_info, be_buffer_price):
                                    self._move_sl(pos, be_level, sym_info)
                                    state["be_activated"] = True
                            self.log_message.emit(
                                f"  TP1 HIT: {pos.symbol} closed {close_lot:.2f}L "
                                f"@ {current:.5f} | Trail activated")

            # ── Layer 5: Post-TP1 Profit Trail ──
            if state["tp1"] and not state["tp3"]:
                trail_dist = _trail_dist_mult * atr
                if is_buy:
                    new_trail = state["max_favorable"] - trail_dist
                    if new_trail > entry and (state["profit_trail_sl"] is None or new_trail > state["profit_trail_sl"]):
                        state["profit_trail_sl"] = new_trail
                        # Move SL to trail level
                        if new_trail > pos.sl:
                            ok = self._move_sl(pos, new_trail, sym_info)
                            if ok:
                                self.log_message.emit(
                                    f"  TRAIL: {pos.symbol} SL -> {new_trail:.5f} "
                                    f"(peak={state['max_favorable']:.5f} - {_trail_dist_mult}x ATR)")
                else:
                    new_trail = state["max_favorable"] + trail_dist
                    if new_trail < entry and (state["profit_trail_sl"] is None or new_trail < state["profit_trail_sl"]):
                        state["profit_trail_sl"] = new_trail
                        if new_trail < pos.sl:
                            ok = self._move_sl(pos, new_trail, sym_info)
                            if ok:
                                self.log_message.emit(
                                    f"  TRAIL: {pos.symbol} SL -> {new_trail:.5f} "
                                    f"(peak={state['max_favorable']:.5f} + {_trail_dist_mult}x ATR)")

            # ── TP2: partial close + tighten trail ──
            if state["tp1"] and not state["tp2"]:
                hit = (is_buy and current >= info["tp2"]) or (not is_buy and current <= info["tp2"])
                if hit:
                    refreshed = mt5.positions_get(ticket=ticket)
                    if refreshed and len(refreshed) > 0:
                        cur_vol = refreshed[0].volume
                    else:
                        continue
                    close_lot = round(original_lot * TP2_CLOSE_PCT, 2)
                    vol_step = sym_info.volume_step
                    vol_min = sym_info.volume_min
                    close_lot = max(vol_min, round(close_lot / vol_step) * vol_step)
                    close_lot = min(close_lot, cur_vol)
                    if close_lot >= vol_min:
                        ok = self._partial_close(pos, close_lot, sym_info)
                        if ok:
                            state["tp2"] = True
                            # Move SL to TP1 level
                            self._move_sl(pos, info["tp1"], sym_info)
                            self.log_message.emit(
                                f"  TP2 HIT: {pos.symbol} closed {close_lot:.2f}L "
                                f"@ {current:.5f} | SL -> TP1")

            # ── TP3: close everything ──
            if state["tp2"] and not state["tp3"]:
                hit = (is_buy and current >= info["tp3"]) or (not is_buy and current <= info["tp3"])
                if hit:
                    refreshed = mt5.positions_get(ticket=ticket)
                    if refreshed and len(refreshed) > 0:
                        cur_vol = refreshed[0].volume
                    else:
                        continue
                    if cur_vol > 0:
                        ok = self._partial_close(pos, cur_vol, sym_info)
                        if ok:
                            state["tp3"] = True
                            self.log_message.emit(
                                f"  TP3 HIT: {pos.symbol} CLOSED ALL {cur_vol:.2f}L "
                                f"@ {current:.5f}")

        # Persist state after each full cycle (covers all mutations above)
        if self._trade_info:
            self._save_state()

    def _partial_close(self, pos, close_volume, sym_info):
        """Partially close a position."""
        try:
            close_type = mt5.ORDER_TYPE_SELL if pos.type == 0 else mt5.ORDER_TYPE_BUY
            price = sym_info.bid if pos.type == 0 else sym_info.ask

            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": pos.symbol,
                "volume": close_volume,
                "type": close_type,
                "position": pos.ticket,
                "price": price,
                "deviation": 20,
                "magic": MAGIC_NUMBER,
                "comment": "ACi-TP-Partial",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_FOK,
            }
            for fill in [mt5.ORDER_FILLING_FOK, mt5.ORDER_FILLING_IOC, mt5.ORDER_FILLING_RETURN]:
                request["type_filling"] = fill
                result = mt5.order_send(request)
                if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                    return True
            rc = result.retcode if result else "?"
            self.log_message.emit(f"  Partial close FAILED: {pos.symbol} retcode={rc}")
            return False
        except Exception as e:
            self.log_message.emit(f"  Partial close error: {e}")
            return False

    def _is_spread_safe(self, sym_info, be_buffer_price):
        """Check if current spread is safe for SL modification.

        Returns True if spread < 50% of the BE buffer (in price units).
        During rollover/news, spreads widen 5-10x and would eat the entire
        buffer, causing instant stop-out after moving SL to BE.
        """
        spread = sym_info.ask - sym_info.bid
        if be_buffer_price <= 0:
            return False
        if spread > be_buffer_price * 0.50:
            self.log_message.emit(
                f"  SPREAD GUARD: {sym_info.name} spread={spread:.5f} > "
                f"50% of buffer={be_buffer_price:.5f} — SL move SKIPPED")
            return False
        return True

    def _move_sl_to_be(self, pos, entry, sym_info):
        """Move SL to breakeven (entry price)."""
        self._move_sl(pos, entry, sym_info)

    def _move_sl(self, pos, new_sl, sym_info, skip_spread_check=False):
        """Modify position SL.

        Args:
            skip_spread_check: If True, bypass spread safety check (used for
                               trailing stops that move SL further from price).
        """
        try:
            digits = sym_info.digits
            request = {
                "action": mt5.TRADE_ACTION_SLTP,
                "symbol": pos.symbol,
                "position": pos.ticket,
                "sl": round(new_sl, digits),
                "tp": pos.tp,
                "magic": MAGIC_NUMBER,
            }
            result = mt5.order_send(request)
            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                return True
            rc = result.retcode if result else "?"
            self.log_message.emit(f"  SL modify FAILED: {pos.symbol} retcode={rc}")
            return False
        except Exception as e:
            self.log_message.emit(f"  SL modify error: {e}")
            return False


# ---------------------------------------------------------------------------
# Main App
# ---------------------------------------------------------------------------
# WebBridge — Python <-> JavaScript communication via QWebChannel
# ---------------------------------------------------------------------------
class WebBridge(QObject):
    """Exposes Python methods callable from JavaScript UI."""

    def __init__(self, app):
        super().__init__()
        self._app = app

    @Slot(str)
    def startScanner(self, config_json=""):
        self._app._on_start()

    @Slot()
    def stopScanner(self):
        self._app._on_stop()

    @Slot()
    def toggleTrading(self):
        self._app._toggle_trading()

    @Slot()
    def closeAll(self):
        self._app._on_close_all()

    @Slot(str)
    def updateSettings(self, settings_json):
        try:
            settings = json.loads(settings_json)
            self._app._apply_settings(settings)
        except Exception as e:
            self._app._log(f"Settings update error: {e}")

    @Slot(result=str)
    def requestInitialState(self):
        return json.dumps(self._app._get_full_state())

    @Slot(str)
    def selectPair(self, symbol):
        self._app._current_pair = symbol

    @Slot()
    def refreshNews(self):
        self._app._fetch_and_push_news()


# ---------------------------------------------------------------------------
# Main App — Thin PySide6 shell with QWebEngineView
# ---------------------------------------------------------------------------
class ACiApp(QMainWindow):
    _log_signal = Signal(str)
    _news_ready = Signal(str, str)  # news_json, sentiment_json

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Velocity 4")
        self.setMinimumSize(1100, 750)
        self.resize(1400, 900)

        self._scanner = None
        self._running = False
        self._trading_enabled = False
        self._chart_data = {}       # {symbol: DataFrame}  — cached for chart
        self._chart_loaded = {}     # {symbol: bool}
        self._starting_balance = None  # set on first MT5 connect for growth calc
        self._current_pair = "EURUSD"

        # Settings dict — replaces per-widget reads
        self._settings = {
            "risk": "30", "lots": "0.40", "max_trades": "5",
            "poll_sec": "30", "timeframe": "H4",
            "pairs": {s: True for s in ALL_PAIRS},
            "v4_tp1": str(TP1_MULT_AGG), "v4_tp2": str(TP2_MULT_AGG),
            "v4_tp3": str(TP3_MULT_AGG),
            "v4_be_trigger": str(BE_TRIGGER_MULT),
            "v4_be_buffer": str(BE_BUFFER_MULT),
            "v4_micro_tp": str(MICRO_TP_MULT),
            "v4_micro_pct": str(int(MICRO_TP_PCT * 100)),
            "v4_stall": str(STALL_BARS),
            "v4_trail": str(PROFIT_TRAIL_DISTANCE_MULT),
        }

        # TP Manager
        self._tp_manager = TPManager()
        self._tp_manager.log_message.connect(self._on_scanner_log)
        self._tp_manager.start()

        self._log_signal.connect(self._log_on_main_thread)
        self._news_ready.connect(self._on_news_ready)

        self._load_settings()
        self._build_web_ui()

        # Push V4 params to TP manager after settings loaded
        self._tp_manager.set_v4_params(self._get_v4_params())

        # MT5 check on startup
        self._check_mt5()

        # Live position refresh (2s)
        self._timer = QTimer()
        self._timer.timeout.connect(self._update_live)
        self._timer.start(2000)

        # Portfolio refresh (60s — deal history doesn't change every tick)
        self._portfolio_timer = QTimer()
        self._portfolio_timer.timeout.connect(self._push_portfolio)
        self._portfolio_timer.start(60000)

        # Market news refresh (5 min — calendar data doesn't change fast)
        self._news_timer = QTimer()
        self._news_timer.timeout.connect(self._fetch_and_push_news)
        self._news_timer.start(300000)

    # ── Build Web UI ──

    def _build_web_ui(self):
        """Create QWebEngineView with QWebChannel bridge to v4_ui.html."""
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._web = QWebEngineView()
        self._web.setStyleSheet("background-color: #0B0B11;")

        # Allow file:// pages to load external CDN resources (Tailwind, fonts, icons, charts)
        ws = self._web.page().settings()
        ws.setAttribute(QWebEngineSettings.WebAttribute.LocalContentCanAccessRemoteUrls, True)

        # Set up QWebChannel
        self._channel = QWebChannel()
        self._bridge = WebBridge(self)
        self._channel.registerObject("v4bridge", self._bridge)
        self._web.page().setWebChannel(self._channel)

        # Load HTML
        html_path = os.path.abspath(HTML_UI_PATH)
        self._web.setUrl(QUrl.fromLocalFile(html_path))
        self._web.loadFinished.connect(self._on_web_loaded)

        layout.addWidget(self._web)

    def _on_web_loaded(self, ok):
        """Push initial state to JS after HTML loads."""
        if not ok:
            log.error("Failed to load v4_ui.html")
            return
        self._log("Web UI loaded successfully")
        # Push initial state after a small delay for bridge to init
        QTimer.singleShot(500, self._push_initial_state)

    def _push_initial_state(self):
        """Send full app state to the JS UI."""
        state = self._get_full_state()
        js = f"if (typeof applyInitialState === 'function') applyInitialState({json.dumps(state)});"
        self._push_to_js(js)
        # Re-push MT5 status so JS badge updates after bridge is ready
        try:
            acct = mt5.account_info()
            if acct:
                self._push_to_js(f"updateMT5Status(true, '{acct.login}')")
            else:
                self._push_to_js("updateMT5Status(false, '')")
        except Exception:
            pass

        # Push portfolio equity curve
        QTimer.singleShot(1500, self._push_portfolio)

        # Push market news + sentiment
        QTimer.singleShot(2500, self._fetch_and_push_news)

        # Auto-start scanner on launch (after UI is ready)
        if not self._running:
            self._log("Auto-starting scanner...")
            QTimer.singleShot(2000, self._on_start)

    def _push_to_js(self, js_call):
        """Execute JavaScript in the web view."""
        try:
            self._web.page().runJavaScript(js_call)
        except Exception as e:
            log.warning(f"JS push error: {e}")

    # ── Full State for JS init ──

    def _get_full_state(self):
        """Return JSON-serializable dict of all app state."""
        return {
            "settings": dict(self._settings),
            "scannerRunning": self._running,
            "tradingEnabled": self._trading_enabled,
        }

    # ── MT5 check ──

    def _check_mt5(self):
        try:
            if not mt5.initialize():
                self._log("MT5 not connected. Please start MetaTrader 5.")
                self._push_to_js("updateMT5Status(false, '')")
                return False
            acct = mt5.account_info()
            if acct is None:
                self._log("MT5: Cannot read account info")
                return False
            self._starting_balance = acct.balance
            self._push_to_js(f"updateMT5Status(true, '{acct.login}')")
            self._log(f"MT5 connected: Account {acct.login} | ${acct.balance:.2f}")
            return True
        except Exception as e:
            self._log(f"MT5 error: {e}")
            self._push_to_js("updateMT5Status(false, '')")
            return False

    # ── V4 Params ──

    def _get_v4_params(self):
        """Read V4 profit capture parameters from settings dict."""
        s = self._settings
        try:
            return {
                "tp1_mult": float(s.get("v4_tp1", TP1_MULT_AGG)),
                "tp2_mult": float(s.get("v4_tp2", TP2_MULT_AGG)),
                "tp3_mult": float(s.get("v4_tp3", TP3_MULT_AGG)),
                "be_trigger": float(s.get("v4_be_trigger", BE_TRIGGER_MULT)),
                "be_buffer": float(s.get("v4_be_buffer", BE_BUFFER_MULT)),
                "micro_tp": float(s.get("v4_micro_tp", MICRO_TP_MULT)),
                "micro_pct": float(s.get("v4_micro_pct", str(MICRO_TP_PCT * 100))) / 100,
                "stall_bars": int(float(s.get("v4_stall", STALL_BARS))),
                "trail_dist": float(s.get("v4_trail", PROFIT_TRAIL_DISTANCE_MULT)),
            }
        except (ValueError, AttributeError):
            return {
                "tp1_mult": TP1_MULT_AGG, "tp2_mult": TP2_MULT_AGG, "tp3_mult": TP3_MULT_AGG,
                "be_trigger": BE_TRIGGER_MULT, "be_buffer": BE_BUFFER_MULT,
                "micro_tp": MICRO_TP_MULT, "micro_pct": MICRO_TP_PCT,
                "stall_bars": STALL_BARS, "trail_dist": PROFIT_TRAIL_DISTANCE_MULT,
            }

    def _push_v4_params(self):
        """Push current V4 params to scanner and TPManager."""
        v4 = self._get_v4_params()
        if self._scanner:
            self._scanner.set_v4_params(v4)
        self._tp_manager.set_v4_params(v4)

    # ── Apply Settings from JS ──

    def _apply_settings(self, settings):
        """Apply settings dict received from JS bridge."""
        self._settings.update(settings)
        self._push_v4_params()
        self._save_settings()
        self._log("Settings updated from UI")

        # If scanner is running and TF changed, restart it
        new_tf = settings.get("timeframe")
        if new_tf and self._running and self._scanner:
            old_tf = self._settings.get("_active_tf")
            if old_tf and old_tf != new_tf:
                self._log(f"Timeframe changed to {new_tf} — restarting scanner")
                self._on_stop()
                self._on_start()
        if new_tf:
            self._settings["_active_tf"] = new_tf

    # ── Start/Stop ──

    def _on_start(self):
        if self._running:
            return

        if not mt5.initialize():
            self._log("Cannot start — MT5 not connected")
            return

        pairs = self._settings.get("pairs", {})
        symbols = [s for s in ALL_PAIRS if pairs.get(s, True)]
        if not symbols:
            self._log("No pairs selected!")
            return

        tf_key = self._settings.get("timeframe", "H4")
        poll = int(self._settings.get("poll_sec", "30") or "30")
        risk = float(self._settings.get("risk", "30") or "30") / 100
        lots = float(self._settings.get("lots", "0.40") or "0.40")
        max_trades = int(self._settings.get("max_trades", "5") or "5")

        self._scanner = ScanEngine()
        self._scanner._auto_trade = self._trading_enabled
        self._scanner.log_message.connect(self._on_scanner_log)
        self._scanner.signal_detected.connect(self._on_signal)
        self._scanner.scan_complete.connect(self._on_scan_data)
        self._push_v4_params()
        self._scanner.start_scanning(symbols, tf_key, poll, risk, lots, max_trades)

        self._running = True
        self._settings["_active_tf"] = tf_key
        self._push_to_js(f"updateScannerState(true, {json.dumps(self._trading_enabled)})")
        self._save_settings()

    def _on_stop(self):
        if self._scanner:
            self._scanner.stop()
            self._scanner = None
        self._running = False
        self._push_to_js(f"updateScannerState(false, {json.dumps(self._trading_enabled)})")

    # ── Trade toggle ──

    def _toggle_trading(self):
        self._trading_enabled = not self._trading_enabled
        if self._trading_enabled:
            self._log("TRADING ENABLED — signals will auto-execute")
        else:
            self._log("TRADING DISABLED — scan-only mode")

        # Update scanner's auto-trade flag
        if self._scanner:
            self._scanner._auto_trade = self._trading_enabled
            if self._trading_enabled:
                self._scanner._entered_signals.clear()

        self._push_to_js(f"updateTradingState({json.dumps(self._trading_enabled)})")

    # ── Scanner signal handlers ──

    def _on_scanner_log(self, msg):
        self._log(msg)

    def _on_signal(self, symbol, direction, entry, sl, tp1, tp2, tp3, atr):
        sl_dist = abs(entry - sl)
        rr = abs(tp1 - entry) / sl_dist if sl_dist > 0 else 0
        v4 = self._get_v4_params()
        be_price = entry + (v4["be_trigger"] * atr if direction == "BUY" else -v4["be_trigger"] * atr)
        self._log(f"*** V4 SIGNAL: {direction} {symbol} @ {entry:.5f} | "
                  f"SL={sl:.5f} TP1={tp1:.5f} TP2={tp2:.5f} TP3={tp3:.5f} | "
                  f"R:R={rr:.2f} BE@{be_price:.5f} ATR={atr:.5f} ***")

        # Register with V4 TP Manager if trading is enabled
        if self._trading_enabled:
            QTimer.singleShot(3000, lambda: self._register_trade_for_tp(
                symbol, direction, entry, sl, tp1, tp2, tp3, atr))

        # Push signal levels to JS chart
        levels = {
            "direction": direction,
            "entry": entry, "sl": sl,
            "be": be_price,
            "tp1": tp1, "tp2": tp2, "tp3": tp3,
        }
        self._push_to_js(f"updateSignalLevels('{symbol}', {json.dumps(levels)})")

    def _register_trade_for_tp(self, symbol, direction, entry, sl, tp1, tp2, tp3, atr_val):
        """Find the just-placed trade ticket and register it with V4 TP manager."""
        try:
            positions = mt5.positions_get()
            if not positions:
                return
            norm = symbol.upper().replace(".", "").replace("#", "")
            for pos in positions:
                pos_norm = pos.symbol.upper().replace(".", "").replace("#", "")
                if pos_norm == norm and pos.magic == MAGIC_NUMBER:
                    if pos.ticket not in self._tp_manager._trade_info:
                        self._tp_manager.register_trade(
                            pos.ticket, direction, entry, sl, tp1, tp2, tp3, atr_val, pos.volume)
                        return
        except Exception as e:
            self._log(f"TP registration error: {e}")

    def _on_scan_data(self, symbol, df):
        """Serialize scan DataFrame and push chart data to JS."""
        self._chart_data[symbol] = df

        try:
            # Prepare OHLCV as JSON-serializable list
            ohlcv = []
            for _, row in df.iterrows():
                t = row["time"]
                # Convert pandas Timestamp to unix seconds
                if hasattr(t, "timestamp"):
                    ts = int(t.timestamp())
                else:
                    ts = int(t)
                ohlcv.append({
                    "time": ts,
                    "open": float(row["open"]),
                    "high": float(row["high"]),
                    "low": float(row["low"]),
                    "close": float(row["close"]),
                    "volume": float(row.get("volume", 0) or 0),
                })

            # Split trailing stop into bull/bear
            bull_data = []
            bear_data = []
            for _, row in df.iterrows():
                t = row["time"]
                ts = int(t.timestamp()) if hasattr(t, "timestamp") else int(t)
                zp = row.get("xATRTrailingStop")
                pos = row.get("pos")
                if zp is not None and not (isinstance(zp, float) and math.isnan(zp)):
                    if pos == 1:
                        bull_data.append({"time": ts, "value": float(zp)})
                    elif pos == -1:
                        bear_data.append({"time": ts, "value": float(zp)})

            # Collect markers
            markers = []
            for _, row in df.iterrows():
                t = row["time"]
                ts = int(t.timestamp()) if hasattr(t, "timestamp") else int(t)
                if row.get("buy_signal", False):
                    markers.append({
                        "time": ts,
                        "position": "belowBar",
                        "shape": "arrowUp",
                        "color": "#4ADE80",
                        "text": "BUY",
                    })
                elif row.get("sell_signal", False):
                    markers.append({
                        "time": ts,
                        "position": "aboveBar",
                        "shape": "arrowDown",
                        "color": "#F87171",
                        "text": "SELL",
                    })

            if not self._chart_loaded.get(symbol, False):
                # Full chart load
                js = (f"updateChartData('{symbol}',"
                      f"{json.dumps(ohlcv)},"
                      f"{json.dumps(bull_data)},"
                      f"{json.dumps(bear_data)},"
                      f"{json.dumps(markers)})")
                self._push_to_js(js)
                self._chart_loaded[symbol] = True
            else:
                # Incremental update — last bar only
                last_bar = ohlcv[-1] if ohlcv else None
                last_bull = bull_data[-1] if bull_data else None
                last_bear = bear_data[-1] if bear_data else None

                # Check for new signal on confirmed bar (second to last)
                new_markers = []
                if len(df) > 1:
                    prev = df.iloc[-2]
                    t = prev["time"]
                    ts = int(t.timestamp()) if hasattr(t, "timestamp") else int(t)
                    if prev.get("buy_signal", False):
                        new_markers.append({
                            "time": ts, "position": "belowBar",
                            "shape": "arrowUp", "color": "#4ADE80", "text": "BUY",
                        })
                    elif prev.get("sell_signal", False):
                        new_markers.append({
                            "time": ts, "position": "aboveBar",
                            "shape": "arrowDown", "color": "#F87171", "text": "SELL",
                        })

                js = (f"updateChartIncremental('{symbol}',"
                      f"{json.dumps(last_bar)},"
                      f"{json.dumps(last_bull)},"
                      f"{json.dumps(last_bear)},"
                      f"{json.dumps(new_markers)})")
                self._push_to_js(js)

        except Exception as e:
            self._log(f"[{symbol}] Chart data push error: {e}")

    # ── Logging ──

    def _log(self, msg):
        ts = datetime.now().strftime("%H:%M:%S")
        full = f"[{ts}] {msg}"
        log.info(msg)
        self._log_signal.emit(full)

    def _log_on_main_thread(self, msg):
        """Push log message to JS UI."""
        safe = msg.replace("\\", "\\\\").replace("'", "\\'").replace("\n", "\\n")
        self._push_to_js(f"appendLog('{safe}')")

    # ── Live account stats ──

    def _update_live(self):
        try:
            if not mt5.terminal_info():
                if not mt5.initialize():
                    return

            acct = mt5.account_info()
            if not acct:
                return

            balance = acct.balance
            equity = acct.equity
            margin = acct.margin
            free_margin = acct.margin_free

            if self._starting_balance is None:
                self._starting_balance = balance

            # Growth %
            growth = 0.0
            if self._starting_balance and self._starting_balance > 0:
                growth = ((equity - self._starting_balance) / self._starting_balance) * 100

            # Margin level
            margin_level = 0.0
            if margin > 0:
                margin_level = (equity / margin) * 100

            # Push account info to JS
            acct_data = {
                "balance": balance,
                "equity": equity,
                "margin": margin,
                "freeMargin": free_margin,
                "growth": growth,
                "marginLevel": margin_level,
            }
            self._push_to_js(f"updateAccountInfo({json.dumps(acct_data)})")

            # V4 manager status
            managed = len(self._tp_manager._trade_info)
            be_count = sum(1 for s in self._tp_manager._tp_state.values() if s.get("be_activated"))
            tp1_count = sum(1 for s in self._tp_manager._tp_state.values() if s.get("tp1"))
            trail_count = sum(1 for s in self._tp_manager._tp_state.values()
                              if s.get("profit_trail_sl") is not None)
            micro_count = sum(1 for s in self._tp_manager._tp_state.values() if s.get("micro_tp"))

            v4_data = {
                "managed": managed,
                "be": be_count,
                "tp1": tp1_count,
                "trail": trail_count,
                "micro": micro_count,
            }
            self._push_to_js(f"updateV4Status({json.dumps(v4_data)})")

            # Push positions data to JS
            self._push_positions()

        except Exception:
            pass

    def _push_positions(self):
        """Push open positions + trade history to JS."""
        try:
            positions = mt5.positions_get()
            if positions is None:
                positions = []
            my_pos = [p for p in positions if p.magic == MAGIC_NUMBER]

            open_list = []
            for pos in my_pos:
                direction = "BUY" if pos.type == 0 else "SELL"
                tp_state = self._tp_manager._tp_state.get(pos.ticket, {})
                open_list.append({
                    "symbol": pos.symbol,
                    "direction": direction,
                    "lots": pos.volume,
                    "entry": pos.price_open,
                    "current": pos.price_current,
                    "pl": pos.profit,
                    "sl": pos.sl,
                    "tp": pos.tp,
                    "micro_tp": tp_state.get("micro_tp", False),
                    "be_activated": tp_state.get("be_activated", False),
                    "stall_be": tp_state.get("stall_be", False),
                    "tp1": tp_state.get("tp1", False),
                    "tp2": tp_state.get("tp2", False),
                    "profit_trail_sl": tp_state.get("profit_trail_sl") is not None,
                    "peak": tp_state.get("max_favorable"),
                })

            # Trade history (today)
            now = datetime.now()
            today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
            deals = mt5.history_deals_get(today_start, now)
            if deals is None:
                deals = []

            close_deals = [d for d in deals
                           if d.magic == MAGIC_NUMBER
                           and d.entry == mt5.DEAL_ENTRY_OUT
                           and d.type in (mt5.DEAL_TYPE_BUY, mt5.DEAL_TYPE_SELL)]

            history_list = []
            for deal in close_deals:
                direction = "BUY" if deal.type == mt5.DEAL_TYPE_BUY else "SELL"
                deal_time = datetime.fromtimestamp(deal.time).strftime("%H:%M:%S")
                history_list.append({
                    "symbol": deal.symbol,
                    "direction": direction,
                    "lots": deal.volume,
                    "entry": deal.price,
                    "close": deal.price,
                    "pl": deal.profit,
                    "time": deal_time,
                })

            pos_data = {"open": open_list, "history": history_list}
            self._push_to_js(f"updatePositions({json.dumps(pos_data)})")

        except Exception:
            pass

    # ── Portfolio Equity Curve ──

    def _push_portfolio(self):
        """Fetch full deal history from MT5 and push equity curve + stats + pair breakdown to JS."""
        try:
            # Fetch ALL deals from the last 90 days (covers most trading history)
            from datetime import timedelta
            from collections import defaultdict
            now = datetime.now()
            start = now - timedelta(days=90)
            deals = mt5.history_deals_get(start, now)
            if deals is None:
                deals = []

            # Filter to our magic number close deals (entry_out = profit/loss realized)
            close_deals = [d for d in deals
                           if d.magic == MAGIC_NUMBER
                           and d.entry == mt5.DEAL_ENTRY_OUT
                           and d.type in (mt5.DEAL_TYPE_BUY, mt5.DEAL_TYPE_SELL)]

            if not close_deals:
                # No history — push empty
                self._push_to_js("updatePortfolio([], {})")
                self._push_to_js("updatePortfolioPairs([])")
                return

            # Sort by time
            close_deals.sort(key=lambda d: d.time)

            # Build cumulative equity curve
            acct = mt5.account_info()
            current_balance = acct.balance if acct else 0
            total_pnl = sum(d.profit + d.commission + d.swap for d in close_deals)
            starting_eq = current_balance - total_pnl

            curve = []
            running = starting_eq
            wins = 0
            losses = 0
            gross_profit = 0.0
            gross_loss = 0.0
            win_amounts = []
            loss_amounts = []

            # Per-day P/L tracking
            daily_pnl = defaultdict(float)
            # Per-symbol tracking
            symbol_stats = defaultdict(lambda: {"wins": 0, "losses": 0, "gross_profit": 0.0, "gross_loss": 0.0, "pnls": []})
            # Win streak tracking
            current_streak = 0
            max_streak = 0

            for deal in close_deals:
                pnl = deal.profit + deal.commission + deal.swap
                running += pnl
                ts = int(deal.time)  # unix seconds
                curve.append({"time": ts, "value": round(running, 2)})

                # Day key for daily P/L
                deal_date = datetime.fromtimestamp(deal.time).strftime("%Y-%m-%d")
                daily_pnl[deal_date] += pnl

                # Symbol breakdown
                sym = deal.symbol.upper().replace(".", "").replace("#", "")
                # Normalize to 6-char base (e.g., EURUSDm -> EURUSD)
                for base in ALL_PAIRS:
                    if sym.startswith(base):
                        sym = base
                        break
                ss = symbol_stats[sym]
                ss["pnls"].append(pnl)

                if pnl >= 0:
                    wins += 1
                    gross_profit += pnl
                    win_amounts.append(pnl)
                    ss["wins"] += 1
                    ss["gross_profit"] += pnl
                    current_streak += 1
                    max_streak = max(max_streak, current_streak)
                else:
                    losses += 1
                    gross_loss += abs(pnl)
                    loss_amounts.append(pnl)
                    ss["losses"] += 1
                    ss["gross_loss"] += abs(pnl)
                    current_streak = 0

            # Add current equity as last point
            if acct:
                curve.append({"time": int(now.timestamp()), "value": round(acct.equity, 2)})

            total = wins + losses
            win_rate = (wins / total * 100) if total > 0 else 0
            pf = (gross_profit / gross_loss) if gross_loss > 0 else 999.0
            net = gross_profit - gross_loss
            avg_trade = net / total if total > 0 else 0
            avg_win = sum(win_amounts) / len(win_amounts) if win_amounts else 0
            avg_loss = sum(loss_amounts) / len(loss_amounts) if loss_amounts else 0

            # Best/worst day
            best_day = max(daily_pnl.values()) if daily_pnl else None
            worst_day = min(daily_pnl.values()) if daily_pnl else None

            # Best pair by net profit
            best_pair = ""
            best_pair_net = -float("inf")
            for sym, ss in symbol_stats.items():
                sym_net = ss["gross_profit"] - ss["gross_loss"]
                if sym_net > best_pair_net:
                    best_pair_net = sym_net
                    best_pair = sym

            # Growth %
            equity = acct.equity if acct else 0
            growth_pct = 0.0
            if starting_eq > 0:
                growth_pct = ((equity - starting_eq) / starting_eq) * 100

            stats = {
                "totalTrades": total,
                "wins": wins,
                "losses": losses,
                "winRate": round(win_rate, 1),
                "profitFactor": round(pf, 2),
                "netProfit": round(net, 2),
                "grossProfit": round(gross_profit, 2),
                "grossLoss": round(gross_loss, 2),
                "avgTrade": round(avg_trade, 2),
                "avgWin": round(avg_win, 2) if win_amounts else None,
                "avgLoss": round(avg_loss, 2) if loss_amounts else None,
                "bestDay": round(best_day, 2) if best_day is not None else None,
                "worstDay": round(worst_day, 2) if worst_day is not None else None,
                "winStreak": max_streak,
                "bestPair": best_pair,
                "netWorth": round(equity, 2),
                "growthPct": round(growth_pct, 2),
            }

            self._push_to_js(f"updatePortfolio({json.dumps(curve)}, {json.dumps(stats)})")

            # ── Pair breakdown for Portfolio table ──
            pair_list = []
            for sym, ss in symbol_stats.items():
                sym_total = ss["wins"] + ss["losses"]
                sym_wr = (ss["wins"] / sym_total * 100) if sym_total > 0 else 0
                sym_net = ss["gross_profit"] - ss["gross_loss"]
                sym_avg = sym_net / sym_total if sym_total > 0 else 0
                sym_pf = (ss["gross_profit"] / ss["gross_loss"]) if ss["gross_loss"] > 0 else 999.0
                pair_list.append({
                    "symbol": sym,
                    "trades": sym_total,
                    "winRate": round(sym_wr, 1),
                    "net": round(sym_net, 2),
                    "avg": round(sym_avg, 2),
                    "pf": round(sym_pf, 2),
                })

            # Sort by net profit descending
            pair_list.sort(key=lambda x: x["net"], reverse=True)

            self._push_to_js(f"updatePortfolioPairs({json.dumps(pair_list)})")

        except Exception as e:
            log.warning(f"Portfolio push error: {e}")

    # ── Dashboard Market News + Sentiment ──

    def _fetch_and_push_news(self):
        """Fetch forex news headlines and USD sentiment in background thread."""
        def _worker():
            try:
                import urllib.request
                import xml.etree.ElementTree as ET

                news_items = []
                now = datetime.now()

                # ── Fetch ForexFactory calendar RSS (forex news) ──
                try:
                    req = urllib.request.Request(
                        "https://nfs.faireconomy.media/ff_calendar_thisweek.json",
                        headers={"User-Agent": "Mozilla/5.0"}
                    )
                    with urllib.request.urlopen(req, timeout=8) as resp:
                        cal_data = json.loads(resp.read().decode("utf-8"))

                    # Filter to USD and high/medium impact, recent or upcoming
                    for evt in cal_data:
                        currency = evt.get("country", "")
                        impact = evt.get("impact", "").lower()
                        title = evt.get("title", "")
                        date_str = evt.get("date", "")

                        if not title:
                            continue

                        # Parse event date
                        try:
                            evt_dt = datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%S%z")
                            evt_naive = evt_dt.replace(tzinfo=None)
                        except Exception:
                            evt_naive = now

                        diff = (now - evt_naive).total_seconds()
                        # Show events from past 24h and upcoming 24h
                        if diff > 86400 or diff < -86400:
                            continue

                        if diff > 0:
                            hours = diff / 3600
                            if hours < 1:
                                time_ago = f"{int(diff/60)}m ago"
                            else:
                                time_ago = f"{int(hours)}h ago"
                        else:
                            hours = abs(diff) / 3600
                            if hours < 1:
                                time_ago = f"in {int(abs(diff)/60)}m"
                            else:
                                time_ago = f"in {int(hours)}h"

                        # Add forecast/actual if available
                        forecast = evt.get("forecast", "")
                        actual = evt.get("actual", "")
                        if actual:
                            title += f" — Actual: {actual}"
                            if forecast:
                                title += f" (Fcst: {forecast})"
                        elif forecast:
                            title += f" — Fcst: {forecast}"

                        news_items.append({
                            "title": title,
                            "currency": currency.upper(),
                            "impact": impact,
                            "timeAgo": time_ago,
                            "sort": diff,
                        })

                except Exception as e:
                    log.warning(f"Calendar fetch error: {e}")

                # Sort: upcoming first (negative diff), then most recent
                news_items.sort(key=lambda x: x.get("sort", 0))
                # Limit to 20 items
                news_items = news_items[:20]

                # ── Fetch DXY from MT5 ──
                dxy_val = None
                dxy_change = None
                try:
                    # Try common DXY symbol names on broker
                    for dxy_sym in ["USDX", "DXY", "DX", "USDX.raw", "DXY.raw"]:
                        info = mt5.symbol_info(dxy_sym)
                        if info is not None:
                            mt5.symbol_select(dxy_sym, True)
                            tick = mt5.symbol_info_tick(dxy_sym)
                            if tick:
                                dxy_val = tick.bid
                            # Get daily change from OHLC
                            rates = mt5.copy_rates_from_pos(dxy_sym, mt5.TIMEFRAME_D1, 0, 2)
                            if rates is not None and len(rates) >= 2:
                                prev_close = rates[-2][4]  # close
                                if prev_close > 0:
                                    dxy_change = ((dxy_val - prev_close) / prev_close) * 100
                            break
                except Exception:
                    pass

                # ── Build pair sentiment from MT5 position ratios ──
                # Use our ZP trailing stop direction as a proxy for sentiment
                sentiment_pairs = []
                for sym in ALL_PAIRS:
                    resolved = None
                    for c in [sym, sym + ".raw", sym + "m"]:
                        si = mt5.symbol_info(c)
                        if si is not None:
                            resolved = c
                            break
                    if not resolved:
                        continue

                    si = mt5.symbol_info(resolved)
                    if not si:
                        continue
                    price = f"{si.bid:.5f}" if "JPY" not in sym else f"{si.bid:.3f}"

                    # Get session stats for sentiment approximation
                    rates = mt5.copy_rates_from_pos(resolved, mt5.TIMEFRAME_H4, 0, 6)
                    if rates is not None and len(rates) >= 6:
                        # Count bullish vs bearish H4 bars as sentiment proxy
                        bulls = sum(1 for r in rates if r[4] > r[1])  # close > open
                        bears = len(rates) - bulls
                        total = bulls + bears
                        long_pct = (bulls / total * 100) if total > 0 else 50
                    else:
                        long_pct = 50

                    sentiment_pairs.append({
                        "symbol": sym,
                        "long": round(long_pct),
                        "price": price,
                    })

                # Marshal to main thread via signal (can't call runJavaScript from bg thread)
                sent_data = {
                    "dxy": dxy_val,
                    "dxyChange": dxy_change,
                    "pairs": sentiment_pairs,
                }
                self._news_ready.emit(json.dumps(news_items), json.dumps(sent_data))

            except Exception as e:
                log.warning(f"News fetch error: {e}")

        threading.Thread(target=_worker, daemon=True).start()

    def _on_news_ready(self, news_json, sentiment_json):
        """Called on main thread when background news fetch completes."""
        self._push_to_js(f"updateDashNews({news_json})")
        self._push_to_js(f"updateDashSentiment({sentiment_json})")

    # ── Close All ──

    def _on_close_all(self):
        try:
            positions = mt5.positions_get()
            if not positions:
                self._log("No positions to close")
                return

            my_pos = [p for p in positions if p.magic == MAGIC_NUMBER]
            closed = 0
            for pos in my_pos:
                close_type = mt5.ORDER_TYPE_SELL if pos.type == 0 else mt5.ORDER_TYPE_BUY
                sym_info = mt5.symbol_info(pos.symbol)
                if sym_info is None:
                    continue
                price = sym_info.bid if pos.type == 0 else sym_info.ask

                request = {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "symbol": pos.symbol,
                    "volume": pos.volume,
                    "type": close_type,
                    "position": pos.ticket,
                    "price": price,
                    "deviation": 20,
                    "magic": MAGIC_NUMBER,
                    "comment": "ACi-CloseAll",
                    "type_time": mt5.ORDER_TIME_GTC,
                    "type_filling": mt5.ORDER_FILLING_FOK,
                }
                for fill in [mt5.ORDER_FILLING_FOK, mt5.ORDER_FILLING_IOC, mt5.ORDER_FILLING_RETURN]:
                    request["type_filling"] = fill
                    result = mt5.order_send(request)
                    if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                        closed += 1
                        break

            self._log(f"Closed {closed}/{len(my_pos)} positions")
        except Exception as e:
            self._log(f"Close all error: {e}")

    # ── Settings ──

    def _save_settings(self):
        try:
            with open(SETTINGS_PATH, "w") as f:
                # Filter out internal keys
                save = {k: v for k, v in self._settings.items() if not k.startswith("_")}
                json.dump(save, f, indent=2)
        except Exception:
            pass

    def _load_settings(self):
        try:
            with open(SETTINGS_PATH, "r") as f:
                s = json.load(f)
            self._settings.update(s)
        except FileNotFoundError:
            pass
        except Exception:
            pass

    def closeEvent(self, event):
        self._on_stop()
        self._tp_manager.stop()
        self._save_settings()
        try:
            mt5.shutdown()
        except Exception:
            pass
        event.accept()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    app = QApplication(sys.argv)
    win = ACiApp()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
