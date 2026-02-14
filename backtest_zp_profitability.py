#!/usr/bin/env python3
"""
ZeroPoint PRO Indicator -- V4 Profit Capture Backtest
=====================================================
Compares BASELINE (old 2.0/3.5/5.0x ATR TPs, no protection) vs
V4 PROTECT (1.5/3.0/5.0x TPs + early BE + micro-partial + stall + trail).

Two modes side-by-side on the same signals to see the improvement.
"""

import sys
import os
import numpy as np
import pandas as pd
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "app"))

import MetaTrader5 as mt5
from app.zeropoint_signal import (
    compute_zeropoint_state,
    ATR_PERIOD, ATR_MULTIPLIER,
    TP1_MULT, TP2_MULT, TP3_MULT,
    SL_BUFFER_PCT, SL_ATR_MIN_MULT, SWING_LOOKBACK,
    # V4 constants
    BE_TRIGGER_MULT, BE_BUFFER_MULT,
    PROFIT_TRAIL_DISTANCE_MULT, STALL_BARS,
    MICRO_TP_MULT, MICRO_TP_PCT,
    TP1_MULT_AGG, TP2_MULT_AGG, TP3_MULT_AGG,
)

# Config
SYMBOLS = ["AUDUSD", "EURJPY", "EURUSD", "GBPJPY", "GBPUSD", "NZDUSD", "USDCAD", "USDJPY"]
FETCH_BARS = 5000
WARMUP_BARS = 50
LOT_SIZE = 0.10
STARTING_BALANCE = 200.0


def pip_value(symbol):
    sym = symbol.upper()
    if "JPY" in sym:
        return 0.01
    elif "BTC" in sym or "XAU" in sym:
        return 1.0
    return 0.0001


def contract_size(symbol):
    sym = symbol.upper()
    if "BTC" in sym:
        return 1.0
    return 100_000.0


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
        structural_sl = recent_low - buffer
        atr_min_sl = cur_close - atr_val * SL_ATR_MIN_MULT
        return min(structural_sl, atr_min_sl)
    else:
        recent_high = float(df["high"].iloc[lookback_start:bar_idx + 1].max())
        buffer = recent_high * SL_BUFFER_PCT
        structural_sl = recent_high + buffer
        atr_max_sl = cur_close + atr_val * SL_ATR_MIN_MULT
        return max(structural_sl, atr_max_sl)


# ═══════════════════════════════════════════════════════════════════════════
# BASELINE Position — old logic (2.0/3.5/5.0x, BE after TP2, no V4)
# ═══════════════════════════════════════════════════════════════════════════

class BaselinePosition:
    def __init__(self, sym, direction, entry, sl, atr_val, lot, pip_sz, cont_sz):
        self.sym = sym
        self.direction = direction
        self.entry = entry
        self.sl = sl
        self.atr_val = atr_val
        self.total_lot = lot
        self.remaining_lot = lot
        self.pip_size = pip_sz
        self.contract_size = cont_sz
        # Baseline TPs
        sign = 1 if direction == "BUY" else -1
        self.tp1 = entry + sign * TP1_MULT * atr_val
        self.tp2 = entry + sign * TP2_MULT * atr_val
        self.tp3 = entry + sign * TP3_MULT * atr_val
        self.tp1_hit = False
        self.tp2_hit = False
        self.tp3_hit = False
        self.closed = False
        self.partials = []
        self.bars_in_trade = 0
        self.exit_time = None
        self.max_profit_reached = 0.0
        self.sl_hit = False
        self.final_exit_type = None

    def partial_lot(self):
        return max(0.01, round(self.total_lot / 3, 2))

    def pnl_for_price(self, price, lot):
        if self.direction == 'BUY':
            return (price - self.entry) * lot * self.contract_size
        else:
            return (self.entry - price) * lot * self.contract_size

    def check_bar(self, high, low, close, confirmed_pos):
        if self.closed:
            return []
        events = []
        self.bars_in_trade += 1
        is_buy = self.direction == 'BUY'

        # Track MFE
        cur_profit = (high - self.entry) if is_buy else (self.entry - low)
        if cur_profit > self.max_profit_reached:
            self.max_profit_reached = cur_profit

        # TP1
        if not self.tp1_hit:
            if (is_buy and high >= self.tp1) or (not is_buy and low <= self.tp1):
                self.tp1_hit = True
                partial = min(self.partial_lot(), self.remaining_lot)
                pnl = self.pnl_for_price(self.tp1, partial)
                self.partials.append((self.tp1, partial, pnl, 'TP1'))
                self.remaining_lot = round(self.remaining_lot - partial, 2)
                events.append(('TP1', pnl))
                if self.remaining_lot <= 0:
                    self.closed = True
                    self.final_exit_type = 'TP1'
                    return events

        # TP2 — move SL to BE after TP2
        if self.tp1_hit and not self.tp2_hit:
            if (is_buy and high >= self.tp2) or (not is_buy and low <= self.tp2):
                self.tp2_hit = True
                self.sl = self.entry  # BE
                partial = min(self.partial_lot(), self.remaining_lot)
                pnl = self.pnl_for_price(self.tp2, partial)
                self.partials.append((self.tp2, partial, pnl, 'TP2'))
                self.remaining_lot = round(self.remaining_lot - partial, 2)
                events.append(('TP2', pnl))
                if self.remaining_lot <= 0:
                    self.closed = True
                    self.final_exit_type = 'TP2'
                    return events

        # TP3
        if self.tp2_hit and not self.tp3_hit:
            if (is_buy and high >= self.tp3) or (not is_buy and low <= self.tp3):
                self.tp3_hit = True
                pnl = self.pnl_for_price(self.tp3, self.remaining_lot)
                self.partials.append((self.tp3, self.remaining_lot, pnl, 'TP3'))
                events.append(('TP3', pnl))
                self.remaining_lot = 0
                self.closed = True
                self.final_exit_type = 'TP3'
                return events

        # SL
        if (is_buy and low <= self.sl) or (not is_buy and high >= self.sl):
            self.sl_hit = True
            label = 'SL_AFTER_TP' if self.tp1_hit else 'SL'
            pnl = self.pnl_for_price(self.sl, self.remaining_lot)
            self.partials.append((self.sl, self.remaining_lot, pnl, label))
            self.remaining_lot = 0
            self.closed = True
            self.final_exit_type = label
            events.append((label, pnl))
            return events

        # ZP flip
        if confirmed_pos != 0:
            if (is_buy and confirmed_pos == -1) or (not is_buy and confirmed_pos == 1):
                pnl = self.pnl_for_price(close, self.remaining_lot)
                self.partials.append((close, self.remaining_lot, pnl, 'ZP_FLIP'))
                self.remaining_lot = 0
                self.closed = True
                self.final_exit_type = 'ZP_FLIP'
                events.append(('ZP_FLIP', pnl))
                return events

        return events

    def force_close(self, price):
        if self.closed or self.remaining_lot <= 0:
            return 0
        pnl = self.pnl_for_price(price, self.remaining_lot)
        self.partials.append((price, self.remaining_lot, pnl, 'END'))
        self.remaining_lot = 0
        self.closed = True
        self.final_exit_type = 'END'
        return pnl

    @property
    def total_pnl(self):
        return sum(p[2] for p in self.partials)


# ═══════════════════════════════════════════════════════════════════════════
# V4 PROTECT Position — full profit capture
# ═══════════════════════════════════════════════════════════════════════════

class V4Position:
    """V4 Profit Capture: early BE, micro-partial, post-TP1 trail, stall exit, tighter TPs."""

    def __init__(self, sym, direction, entry, sl, atr_val, lot, pip_sz, cont_sz):
        self.sym = sym
        self.direction = direction
        self.entry = entry
        self.sl = sl
        self.original_sl = sl
        self.atr_val = atr_val
        self.total_lot = lot
        self.remaining_lot = lot
        self.pip_size = pip_sz
        self.contract_size = cont_sz
        # V4 tighter TPs
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
        self.exit_time = None
        self.max_profit_reached = 0.0
        self.max_favorable_price = entry
        self.sl_hit = False
        self.final_exit_type = None
        # V4 state
        self.be_activated = False
        self.profit_lock_sl = None
        self.profit_lock_active = False
        self.micro_tp_hit = False
        self.stall_be_activated = False

    def partial_lot(self):
        return max(0.01, round(self.total_lot / 3, 2))

    def pnl_for_price(self, price, lot):
        if self.direction == 'BUY':
            return (price - self.entry) * lot * self.contract_size
        else:
            return (self.entry - price) * lot * self.contract_size

    def check_bar(self, high, low, close, confirmed_pos):
        if self.closed:
            return []
        events = []
        self.bars_in_trade += 1
        is_buy = self.direction == 'BUY'
        atr = self.atr_val if self.atr_val > 1e-12 else 1.0

        # Track max favorable excursion
        if is_buy:
            cur_profit = high - self.entry
            if high > self.max_favorable_price:
                self.max_favorable_price = high
        else:
            cur_profit = self.entry - low
            if low < self.max_favorable_price:
                self.max_favorable_price = low
        if cur_profit > self.max_profit_reached:
            self.max_profit_reached = cur_profit

        # V4: Post-TP1 trailing stop update
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

        # V4: Early breakeven
        if not self.be_activated:
            if self.max_profit_reached >= BE_TRIGGER_MULT * atr:
                be_buffer = BE_BUFFER_MULT * atr
                if is_buy:
                    new_sl = self.entry + be_buffer
                    if new_sl > self.sl:
                        self.sl = new_sl
                        self.be_activated = True
                else:
                    new_sl = self.entry - be_buffer
                    if new_sl < self.sl:
                        self.sl = new_sl
                        self.be_activated = True

        # V4: Stall exit — move to BE if no TP1 after STALL_BARS
        if not self.tp1_hit and not self.stall_be_activated:
            if self.bars_in_trade >= STALL_BARS:
                be_buffer = BE_BUFFER_MULT * atr
                if is_buy:
                    new_sl = self.entry + be_buffer
                    if new_sl > self.sl:
                        self.sl = new_sl
                        self.stall_be_activated = True
                        self.be_activated = True
                else:
                    new_sl = self.entry - be_buffer
                    if new_sl < self.sl:
                        self.sl = new_sl
                        self.stall_be_activated = True
                        self.be_activated = True

        # V4: Micro-partial at MICRO_TP_MULT * ATR
        if not self.micro_tp_hit and not self.tp1_hit:
            micro_price = (self.entry + MICRO_TP_MULT * atr) if is_buy else (self.entry - MICRO_TP_MULT * atr)
            micro_triggered = (is_buy and high >= micro_price) or (not is_buy and low <= micro_price)
            if micro_triggered:
                self.micro_tp_hit = True
                micro_lot = round(self.total_lot * MICRO_TP_PCT, 2)
                micro_lot = max(0.01, min(micro_lot, self.remaining_lot))
                pnl = self.pnl_for_price(micro_price, micro_lot)
                self.partials.append((micro_price, micro_lot, pnl, 'MICRO_TP'))
                self.remaining_lot = round(self.remaining_lot - micro_lot, 2)
                events.append(('MICRO_TP', pnl))
                if self.remaining_lot <= 0:
                    self.closed = True
                    self.final_exit_type = 'MICRO_TP'
                    return events

        # TP1
        if not self.tp1_hit:
            if (is_buy and high >= self.tp1) or (not is_buy and low <= self.tp1):
                self.tp1_hit = True
                partial = min(self.partial_lot(), self.remaining_lot)
                pnl = self.pnl_for_price(self.tp1, partial)
                self.partials.append((self.tp1, partial, pnl, 'TP1'))
                self.remaining_lot = round(self.remaining_lot - partial, 2)
                events.append(('TP1', pnl))
                if self.remaining_lot <= 0:
                    self.closed = True
                    self.final_exit_type = 'TP1'
                    return events

        # TP2
        if self.tp1_hit and not self.tp2_hit:
            if (is_buy and high >= self.tp2) or (not is_buy and low <= self.tp2):
                self.tp2_hit = True
                self.sl = self.entry
                self.be_activated = True
                partial = min(self.partial_lot(), self.remaining_lot)
                pnl = self.pnl_for_price(self.tp2, partial)
                self.partials.append((self.tp2, partial, pnl, 'TP2'))
                self.remaining_lot = round(self.remaining_lot - partial, 2)
                events.append(('TP2', pnl))
                if self.remaining_lot <= 0:
                    self.closed = True
                    self.final_exit_type = 'TP2'
                    return events

        # TP3
        if self.tp2_hit and not self.tp3_hit:
            if (is_buy and high >= self.tp3) or (not is_buy and low <= self.tp3):
                self.tp3_hit = True
                pnl = self.pnl_for_price(self.tp3, self.remaining_lot)
                self.partials.append((self.tp3, self.remaining_lot, pnl, 'TP3'))
                events.append(('TP3', pnl))
                self.remaining_lot = 0
                self.closed = True
                self.final_exit_type = 'TP3'
                return events

        # V4: Post-TP1 profit lock SL
        if self.profit_lock_active and self.profit_lock_sl is not None:
            lock_hit = (is_buy and low <= self.profit_lock_sl) or \
                       (not is_buy and high >= self.profit_lock_sl)
            if lock_hit:
                pnl = self.pnl_for_price(self.profit_lock_sl, self.remaining_lot)
                self.partials.append((self.profit_lock_sl, self.remaining_lot, pnl, 'PROFIT_LOCK'))
                self.remaining_lot = 0
                self.closed = True
                self.final_exit_type = 'PROFIT_LOCK'
                events.append(('PROFIT_LOCK', pnl))
                return events

        # SL
        if (is_buy and low <= self.sl) or (not is_buy and high >= self.sl):
            if self.stall_be_activated:
                exit_label = 'SL_STALL'
            elif self.be_activated:
                exit_label = 'SL_BE'
            elif self.tp1_hit:
                exit_label = 'SL_AFTER_TP'
            else:
                exit_label = 'SL'
                self.sl_hit = True
            pnl = self.pnl_for_price(self.sl, self.remaining_lot)
            self.partials.append((self.sl, self.remaining_lot, pnl, exit_label))
            self.remaining_lot = 0
            self.closed = True
            self.final_exit_type = exit_label
            events.append((exit_label, pnl))
            return events

        # ZP flip
        if confirmed_pos != 0:
            if (is_buy and confirmed_pos == -1) or (not is_buy and confirmed_pos == 1):
                pnl = self.pnl_for_price(close, self.remaining_lot)
                self.partials.append((close, self.remaining_lot, pnl, 'ZP_FLIP'))
                self.remaining_lot = 0
                self.closed = True
                self.final_exit_type = 'ZP_FLIP'
                events.append(('ZP_FLIP', pnl))
                return events

        return events

    def force_close(self, price):
        if self.closed or self.remaining_lot <= 0:
            return 0
        pnl = self.pnl_for_price(price, self.remaining_lot)
        self.partials.append((price, self.remaining_lot, pnl, 'END'))
        self.remaining_lot = 0
        self.closed = True
        self.final_exit_type = 'END'
        return pnl

    @property
    def total_pnl(self):
        return sum(p[2] for p in self.partials)


# ═══════════════════════════════════════════════════════════════════════════
# Reporting helpers
# ═══════════════════════════════════════════════════════════════════════════

def report_results(label, trades, symbols, starting_balance):
    if not trades:
        print(f"  {label}: No trades!")
        return

    pnls = [t.total_pnl for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    total_pnl = sum(pnls)
    balance = starting_balance + total_pnl
    win_rate = len(wins) / len(pnls) * 100
    avg_win = np.mean(wins) if wins else 0
    avg_loss = np.mean(losses) if losses else 0
    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    expectancy = total_pnl / len(pnls)

    # Max drawdown
    peak = starting_balance
    max_dd = 0
    running = starting_balance
    for p in pnls:
        running += p
        if running > peak:
            peak = running
        dd = peak - running
        if dd > max_dd:
            max_dd = dd

    first_time = min(t.exit_time for t in trades if t.exit_time is not None)
    last_time = max(t.exit_time for t in trades if t.exit_time is not None)
    days_tested = (last_time - first_time).days

    print(f"\n{'=' * 100}")
    print(f"  {label}")
    print(f"{'=' * 100}")
    print(f"\n  Period: {first_time.strftime('%Y-%m-%d')} -> {last_time.strftime('%Y-%m-%d')} ({days_tested} days)")
    print(f"  Signals:           {len(trades)}")
    print(f"  Starting balance:  ${starting_balance:.2f}")
    print(f"  Final balance:     ${balance:.2f}")
    print(f"  Total PnL:         ${total_pnl:+.2f}")
    print(f"  Return:            {(total_pnl / starting_balance * 100):+.1f}%")
    print(f"\n  Net winners:       {len(wins)} ({win_rate:.1f}%)")
    print(f"  Net losers:        {len(losses)} ({100 - win_rate:.1f}%)")
    print(f"  Avg win:           ${avg_win:.2f}")
    print(f"  Avg loss:          ${avg_loss:.2f}")
    print(f"  Expectancy:        ${expectancy:.2f} per trade")
    print(f"  Profit factor:     {profit_factor:.2f}")
    print(f"  Max drawdown:      ${max_dd:.2f}")

    # Per-symbol
    print(f"\n{'Symbol':>8} | {'Trades':>7} | {'Win%':>6} | {'PnL':>10} | {'Avg Win':>9} | {'Avg Loss':>9} | {'PF':>6}")
    print("-" * 75)
    for sym in sorted(symbols):
        st = [t for t in trades if t.sym == sym]
        if not st:
            continue
        sp = [t.total_pnl for t in st]
        sw = [p for p in sp if p > 0]
        sl = [p for p in sp if p <= 0]
        wr = len(sw) / len(sp) * 100
        pnl = sum(sp)
        aw = np.mean(sw) if sw else 0
        al = np.mean(sl) if sl else 0
        gp = sum(sw)
        gl = abs(sum(sl))
        pf = gp / gl if gl > 0 else float('inf')
        print(f"{sym:>8} | {len(sp):>7} | {wr:>5.1f}% | ${pnl:>+8.2f} | ${aw:>7.2f} | ${al:>7.2f} | {pf:>5.2f}")

    # Exit types
    print(f"\n  Exit Type Breakdown:")
    exit_counter = Counter()
    exit_pnl = defaultdict(float)
    for t in trades:
        for _, _, pnl, label_e in t.partials:
            exit_counter[label_e] += 1
            exit_pnl[label_e] += pnl
    total_exits = sum(exit_counter.values())
    print(f"  {'Exit Type':>14} | {'Count':>7} | {'%':>6} | {'Total PnL':>11} | {'Avg PnL':>9}")
    print(f"  " + "-" * 63)
    for label_e in sorted(exit_counter.keys(), key=lambda x: -exit_counter[x]):
        cnt = exit_counter[label_e]
        pct = cnt / total_exits * 100
        tot = exit_pnl[label_e]
        avg = tot / cnt
        print(f"  {label_e:>14} | {cnt:>7} | {pct:>5.1f}% | ${tot:>+9.2f} | ${avg:>+7.2f}")

    # TP hit rates
    tp1_hits = sum(1 for t in trades if t.tp1_hit)
    tp2_hits = sum(1 for t in trades if t.tp2_hit)
    tp3_hits = sum(1 for t in trades if t.tp3_hit)
    n_t = len(trades)
    print(f"\n  TP1 hit:  {tp1_hits:>5} / {n_t} ({tp1_hits/n_t*100:.1f}%)")
    print(f"  TP2 hit:  {tp2_hits:>5} / {n_t} ({tp2_hits/n_t*100:.1f}%)")
    print(f"  TP3 hit:  {tp3_hits:>5} / {n_t} ({tp3_hits/n_t*100:.1f}%)")

    # Outcome breakdown
    tp3_full = sum(1 for t in trades if t.tp3_hit)
    tp2_then_exit = sum(1 for t in trades if t.tp2_hit and not t.tp3_hit)
    tp1_then_exit = sum(1 for t in trades if t.tp1_hit and not t.tp2_hit)
    no_tp = sum(1 for t in trades if not t.tp1_hit)
    print(f"\n  Full TP3 runners:  {tp3_full:>5} ({tp3_full/n_t*100:.1f}%)")
    print(f"  TP2 then exit:     {tp2_then_exit:>5} ({tp2_then_exit/n_t*100:.1f}%)")
    print(f"  TP1 then exit:     {tp1_then_exit:>5} ({tp1_then_exit/n_t*100:.1f}%)")
    print(f"  NO TP hit:         {no_tp:>5} ({no_tp/n_t*100:.1f}%)")

    return {
        "total_pnl": total_pnl, "balance": balance, "win_rate": win_rate,
        "profit_factor": profit_factor, "max_dd": max_dd, "expectancy": expectancy,
        "trades": len(trades), "wins": len(wins), "losses": len(losses),
        "tp1_rate": tp1_hits / n_t * 100 if n_t > 0 else 0,
    }


def main():
    print("=" * 100)
    print("  ZEROPOINT PRO -- V4 PROFIT CAPTURE BACKTEST")
    print(f"  BASELINE: TP {TP1_MULT}/{TP2_MULT}/{TP3_MULT}x ATR | BE after TP2 | No protection")
    print(f"  V4 PROTECT: TP {TP1_MULT_AGG}/{TP2_MULT_AGG}/{TP3_MULT_AGG}x ATR | Early BE @ {BE_TRIGGER_MULT}x | Micro {MICRO_TP_PCT*100:.0f}% @ {MICRO_TP_MULT}x | Stall {STALL_BARS}bar | Trail {PROFIT_TRAIL_DISTANCE_MULT}x")
    print(f"  Lot: {LOT_SIZE} | Start: ${STARTING_BALANCE}")
    print("=" * 100)

    if not mt5.initialize():
        print("ERROR: Could not initialize MT5")
        return

    acct = mt5.account_info()
    if acct:
        print(f"  MT5: Account {acct.login} | Leverage 1:{acct.leverage}")

    symbol_data = {}
    print(f"\nFetching up to {FETCH_BARS} H4 bars per symbol...")
    for sym in SYMBOLS:
        resolved = resolve_symbol(sym)
        if resolved is None:
            print(f"  {sym}: SKIP (cannot resolve)")
            continue
        rates = mt5.copy_rates_from_pos(resolved, mt5.TIMEFRAME_H4, 0, FETCH_BARS)
        if rates is None or len(rates) < 100:
            print(f"  {sym}: SKIP ({len(rates) if rates is not None else 0} bars)")
            continue
        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        df.rename(columns={"tick_volume": "volume"}, inplace=True)
        df_zp = compute_zeropoint_state(df)
        if df_zp is None or len(df_zp) < WARMUP_BARS:
            print(f"  {sym}: SKIP (ZP failed)")
            continue
        symbol_data[sym] = df_zp
        days = (df_zp["time"].iloc[-1] - df_zp["time"].iloc[WARMUP_BARS]).days
        print(f"  {sym}: {len(df_zp)} bars ({days} days)")

    if not symbol_data:
        print("ERROR: No data loaded")
        mt5.shutdown()
        return

    # Simulate BOTH modes on the same signals
    print(f"\nSimulating trades (BASELINE + V4 PROTECT)...")
    baseline_trades = []
    v4_trades = []

    for sym, df in symbol_data.items():
        pip_sz = pip_value(sym)
        cont_sz = contract_size(sym)
        n = len(df)
        baseline_pos = None
        v4_pos = None

        for i in range(WARMUP_BARS, n):
            row = df.iloc[i]
            high = float(row["high"])
            low = float(row["low"])
            close = float(row["close"])
            atr_val = float(row["atr"])
            pos = int(row.get("pos", 0))
            buy_sig = bool(row.get("buy_signal", False))
            sell_sig = bool(row.get("sell_signal", False))

            if np.isnan(atr_val) or atr_val <= 0:
                continue

            # Check active positions
            if baseline_pos is not None and not baseline_pos.closed:
                baseline_pos.check_bar(high, low, close, pos)
                if baseline_pos.closed:
                    baseline_pos.exit_time = row["time"]
                    baseline_trades.append(baseline_pos)
                    baseline_pos = None

            if v4_pos is not None and not v4_pos.closed:
                v4_pos.check_bar(high, low, close, pos)
                if v4_pos.closed:
                    v4_pos.exit_time = row["time"]
                    v4_trades.append(v4_pos)
                    v4_pos = None

            # Open on signal — same entry for both modes
            if buy_sig or sell_sig:
                direction = "BUY" if buy_sig else "SELL"
                entry = close
                smart_sl = compute_smart_sl(df, i, direction, atr_val)

                # Close existing positions on flip
                if baseline_pos is not None and not baseline_pos.closed:
                    baseline_pos.force_close(close)
                    baseline_pos.exit_time = row["time"]
                    baseline_trades.append(baseline_pos)

                if v4_pos is not None and not v4_pos.closed:
                    v4_pos.force_close(close)
                    v4_pos.exit_time = row["time"]
                    v4_trades.append(v4_pos)

                # Open new positions
                baseline_pos = BaselinePosition(
                    sym=sym, direction=direction, entry=entry, sl=smart_sl,
                    atr_val=atr_val, lot=LOT_SIZE, pip_sz=pip_sz, cont_sz=cont_sz,
                )

                v4_pos = V4Position(
                    sym=sym, direction=direction, entry=entry, sl=smart_sl,
                    atr_val=atr_val, lot=LOT_SIZE, pip_sz=pip_sz, cont_sz=cont_sz,
                )

        # Force close at end
        if baseline_pos is not None and not baseline_pos.closed:
            baseline_pos.force_close(float(df.iloc[-1]["close"]))
            baseline_pos.exit_time = df.iloc[-1]["time"]
            baseline_trades.append(baseline_pos)

        if v4_pos is not None and not v4_pos.closed:
            v4_pos.force_close(float(df.iloc[-1]["close"]))
            v4_pos.exit_time = df.iloc[-1]["time"]
            v4_trades.append(v4_pos)

    mt5.shutdown()

    if not baseline_trades and not v4_trades:
        print("No trades generated!")
        return

    symbols_list = sorted(symbol_data.keys())

    # Report both
    baseline_stats = report_results(
        f"BASELINE (TP {TP1_MULT}/{TP2_MULT}/{TP3_MULT}x ATR — original ZeroPoint)",
        baseline_trades, symbols_list, STARTING_BALANCE,
    )

    v4_stats = report_results(
        f"V4 PROTECT (TP {TP1_MULT_AGG}/{TP2_MULT_AGG}/{TP3_MULT_AGG}x ATR + Early BE + Micro + Stall + Trail)",
        v4_trades, symbols_list, STARTING_BALANCE,
    )

    # Comparison
    if baseline_stats and v4_stats:
        print(f"\n{'=' * 100}")
        print(f"  HEAD-TO-HEAD COMPARISON")
        print(f"{'=' * 100}")
        print(f"\n  {'Metric':<25} | {'BASELINE':>15} | {'V4 PROTECT':>15} | {'Delta':>12}")
        print(f"  " + "-" * 75)

        metrics = [
            ("Win Rate", f"{baseline_stats['win_rate']:.1f}%", f"{v4_stats['win_rate']:.1f}%",
             f"{v4_stats['win_rate'] - baseline_stats['win_rate']:+.1f}%"),
            ("Profit Factor", f"{baseline_stats['profit_factor']:.2f}", f"{v4_stats['profit_factor']:.2f}",
             f"{v4_stats['profit_factor'] - baseline_stats['profit_factor']:+.2f}"),
            ("Total PnL", f"${baseline_stats['total_pnl']:+,.2f}", f"${v4_stats['total_pnl']:+,.2f}",
             f"${v4_stats['total_pnl'] - baseline_stats['total_pnl']:+,.2f}"),
            ("Final Balance", f"${baseline_stats['balance']:,.2f}", f"${v4_stats['balance']:,.2f}", ""),
            ("Expectancy", f"${baseline_stats['expectancy']:.2f}", f"${v4_stats['expectancy']:.2f}",
             f"${v4_stats['expectancy'] - baseline_stats['expectancy']:+.2f}"),
            ("Max Drawdown", f"${baseline_stats['max_dd']:,.2f}", f"${v4_stats['max_dd']:,.2f}",
             f"${v4_stats['max_dd'] - baseline_stats['max_dd']:+,.2f}"),
            ("TP1 Hit Rate", f"{baseline_stats['tp1_rate']:.1f}%", f"{v4_stats['tp1_rate']:.1f}%",
             f"{v4_stats['tp1_rate'] - baseline_stats['tp1_rate']:+.1f}%"),
        ]

        for name, base_val, v4_val, delta in metrics:
            print(f"  {name:<25} | {base_val:>15} | {v4_val:>15} | {delta:>12}")

        # Trade-by-trade comparison
        print(f"\n  Trade-by-Trade Analysis (same entries, different exits):")
        improved = 0
        worsened = 0
        same = 0
        for bt, vt in zip(baseline_trades, v4_trades):
            diff = vt.total_pnl - bt.total_pnl
            if diff > 0.01:
                improved += 1
            elif diff < -0.01:
                worsened += 1
            else:
                same += 1
        total_comp = improved + worsened + same
        print(f"    V4 improved:   {improved:>5} ({improved/total_comp*100:.1f}%)")
        print(f"    V4 worsened:   {worsened:>5} ({worsened/total_comp*100:.1f}%)")
        print(f"    Same:          {same:>5} ({same/total_comp*100:.1f}%)")

        # Show the deltas for improved vs worsened
        improved_delta = sum(vt.total_pnl - bt.total_pnl for bt, vt in zip(baseline_trades, v4_trades)
                           if vt.total_pnl - bt.total_pnl > 0.01)
        worsened_delta = sum(vt.total_pnl - bt.total_pnl for bt, vt in zip(baseline_trades, v4_trades)
                            if vt.total_pnl - bt.total_pnl < -0.01)
        print(f"    Saved by V4:   ${improved_delta:+,.2f}")
        print(f"    Lost by V4:    ${worsened_delta:+,.2f}")
        print(f"    Net benefit:   ${improved_delta + worsened_delta:+,.2f}")

    # ═══════════════════════════════════════════════════════════════════════
    # DEEP LOSS ANALYSIS — V4 PROTECT losers only
    # ═══════════════════════════════════════════════════════════════════════
    v4_losers = [t for t in v4_trades if t.total_pnl <= 0]
    if v4_losers:
        print(f"\n{'=' * 100}")
        print(f"  DEEP LOSS ANALYSIS — V4 PROTECT ({len(v4_losers)} losing trades)")
        print(f"{'=' * 100}")

        # 1) Exit type breakdown for losers
        loser_exit_counter = Counter()
        loser_exit_pnl = defaultdict(list)
        for t in v4_losers:
            loser_exit_counter[t.final_exit_type] += 1
            loser_exit_pnl[t.final_exit_type].append(t.total_pnl)

        print(f"\n  1) EXIT TYPES PRODUCING LOSSES:")
        print(f"  {'Exit Type':>14} | {'Count':>5} | {'% of losers':>12} | {'Total Loss':>12} | {'Avg Loss':>10} | {'Max Loss':>10} | {'Min Loss':>10}")
        print(f"  " + "-" * 95)
        for et in sorted(loser_exit_counter.keys(), key=lambda x: -loser_exit_counter[x]):
            cnt = loser_exit_counter[et]
            pnls_list = loser_exit_pnl[et]
            pct = cnt / len(v4_losers) * 100
            tot = sum(pnls_list)
            avg = np.mean(pnls_list)
            mx = max(pnls_list)
            mn = min(pnls_list)
            print(f"  {et:>14} | {cnt:>5} | {pct:>10.1f}% | ${tot:>+10.2f} | ${avg:>+8.2f} | ${mx:>+8.2f} | ${mn:>+8.2f}")

        # 2) Per-symbol loss breakdown
        print(f"\n  2) SYMBOL BREAKDOWN OF LOSSES:")
        sym_losers = defaultdict(list)
        for t in v4_losers:
            sym_losers[t.sym].append(t)
        print(f"  {'Symbol':>8} | {'Losing':>7} | {'Total':>7} | {'Loss%':>6} | {'Total Loss':>12} | {'Avg Loss':>10} | {'Max Loss':>10} | {'Worst Exit':>12}")
        print(f"  " + "-" * 100)
        for sym in sorted(sym_losers.keys()):
            losers = sym_losers[sym]
            total_sym = len([t for t in v4_trades if t.sym == sym])
            loss_pct = len(losers) / total_sym * 100
            pnls_list = [t.total_pnl for t in losers]
            tot = sum(pnls_list)
            avg = np.mean(pnls_list)
            worst = min(pnls_list)
            worst_exit = min(losers, key=lambda t: t.total_pnl).final_exit_type
            print(f"  {sym:>8} | {len(losers):>7} | {total_sym:>7} | {loss_pct:>5.1f}% | ${tot:>+10.2f} | ${avg:>+8.2f} | ${worst:>+8.2f} | {worst_exit:>12}")

        # 3) Individual losing trade details
        print(f"\n  3) ALL {len(v4_losers)} LOSING TRADES (sorted by PnL):")
        print(f"  {'#':>3} | {'Symbol':>8} | {'Dir':>4} | {'Exit Type':>12} | {'PnL':>12} | {'Bars':>5} | {'ATR':>8} | {'SL dist (ATR)':>14} | {'MFE (ATR)':>10} | {'Exit Time'}")
        print(f"  " + "-" * 120)
        for idx, t in enumerate(sorted(v4_losers, key=lambda x: x.total_pnl), 1):
            sl_dist_atr = abs(t.entry - t.original_sl) / t.atr_val if t.atr_val > 0 else 0
            mfe_atr = t.max_profit_reached / t.atr_val if t.atr_val > 0 else 0
            exit_str = t.exit_time.strftime('%Y-%m-%d %H:%M') if t.exit_time else 'N/A'
            print(f"  {idx:>3} | {t.sym:>8} | {t.direction:>4} | {t.final_exit_type:>12} | ${t.total_pnl:>+10.2f} | {t.bars_in_trade:>5} | {t.atr_val:>8.5f} | {sl_dist_atr:>13.2f} | {mfe_atr:>9.2f} | {exit_str}")

        # 4) Loss clustering — time gaps between consecutive losers
        print(f"\n  4) LOSS CLUSTERING ANALYSIS:")
        losers_sorted = sorted(v4_losers, key=lambda t: t.exit_time if t.exit_time else pd.Timestamp.min)
        if len(losers_sorted) >= 2:
            gaps_days = []
            for i in range(1, len(losers_sorted)):
                if losers_sorted[i].exit_time and losers_sorted[i-1].exit_time:
                    gap = (losers_sorted[i].exit_time - losers_sorted[i-1].exit_time).days
                    gaps_days.append(gap)
            if gaps_days:
                print(f"    Avg days between losses: {np.mean(gaps_days):.1f}")
                print(f"    Min days between losses: {min(gaps_days)}")
                print(f"    Max days between losses: {max(gaps_days)}")
                # Check for clusters (< 3 days apart)
                clusters = []
                current_cluster = [losers_sorted[0]]
                for i in range(1, len(losers_sorted)):
                    if losers_sorted[i].exit_time and losers_sorted[i-1].exit_time:
                        gap = (losers_sorted[i].exit_time - losers_sorted[i-1].exit_time).days
                        if gap <= 3:
                            current_cluster.append(losers_sorted[i])
                        else:
                            if len(current_cluster) >= 2:
                                clusters.append(current_cluster)
                            current_cluster = [losers_sorted[i]]
                if len(current_cluster) >= 2:
                    clusters.append(current_cluster)
                print(f"    Loss clusters (<=3 days apart): {len(clusters)}")
                for ci, cluster in enumerate(clusters, 1):
                    syms = [t.sym for t in cluster]
                    dates = [t.exit_time.strftime('%Y-%m-%d') for t in cluster if t.exit_time]
                    total_cluster_loss = sum(t.total_pnl for t in cluster)
                    print(f"      Cluster {ci}: {len(cluster)} trades | {dates[0]} -> {dates[-1]} | Symbols: {', '.join(syms)} | Loss: ${total_cluster_loss:+.2f}")

        # 5) Monthly distribution
        print(f"\n  5) MONTHLY LOSS DISTRIBUTION:")
        monthly_losses = defaultdict(lambda: {"count": 0, "pnl": 0.0})
        for t in v4_losers:
            if t.exit_time:
                month_key = t.exit_time.strftime('%Y-%m')
                monthly_losses[month_key]["count"] += 1
                monthly_losses[month_key]["pnl"] += t.total_pnl
        print(f"  {'Month':>8} | {'Losers':>7} | {'Total Loss':>12}")
        print(f"  " + "-" * 35)
        for month in sorted(monthly_losses.keys()):
            data = monthly_losses[month]
            print(f"  {month:>8} | {data['count']:>7} | ${data['pnl']:>+10.2f}")

        # 6) Direction bias
        print(f"\n  6) DIRECTION BIAS:")
        buy_losers = [t for t in v4_losers if t.direction == 'BUY']
        sell_losers = [t for t in v4_losers if t.direction == 'SELL']
        buy_total = len([t for t in v4_trades if t.direction == 'BUY'])
        sell_total = len([t for t in v4_trades if t.direction == 'SELL'])
        print(f"    BUY losses:  {len(buy_losers):>3} / {buy_total} total BUY trades ({len(buy_losers)/buy_total*100:.1f}% loss rate)")
        print(f"    SELL losses: {len(sell_losers):>3} / {sell_total} total SELL trades ({len(sell_losers)/sell_total*100:.1f}% loss rate)")
        print(f"    BUY avg loss:  ${np.mean([t.total_pnl for t in buy_losers]):+.2f}" if buy_losers else "    BUY avg loss:  N/A")
        print(f"    SELL avg loss: ${np.mean([t.total_pnl for t in sell_losers]):+.2f}" if sell_losers else "    SELL avg loss: N/A")

        # 7) MFE analysis — how much profit the losers DID see before reversing
        print(f"\n  7) MAX FAVORABLE EXCURSION (MFE) — how much profit losers saw before losing:")
        mfe_atr_values = [t.max_profit_reached / t.atr_val if t.atr_val > 0 else 0 for t in v4_losers]
        print(f"    Avg MFE (ATR multiples):  {np.mean(mfe_atr_values):.3f}")
        print(f"    Max MFE (ATR multiples):  {max(mfe_atr_values):.3f}")
        print(f"    Min MFE (ATR multiples):  {min(mfe_atr_values):.3f}")
        mfe_zero = sum(1 for m in mfe_atr_values if m < 0.1)
        mfe_tiny = sum(1 for m in mfe_atr_values if 0.1 <= m < 0.5)
        mfe_ok = sum(1 for m in mfe_atr_values if m >= 0.5)
        print(f"    MFE < 0.1 ATR (immediate reversal): {mfe_zero} trades")
        print(f"    MFE 0.1-0.5 ATR (small move then reversal): {mfe_tiny} trades")
        print(f"    MFE >= 0.5 ATR (decent move, then gave back): {mfe_ok} trades")

        # 8) SL distance analysis — how far SL was
        print(f"\n  8) STOP LOSS DISTANCE ANALYSIS:")
        sl_dist_values = [abs(t.entry - t.original_sl) / t.atr_val if t.atr_val > 0 else 0 for t in v4_losers]
        print(f"    Avg SL distance (ATR multiples): {np.mean(sl_dist_values):.2f}")
        print(f"    Max SL distance (ATR multiples): {max(sl_dist_values):.2f}")
        print(f"    Min SL distance (ATR multiples): {min(sl_dist_values):.2f}")
        print(f"    Median SL distance (ATR mult.):  {np.median(sl_dist_values):.2f}")

        # 9) Bars in trade for losers vs winners
        loser_bars = [t.bars_in_trade for t in v4_losers]
        winner_bars = [t.bars_in_trade for t in v4_trades if t.total_pnl > 0]
        print(f"\n  9) TRADE DURATION (bars in trade):")
        print(f"    Losers — avg: {np.mean(loser_bars):.1f}, median: {np.median(loser_bars):.0f}, min: {min(loser_bars)}, max: {max(loser_bars)}")
        print(f"    Winners — avg: {np.mean(winner_bars):.1f}, median: {np.median(winner_bars):.0f}, min: {min(winner_bars)}, max: {max(winner_bars)}")

    print(f"\n{'=' * 100}")
    print(f"  SUMMARY")
    print(f"{'=' * 100}")
    if baseline_stats and v4_stats:
        print(f"\n  BASELINE:   {baseline_stats['win_rate']:.1f}% WR | PF {baseline_stats['profit_factor']:.2f} | ${baseline_stats['total_pnl']:+,.2f}")
        print(f"  V4 PROTECT: {v4_stats['win_rate']:.1f}% WR | PF {v4_stats['profit_factor']:.2f} | ${v4_stats['total_pnl']:+,.2f}")
        pnl_diff = v4_stats['total_pnl'] - baseline_stats['total_pnl']
        print(f"  V4 vs BASE: {v4_stats['win_rate'] - baseline_stats['win_rate']:+.1f}% WR | PF {v4_stats['profit_factor'] - baseline_stats['profit_factor']:+.2f} | ${pnl_diff:+,.2f}")
    print(f"{'=' * 100}")


if __name__ == "__main__":
    main()
