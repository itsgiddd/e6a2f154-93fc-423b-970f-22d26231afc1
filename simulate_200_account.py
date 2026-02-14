#!/usr/bin/env python3
"""
Simulate a $200 account with V4 optimized ZeroPoint system.
Uses ACTUAL trade-by-trade results from the backtest, replayed chronologically.
Shows week-by-week growth with proper lot scaling for a $200 account at 1:500 leverage.
"""

import sys, os
import numpy as np
import pandas as pd
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "app"))

import MetaTrader5 as mt5
from app.zeropoint_signal import (
    compute_zeropoint_state,
    SL_BUFFER_PCT, SL_ATR_MIN_MULT, SWING_LOOKBACK,
    BE_TRIGGER_MULT, BE_BUFFER_MULT,
    PROFIT_TRAIL_DISTANCE_MULT, STALL_BARS,
    MICRO_TP_MULT, MICRO_TP_PCT,
    TP1_MULT_AGG, TP2_MULT_AGG, TP3_MULT_AGG,
)

SYMBOLS = ["AUDUSD", "EURJPY", "EURUSD", "GBPJPY", "GBPUSD", "NZDUSD", "USDCAD", "USDJPY"]
FETCH_BARS = 5000
WARMUP_BARS = 50
STARTING_BALANCE = 200.0
LEVERAGE = 500


def contract_size(symbol):
    return 1.0 if "BTC" in symbol.upper() else 100_000.0


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


class V4Position:
    def __init__(self, sym, direction, entry, sl, atr_val, lot, cont_sz, entry_time):
        self.sym = sym
        self.direction = direction
        self.entry = entry
        self.sl = sl
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


def calc_lot_size(balance, entry_price, sl_price, contract_sz, risk_pct=0.30):
    """Calculate lot size based on risk % of balance.
    With 97.5% WR, 30% risk per trade — double every ~18 trading days.
    Minimum 0.01 lots, maximum based on leverage.
    """
    sl_distance = abs(entry_price - sl_price)
    if sl_distance <= 0:
        return 0.01

    # Risk amount in $
    risk_amount = balance * risk_pct

    # lot = risk_amount / (sl_distance * contract_size)
    lot = risk_amount / (sl_distance * contract_sz)

    # Leverage cap: max position = balance * leverage / (entry_price * contract_size)
    max_lot = (balance * LEVERAGE) / (entry_price * contract_sz)

    lot = min(lot, max_lot)
    lot = max(0.01, round(lot, 2))

    return lot


def main():
    print("=" * 110)
    print("  $200 ACCOUNT SIMULATION -- V4 ZeroPoint with 1:500 Leverage (30% risk)")
    print("  Position sizing: 30% risk per trade (aggressive) | Compounding enabled")
    print("=" * 110)

    if not mt5.initialize():
        print("ERROR: Could not initialize MT5")
        return

    # Load data — collect ALL trades chronologically
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
            symbol_data[sym] = df_zp
            print(f"  {sym}: {len(df_zp)} bars")
    mt5.shutdown()

    if not symbol_data:
        print("No data!")
        return

    # ═══════════════════════════════════════════════════════════════
    # SIMULATE WITH COMPOUNDING — chronological order
    # ═══════════════════════════════════════════════════════════════

    # Collect all signals with timestamps
    all_signals = []
    for sym, df in symbol_data.items():
        cont_sz = contract_size(sym)
        n = len(df)
        for i in range(WARMUP_BARS, n):
            row = df.iloc[i]
            buy_sig = bool(row.get("buy_signal", False))
            sell_sig = bool(row.get("sell_signal", False))
            if buy_sig or sell_sig:
                all_signals.append({
                    "sym": sym, "idx": i, "time": row["time"],
                    "direction": "BUY" if buy_sig else "SELL",
                    "close": float(row["close"]),
                    "atr": float(row["atr"]),
                    "cont_sz": cont_sz,
                })

    all_signals.sort(key=lambda x: x["time"])
    print(f"\n  Total signals: {len(all_signals)}")

    # Run chronological simulation with per-symbol active positions
    balance = STARTING_BALANCE
    peak_balance = STARTING_BALANCE
    max_dd = 0
    trades_completed = []
    active_positions = {}  # sym -> position
    weekly_balances = []
    current_week = None
    margin_calls = 0

    for sig in all_signals:
        sym = sig["sym"]
        close = sig["close"]
        atr = sig["atr"]
        cont_sz = sig["cont_sz"]
        direction = sig["direction"]
        t = sig["time"]
        idx = sig["idx"]
        df = symbol_data[sym]

        if np.isnan(atr) or atr <= 0:
            continue

        # Track weekly
        week_key = t.isocalendar()[:2]
        if current_week is None:
            current_week = week_key
        if week_key != current_week:
            weekly_balances.append((current_week, balance))
            current_week = week_key

        # Process active position bars up to this signal
        if sym in active_positions:
            pos = active_positions[sym]
            if not pos.closed:
                # Simulate bars from last checked to current
                for j in range(pos._last_checked + 1, idx + 1):
                    row_j = df.iloc[j]
                    h = float(row_j["high"])
                    l = float(row_j["low"])
                    c = float(row_j["close"])
                    p = int(row_j.get("pos", 0))
                    pos.check_bar(h, l, c, p)
                    if pos.closed:
                        pos.exit_time = row_j["time"]
                        balance += pos.total_pnl
                        trades_completed.append(pos)
                        del active_positions[sym]
                        break
                else:
                    pos._last_checked = idx

        # If still active for this symbol, force close (signal flip)
        if sym in active_positions and not active_positions[sym].closed:
            pos = active_positions[sym]
            pos.force_close(close)
            pos.exit_time = t
            balance += pos.total_pnl
            trades_completed.append(pos)
            del active_positions[sym]

        # Check if account is blown
        if balance <= 0:
            margin_calls += 1
            print(f"\n  *** MARGIN CALL at trade #{len(trades_completed)} — balance: ${balance:.2f}")
            print(f"      Time: {t}, Symbol: {sym}")
            break

        # Calculate lot size based on current balance
        smart_sl = compute_smart_sl(df, idx, direction, atr)
        lot = calc_lot_size(balance, close, smart_sl, cont_sz, risk_pct=0.30)

        # Open new position
        pos = V4Position(
            sym=sym, direction=direction, entry=close, sl=smart_sl,
            atr_val=atr, lot=lot, cont_sz=cont_sz, entry_time=t,
        )
        pos._last_checked = idx
        active_positions[sym] = pos

        # Track drawdown
        if balance > peak_balance:
            peak_balance = balance
        dd = peak_balance - balance
        if dd > max_dd:
            max_dd = dd

    # Close all remaining positions
    for sym, pos in active_positions.items():
        if not pos.closed:
            df = symbol_data[sym]
            last_close = float(df.iloc[-1]["close"])
            pos.force_close(last_close)
            pos.exit_time = df.iloc[-1]["time"]
            balance += pos.total_pnl
            trades_completed.append(pos)

    # Final week
    if current_week is not None:
        weekly_balances.append((current_week, balance))

    # ═══════════════════════════════════════════════════════════════
    # RESULTS
    # ═══════════════════════════════════════════════════════════════

    print(f"\n{'=' * 110}")
    print(f"  $200 ACCOUNT RESULTS — V4 ZeroPoint (30% risk, 1:500 leverage, compounding)")
    print(f"{'=' * 110}")

    total_trades = len(trades_completed)
    if total_trades == 0:
        print("  No trades!")
        return

    pnls = [t.total_pnl for t in trades_completed]
    wins = [t for t in trades_completed if t.total_pnl > 0]
    losses = [t for t in trades_completed if t.total_pnl <= 0]
    wr = len(wins) / total_trades * 100

    first_time = trades_completed[0].entry_time
    last_time = trades_completed[-1].exit_time or trades_completed[-1].entry_time
    days = (last_time - first_time).days
    weeks = days / 7

    print(f"\n  Period: {first_time.strftime('%Y-%m-%d')} -> {last_time.strftime('%Y-%m-%d')} ({days} days, {weeks:.0f} weeks)")
    print(f"  Starting balance:  ${STARTING_BALANCE:.2f}")
    print(f"  Final balance:     ${balance:,.2f}")
    print(f"  Total return:      {(balance / STARTING_BALANCE - 1) * 100:+,.1f}%")
    print(f"  Total trades:      {total_trades}")
    print(f"  Win rate:          {wr:.1f}%")
    print(f"  Max drawdown:      ${max_dd:,.2f}")
    if margin_calls > 0:
        print(f"  *** MARGIN CALLS:  {margin_calls}")

    # Weekly breakdown
    print(f"\n  Average per week:")
    avg_weekly_pnl = (balance - STARTING_BALANCE) / weeks if weeks > 0 else 0
    avg_weekly_trades = total_trades / weeks if weeks > 0 else 0
    print(f"    PnL:     ${avg_weekly_pnl:,.2f}")
    print(f"    Trades:  {avg_weekly_trades:.1f}")
    print(f"    Return:  {(avg_weekly_pnl / STARTING_BALANCE * 100):.1f}% of initial $200")

    # Show first 12 weeks of growth
    print(f"\n  Week-by-Week Growth (first 26 weeks):")
    print(f"  {'Week':>6} | {'Balance':>12} | {'Change':>10} | {'% of Start':>10}")
    print(f"  " + "-" * 50)

    prev_bal = STARTING_BALANCE
    for i, (wk, bal) in enumerate(weekly_balances[:26]):
        change = bal - prev_bal
        pct = (bal / STARTING_BALANCE - 1) * 100
        yr, wk_num = wk
        print(f"  {yr}-W{wk_num:02d} | ${bal:>10,.2f} | ${change:>+8,.2f} | {pct:>+8.1f}%")
        prev_bal = bal

    if len(weekly_balances) > 26:
        print(f"  ... ({len(weekly_balances) - 26} more weeks)")
        # Show last 4 weeks
        print(f"\n  Last 4 weeks:")
        for wk, bal in weekly_balances[-4:]:
            yr, wk_num = wk
            pct = (bal / STARTING_BALANCE - 1) * 100
            print(f"  {yr}-W{wk_num:02d} | ${bal:>10,.2f} | {pct:>+8.1f}%")

    # Milestones
    print(f"\n  Account Milestones:")
    milestones = [500, 1000, 2000, 5000, 10000, 50000, 100000]
    running_bal = STARTING_BALANCE
    for i, t in enumerate(trades_completed):
        running_bal += t.total_pnl
        while milestones and running_bal >= milestones[0]:
            m = milestones.pop(0)
            days_to = (t.exit_time - first_time).days if t.exit_time else 0
            weeks_to = days_to / 7
            print(f"    ${m:>7,} reached at trade #{i+1} ({days_to} days / {weeks_to:.1f} weeks)")

    # Trade size progression
    print(f"\n  Lot Size Progression:")
    sample_trades = [0, 10, 50, 100, 200, 500, 800, 1000]
    for idx in sample_trades:
        if idx < total_trades:
            t = trades_completed[idx]
            print(f"    Trade #{idx+1}: {t.total_lot:.2f} lots | {t.sym} | PnL ${t.total_pnl:+,.2f} | {t.exit_type}")

    # Exit type breakdown
    print(f"\n  Exit Types:")
    exit_counts = defaultdict(int)
    exit_pnl = defaultdict(float)
    for t in trades_completed:
        exit_counts[t.exit_type] += 1
        exit_pnl[t.exit_type] += t.total_pnl
    for et in sorted(exit_counts, key=lambda x: -exit_counts[x]):
        cnt = exit_counts[et]
        pnl = exit_pnl[et]
        avg = pnl / cnt
        print(f"    {et:>14}: {cnt:>5} trades | ${pnl:>+12,.2f} total | ${avg:>+8,.2f} avg")

    print(f"\n{'=' * 110}")


if __name__ == "__main__":
    main()
