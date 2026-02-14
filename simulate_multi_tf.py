#!/usr/bin/env python3
"""
Multi-Timeframe $200 Account Simulation
========================================
Tests H4, H1, and M15 separately AND combined to see how fast we can compound.
More timeframes = more signals = faster growth.
"""

import sys, os, time
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
WARMUP_BARS = 50
STARTING_BALANCE = 200.0
LEVERAGE = 500

TIMEFRAMES = {
    "M15": (mt5.TIMEFRAME_M15, 20000),   # ~208 days of M15
    "H1":  (mt5.TIMEFRAME_H1,  10000),   # ~416 days of H1
    "H4":  (mt5.TIMEFRAME_H4,  5000),    # ~833 days of H4
}


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
    def __init__(self, sym, direction, entry, sl, atr_val, lot, cont_sz, entry_time, tf_label):
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
        self.tf_label = tf_label
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


def calc_lot_size(balance, entry_price, sl_price, contract_sz, risk_pct=0.05):
    sl_distance = abs(entry_price - sl_price)
    if sl_distance <= 0:
        return 0.01
    risk_amount = balance * risk_pct
    lot = risk_amount / (sl_distance * contract_sz)
    max_lot = (balance * LEVERAGE) / (entry_price * contract_sz)
    lot = min(lot, max_lot)
    lot = max(0.01, round(lot, 2))
    return lot


def run_simulation(tf_label, symbol_data_dict):
    """Run a single-TF simulation and return results."""
    # Collect all signals chronologically
    all_signals = []
    for sym, df in symbol_data_dict.items():
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
                    "tf": tf_label,
                })
    all_signals.sort(key=lambda x: x["time"])

    if not all_signals:
        return None

    # Simulate
    balance = STARTING_BALANCE
    peak_balance = STARTING_BALANCE
    max_dd = 0
    trades_done = []
    active = {}
    weekly_bals = []
    cur_week = None

    for sig in all_signals:
        sym = sig["sym"]
        close = sig["close"]
        atr = sig["atr"]
        cont_sz = sig["cont_sz"]
        direction = sig["direction"]
        t = sig["time"]
        idx = sig["idx"]
        df = symbol_data_dict[sym]

        if np.isnan(atr) or atr <= 0:
            continue

        week_key = t.isocalendar()[:2]
        if cur_week is None:
            cur_week = week_key
        if week_key != cur_week:
            weekly_bals.append((cur_week, balance))
            cur_week = week_key

        # Process bars for active position
        if sym in active:
            pos = active[sym]
            if not pos.closed:
                for j in range(pos._last_checked + 1, idx + 1):
                    row_j = df.iloc[j]
                    pos.check_bar(float(row_j["high"]), float(row_j["low"]),
                                  float(row_j["close"]), int(row_j.get("pos", 0)))
                    if pos.closed:
                        pos.exit_time = row_j["time"]
                        balance += pos.total_pnl
                        trades_done.append(pos)
                        del active[sym]
                        break
                else:
                    pos._last_checked = idx

        # Force close on flip
        if sym in active and not active[sym].closed:
            pos = active[sym]
            pos.force_close(close)
            pos.exit_time = t
            balance += pos.total_pnl
            trades_done.append(pos)
            del active[sym]

        if balance <= 0:
            break

        smart_sl = compute_smart_sl(df, idx, direction, atr)
        lot = calc_lot_size(balance, close, smart_sl, cont_sz, risk_pct=0.05)

        pos = V4Position(sym=sym, direction=direction, entry=close, sl=smart_sl,
                         atr_val=atr, lot=lot, cont_sz=cont_sz, entry_time=t, tf_label=tf_label)
        pos._last_checked = idx
        active[sym] = pos

        if balance > peak_balance:
            peak_balance = balance
        dd = peak_balance - balance
        if dd > max_dd:
            max_dd = dd

    # Close remaining
    for sym, pos in active.items():
        if not pos.closed:
            df = symbol_data_dict[sym]
            pos.force_close(float(df.iloc[-1]["close"]))
            pos.exit_time = df.iloc[-1]["time"]
            balance += pos.total_pnl
            trades_done.append(pos)

    if cur_week:
        weekly_bals.append((cur_week, balance))

    if not trades_done:
        return None

    pnls = [t.total_pnl for t in trades_done]
    wins = [t for t in trades_done if t.total_pnl > 0]
    losses = [t for t in trades_done if t.total_pnl <= 0]
    wr = len(wins) / len(trades_done) * 100

    first_t = trades_done[0].entry_time
    last_t = trades_done[-1].exit_time or trades_done[-1].entry_time
    days = (last_t - first_t).days
    weeks = max(days / 7, 1)

    # Milestones
    milestones_hit = {}
    targets = [500, 1000, 2000, 5000, 10000, 50000, 100000]
    running = STARTING_BALANCE
    for i, tr in enumerate(trades_done):
        running += tr.total_pnl
        while targets and running >= targets[0]:
            m = targets.pop(0)
            d = (tr.exit_time - first_t).days if tr.exit_time else 0
            milestones_hit[m] = {"trade": i+1, "days": d, "weeks": d/7}

    return {
        "tf": tf_label,
        "trades": len(trades_done),
        "signals": len(all_signals),
        "wr": wr,
        "balance": balance,
        "pnl": balance - STARTING_BALANCE,
        "max_dd": max_dd,
        "days": days,
        "weeks": weeks,
        "trades_per_week": len(trades_done) / weeks,
        "pnl_per_week": (balance - STARTING_BALANCE) / weeks,
        "weekly_bals": weekly_bals,
        "milestones": milestones_hit,
        "wins": len(wins),
        "losses": len(losses),
        "first_time": first_t,
        "last_time": last_t,
    }


def main():
    print("=" * 110)
    print("  MULTI-TIMEFRAME $200 ACCOUNT SIMULATION")
    print("  Testing: H4, H1, M15 separately -- which is fastest?")
    print("  V4 Optimized: TP1=0.8x BE=0.5x Stall=6 Trail=0.8x | 5% risk | 1:500")
    print("=" * 110)

    if not mt5.initialize():
        print("ERROR: Could not initialize MT5")
        return

    # Load data for all timeframes
    all_tf_data = {}
    for tf_name, (tf_const, fetch_bars) in TIMEFRAMES.items():
        print(f"\nLoading {tf_name} data...")
        tf_data = {}
        for sym in SYMBOLS:
            resolved = resolve_symbol(sym)
            if resolved is None:
                continue
            rates = mt5.copy_rates_from_pos(resolved, tf_const, 0, fetch_bars)
            if rates is None or len(rates) < 100:
                continue
            df = pd.DataFrame(rates)
            df["time"] = pd.to_datetime(df["time"], unit="s")
            df.rename(columns={"tick_volume": "volume"}, inplace=True)
            df_zp = compute_zeropoint_state(df)
            if df_zp is not None and len(df_zp) >= WARMUP_BARS:
                tf_data[sym] = df_zp
                days_span = (df_zp["time"].iloc[-1] - df_zp["time"].iloc[WARMUP_BARS]).days
                print(f"  {sym}: {len(df_zp)} bars ({days_span} days)")
        all_tf_data[tf_name] = tf_data

    mt5.shutdown()

    # Run simulations for each TF
    results = {}
    for tf_name in TIMEFRAMES:
        if tf_name not in all_tf_data or not all_tf_data[tf_name]:
            continue
        print(f"\n{'=' * 110}")
        print(f"  Simulating {tf_name}...")
        t0 = time.time()
        res = run_simulation(tf_name, all_tf_data[tf_name])
        elapsed = time.time() - t0
        if res:
            results[tf_name] = res
            print(f"  Done in {elapsed:.1f}s")

    # ═══════════════════════════════════════════════════════════════
    # COMPARISON TABLE
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'=' * 110}")
    print(f"  TIMEFRAME COMPARISON -- $200 start, 5% risk, 1:500 leverage")
    print(f"{'=' * 110}")

    header = (f"  {'TF':>4} | {'Period':>20} | {'Trades':>7} | {'Tr/Wk':>6} | "
              f"{'WR%':>6} | {'Final Bal':>14} | {'$/Week':>10} | {'MaxDD':>10}")
    print(header)
    print("  " + "-" * 105)

    for tf_name in ["M15", "H1", "H4"]:
        if tf_name not in results:
            continue
        r = results[tf_name]
        period = f"{r['first_time'].strftime('%Y-%m-%d')} - {r['last_time'].strftime('%Y-%m-%d')}"
        print(f"  {tf_name:>4} | {period:>20} | {r['trades']:>7} | {r['trades_per_week']:>5.1f} | "
              f"{r['wr']:>5.1f}% | ${r['balance']:>12,.2f} | ${r['pnl_per_week']:>8,.0f} | ${r['max_dd']:>8,.0f}")

    # ═══════════════════════════════════════════════════════════════
    # MILESTONE COMPARISON
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'=' * 110}")
    print(f"  MILESTONE RACE -- How fast does each TF reach key balances?")
    print(f"{'=' * 110}")

    targets = [500, 1000, 2000, 5000, 10000, 50000, 100000]
    header2 = f"  {'Target':>10}"
    for tf_name in ["M15", "H1", "H4"]:
        if tf_name in results:
            header2 += f" | {tf_name + ' (weeks)':>14} {tf_name + ' (trade#)':>14}"
    print(header2)
    print("  " + "-" * 90)

    for target in targets:
        line = f"  ${target:>9,}"
        for tf_name in ["M15", "H1", "H4"]:
            if tf_name not in results:
                continue
            ms = results[tf_name]["milestones"]
            if target in ms:
                line += f" | {ms[target]['weeks']:>12.1f}w #{ms[target]['trade']:>12}"
            else:
                line += f" | {'--':>12}  {'--':>12}"
        print(line)

    # ═══════════════════════════════════════════════════════════════
    # WEEKLY GROWTH FOR EACH TF (first 12 weeks)
    # ═══════════════════════════════════════════════════════════════
    for tf_name in ["M15", "H1", "H4"]:
        if tf_name not in results:
            continue
        r = results[tf_name]
        wb = r["weekly_bals"]
        print(f"\n{'=' * 80}")
        print(f"  {tf_name} -- First 12 Weeks")
        print(f"{'=' * 80}")
        print(f"  {'Week':>8} | {'Balance':>12} | {'Change':>10}")
        print(f"  " + "-" * 40)
        prev = STARTING_BALANCE
        for i, (wk, bal) in enumerate(wb[:12]):
            yr, wn = wk
            change = bal - prev
            print(f"  {yr}-W{wn:02d} | ${bal:>10,.2f} | ${change:>+8,.2f}")
            prev = bal

    # ═══════════════════════════════════════════════════════════════
    # WINNER
    # ═══════════════════════════════════════════════════════════════
    if results:
        fastest_to_10k = None
        fastest_tf = None
        for tf_name, r in results.items():
            ms = r["milestones"]
            if 10000 in ms:
                wks = ms[10000]["weeks"]
                if fastest_to_10k is None or wks < fastest_to_10k:
                    fastest_to_10k = wks
                    fastest_tf = tf_name

        print(f"\n{'=' * 110}")
        print(f"  VERDICT")
        print(f"{'=' * 110}")
        if fastest_tf:
            r = results[fastest_tf]
            print(f"\n  FASTEST TO $10K: {fastest_tf} ({fastest_to_10k:.1f} weeks)")
            print(f"  Trades/week: {r['trades_per_week']:.1f}")
            print(f"  Win rate: {r['wr']:.1f}%")
            print(f"  Final balance: ${r['balance']:,.2f}")
        else:
            # Show best overall
            best_tf = max(results, key=lambda x: results[x]["balance"])
            r = results[best_tf]
            print(f"\n  MOST PROFITABLE: {best_tf}")
            print(f"  Final balance: ${r['balance']:,.2f}")
            print(f"  Trades/week: {r['trades_per_week']:.1f}")

        for tf_name in ["M15", "H1", "H4"]:
            if tf_name in results:
                r = results[tf_name]
                print(f"\n  {tf_name}: {r['trades']} trades over {r['weeks']:.0f} weeks "
                      f"({r['trades_per_week']:.1f}/wk) | {r['wr']:.1f}% WR | "
                      f"${r['balance']:,.2f} final")

    print(f"\n{'=' * 110}")


if __name__ == "__main__":
    main()
