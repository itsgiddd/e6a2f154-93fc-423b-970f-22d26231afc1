#!/usr/bin/env python3
"""
V4 Parameter Optimizer — find settings that push PF toward 2-3x
================================================================
FOCUSED grid: ~300 configs (was 5184). Tests the 4 biggest levers:
  1. SL ATR cap (tighter = smaller losses per trade)
  2. TP1 multiplier (lower = more trades reach TP1)
  3. R:R minimum filter (skip low-quality entries)
  4. Stall bars (faster stall exit)
  5. Trail distance
  6. BE trigger sensitivity

Phase 1: Coarse sweep to find promising regions
Phase 2: Fine-tune top 5 configs with ±0.1 increments
"""

import sys, os, time
import numpy as np
import pandas as pd
from collections import defaultdict
from itertools import product

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "app"))

import MetaTrader5 as mt5
from app.zeropoint_signal import (
    compute_zeropoint_state,
    SL_BUFFER_PCT, SWING_LOOKBACK,
)

SYMBOLS = ["AUDUSD", "EURJPY", "EURUSD", "GBPJPY", "GBPUSD", "NZDUSD", "USDCAD", "USDJPY"]
FETCH_BARS = 5000
WARMUP_BARS = 50
LOT_SIZE = 0.10


def pip_value(symbol):
    sym = symbol.upper()
    if "JPY" in sym: return 0.01
    elif "BTC" in sym or "XAU" in sym: return 1.0
    return 0.0001


def contract_size(symbol):
    return 1.0 if "BTC" in symbol.upper() else 100_000.0


def resolve_symbol(ticker):
    for c in [ticker, ticker + ".raw", ticker + "m", ticker + ".a", ticker + ".e"]:
        info = mt5.symbol_info(c)
        if info is not None:
            mt5.symbol_select(c, True)
            return c
    return None


def compute_smart_sl(df, bar_idx, direction, atr_val, sl_atr_min_mult):
    lookback_start = max(0, bar_idx - SWING_LOOKBACK + 1)
    cur_close = float(df["close"].iloc[bar_idx])
    if direction == "BUY":
        recent_low = float(df["low"].iloc[lookback_start:bar_idx + 1].min())
        buffer = recent_low * SL_BUFFER_PCT
        structural_sl = recent_low - buffer
        atr_min_sl = cur_close - atr_val * sl_atr_min_mult
        return min(structural_sl, atr_min_sl)
    else:
        recent_high = float(df["high"].iloc[lookback_start:bar_idx + 1].max())
        buffer = recent_high * SL_BUFFER_PCT
        structural_sl = recent_high + buffer
        atr_max_sl = cur_close + atr_val * sl_atr_min_mult
        return max(structural_sl, atr_max_sl)


class TestPosition:
    """Parameterized V4 position for optimization."""

    def __init__(self, sym, direction, entry, sl, atr_val, lot, cont_sz, params):
        self.sym = sym
        self.direction = direction
        self.entry = entry
        self.sl = sl
        self.atr_val = atr_val
        self.total_lot = lot
        self.remaining_lot = lot
        self.contract_size = cont_sz
        self.params = params
        sign = 1 if direction == "BUY" else -1
        self.tp1 = entry + sign * params["tp1_mult"] * atr_val
        self.tp2 = entry + sign * params["tp2_mult"] * atr_val
        self.tp3 = entry + sign * params["tp3_mult"] * atr_val
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
        p = self.params

        # Track MFE
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

        # Post-TP1 trailing
        if self.tp1_hit:
            trail_dist = p["trail_dist"] * atr
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
            if self.max_profit_reached >= p["be_trigger"] * atr:
                be_buf = p["be_buffer"] * atr
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

        # Stall
        if not self.tp1_hit and not self.stall_be_activated:
            if self.bars_in_trade >= p["stall_bars"]:
                be_buf = p["be_buffer"] * atr
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
            micro_price = (self.entry + p["micro_mult"] * atr) if is_buy else (self.entry - p["micro_mult"] * atr)
            micro_triggered = (is_buy and high >= micro_price) or (not is_buy and low <= micro_price)
            if micro_triggered:
                self.micro_tp_hit = True
                micro_lot = round(self.total_lot * p["micro_pct"], 2)
                micro_lot = max(0.01, min(micro_lot, self.remaining_lot))
                pnl = self.pnl_for_price(micro_price, micro_lot)
                self.partials.append(pnl)
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

        # ZP flip
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


def run_backtest(symbol_data, params):
    """Run a single backtest with given params, return stats dict."""
    trades = []
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
            pos_val = int(row.get("pos", 0))
            buy_sig = bool(row.get("buy_signal", False))
            sell_sig = bool(row.get("sell_signal", False))
            if np.isnan(atr_val) or atr_val <= 0:
                continue

            if pos_obj is not None and not pos_obj.closed:
                pos_obj.check_bar(high, low, close, pos_val)
                if pos_obj.closed:
                    trades.append(pos_obj)
                    pos_obj = None

            if buy_sig or sell_sig:
                direction = "BUY" if buy_sig else "SELL"
                smart_sl = compute_smart_sl(df, i, direction, atr_val, params["sl_atr_min"])

                # R:R filter
                if direction == "BUY":
                    sl_dist = close - smart_sl
                else:
                    sl_dist = smart_sl - close
                tp1_dist = params["tp1_mult"] * atr_val
                rr = tp1_dist / sl_dist if sl_dist > 0 else 0
                if rr < params.get("min_rr", 0):
                    continue

                if pos_obj is not None and not pos_obj.closed:
                    pos_obj.force_close(close)
                    trades.append(pos_obj)

                pos_obj = TestPosition(
                    sym=sym, direction=direction, entry=close, sl=smart_sl,
                    atr_val=atr_val, lot=LOT_SIZE, cont_sz=cont_sz, params=params,
                )

        if pos_obj is not None and not pos_obj.closed:
            pos_obj.force_close(float(df.iloc[-1]["close"]))
            trades.append(pos_obj)

    if not trades:
        return None

    pnls = [t.total_pnl for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    pf = gross_profit / gross_loss if gross_loss > 0 else 999
    wr = len(wins) / len(pnls) * 100
    total_pnl = sum(pnls)
    expectancy = total_pnl / len(pnls)

    # Max drawdown
    peak = 200.0
    max_dd = 0
    running = 200.0
    for p in pnls:
        running += p
        if running > peak:
            peak = running
        dd = peak - running
        if dd > max_dd:
            max_dd = dd

    # Exit type distribution
    exit_types = defaultdict(int)
    exit_pnl = defaultdict(float)
    for t in trades:
        if t.exit_type:
            exit_types[t.exit_type] += 1
            exit_pnl[t.exit_type] += t.total_pnl

    # Count pure SL trades (the biggest profit killers)
    sl_trades = sum(1 for t in trades if t.exit_type == 'SL')
    sl_total_loss = sum(t.total_pnl for t in trades if t.exit_type == 'SL')
    zp_flip_count = sum(1 for t in trades if t.exit_type == 'ZP_FLIP')
    zp_flip_loss = sum(t.total_pnl for t in trades if t.exit_type == 'ZP_FLIP')

    return {
        "trades": len(trades), "wr": wr, "pf": pf, "pnl": total_pnl,
        "expectancy": expectancy, "wins": len(wins), "losses": len(losses),
        "avg_win": np.mean(wins) if wins else 0,
        "avg_loss": np.mean(losses) if losses else 0,
        "max_dd": max_dd,
        "sl_count": sl_trades, "sl_loss": sl_total_loss,
        "zp_flip_count": zp_flip_count, "zp_flip_loss": zp_flip_loss,
        "exit_types": dict(exit_types),
    }


def print_config_row(rank, cfg, stats, show_exits=False):
    line = (f"{rank:>3} | {cfg['tp1_mult']:>4.1f} {cfg['tp2_mult']:>4.1f} "
            f"{cfg['be_trigger']:>4.1f} {cfg['stall_bars']:>5} "
            f"{cfg['trail_dist']:>5.1f} {cfg['micro_pct']:>5.0%} "
            f"{cfg['sl_atr_min']:>6.1f} {cfg['min_rr']:>5.1f} | "
            f"{stats['trades']:>6} {stats['wr']:>5.1f}% {stats['pf']:>5.2f} "
            f"${stats['pnl']:>+10,.0f} ${stats['expectancy']:>+7.0f} "
            f"DD ${stats['max_dd']:>6,.0f}")
    if show_exits:
        line += f" | SL:{stats['sl_count']} ZF:{stats['zp_flip_count']}"
    print(line)


def main():
    print("=" * 120)
    print("  V4 PARAMETER OPTIMIZER — Targeting PF 2-3")
    print("  FOCUSED GRID: testing ~300 configs (key levers only)")
    print("=" * 120)

    if not mt5.initialize():
        print("ERROR: Could not initialize MT5")
        return

    symbol_data = {}
    print("\nLoading H4 data...")
    t0 = time.time()
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
    print(f"  Data loaded in {time.time() - t0:.1f}s")

    if not symbol_data:
        print("No data!")
        return

    # ═══════════════════════════════════════════════════════════════════
    # PHASE 1: Coarse grid (~300 configs)
    # ═══════════════════════════════════════════════════════════════════
    # PRIMARY levers (exhaustive): 4 * 4 * 3 = 48 combos
    tp1_values = [0.8, 1.0, 1.2, 1.5]          # 4 values
    sl_atr_min_values = [1.5, 2.0, 2.5, 3.0]   # 4 values (KEY: tighter SL cap)
    min_rr_values = [0.0, 0.2, 0.4]            # 3 values

    # SECONDARY levers (sampled): 3 combos of (be, stall, trail, tp2)
    secondary_combos = [
        # (be_trigger, stall_bars, trail_dist, tp2_mult)
        (0.5, 6, 0.8, 2.0),    # aggressive: fast BE, fast stall, tight trail
        (0.8, 8, 1.2, 2.5),    # moderate
        (1.0, 12, 1.5, 3.0),   # conservative: slow BE, slow stall, wide trail
    ]

    FIXED_MICRO_PCT = 0.15
    FIXED_BE_BUFFER = 0.15

    configs = []
    # 48 primary combos × 3 secondary = 144 configs total
    for tp1 in tp1_values:
        for sl_min in sl_atr_min_values:
            for min_rr in min_rr_values:
                for be_trigger, stall, trail, tp2 in secondary_combos:
                    configs.append({
                        "tp1_mult": tp1,
                        "tp2_mult": tp2,
                        "tp3_mult": 5.0,
                        "be_trigger": be_trigger,
                        "be_buffer": FIXED_BE_BUFFER,
                        "trail_dist": trail,
                        "stall_bars": stall,
                        "micro_mult": min(tp1, 1.0),
                        "micro_pct": FIXED_MICRO_PCT,
                        "sl_atr_min": sl_min,
                        "min_rr": min_rr,
                    })

    total_configs = len(configs)
    print(f"\n  PHASE 1: Testing {total_configs} configurations...")
    t0 = time.time()

    results = []
    for i, cfg in enumerate(configs):
        if i % 100 == 0 and i > 0:
            elapsed = time.time() - t0
            rate = i / elapsed
            remaining = (total_configs - i) / rate
            print(f"  [{i}/{total_configs}] ({elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining)")
        stats = run_backtest(symbol_data, cfg)
        if stats and stats["trades"] >= 100:  # at least 100 trades
            results.append((cfg, stats))

    elapsed = time.time() - t0
    print(f"\n  Phase 1 complete: {len(results)} valid configs in {elapsed:.1f}s")

    # Sort by PF
    results.sort(key=lambda x: x[1]["pf"], reverse=True)

    # ═══════════════════════════════════════════════════════════════════
    # RESULTS: Top by PF
    # ═══════════════════════════════════════════════════════════════════
    header = (f"{'#':>3} | {'TP1':>4} {'TP2':>4} {'BE':>4} {'Stall':>5} "
              f"{'Trail':>5} {'Micro%':>6} {'SL_Min':>6} {'MinRR':>5} | "
              f"{'Trades':>6} {'WR%':>6} {'PF':>6} {'PnL':>11} {'Exp':>8} {'MaxDD':>10} | SL  ZF")

    print(f"\n{'=' * 130}")
    print(f"  TOP 30 BY PROFIT FACTOR")
    print(f"{'=' * 130}")
    print(header)
    print("-" * 130)
    for rank, (cfg, stats) in enumerate(results[:30], 1):
        print_config_row(rank, cfg, stats, show_exits=True)

    # ═══════════════════════════════════════════════════════════════════
    # RESULTS: PF >= 2.0
    # ═══════════════════════════════════════════════════════════════════
    pf2_plus = [(c, s) for c, s in results if s["pf"] >= 2.0 and s["trades"] >= 100]
    pf2_plus.sort(key=lambda x: x[1]["pnl"], reverse=True)

    print(f"\n{'=' * 130}")
    print(f"  PF >= 2.0 (sorted by PnL) — {len(pf2_plus)} configs found")
    print(f"{'=' * 130}")
    if pf2_plus:
        print(header)
        print("-" * 130)
        for rank, (cfg, stats) in enumerate(pf2_plus[:20], 1):
            print_config_row(rank, cfg, stats, show_exits=True)
    else:
        print("  None found — showing PF >= 1.5:")
        pf15_plus = [(c, s) for c, s in results if s["pf"] >= 1.5 and s["trades"] >= 100]
        pf15_plus.sort(key=lambda x: x[1]["pnl"], reverse=True)
        if pf15_plus:
            print(header)
            print("-" * 130)
            for rank, (cfg, stats) in enumerate(pf15_plus[:20], 1):
                print_config_row(rank, cfg, stats, show_exits=True)

    # ═══════════════════════════════════════════════════════════════════
    # RESULTS: Best balanced (WR >= 75% AND PF >= 1.5 AND PnL > 0)
    # ═══════════════════════════════════════════════════════════════════
    balanced = [(c, s) for c, s in results
                if s["wr"] >= 75 and s["pf"] >= 1.5 and s["trades"] >= 200]
    balanced.sort(key=lambda x: x[1]["pf"], reverse=True)

    print(f"\n{'=' * 130}")
    print(f"  BEST BALANCED (WR>=75% AND PF>=1.5 AND 200+ trades) — {len(balanced)} configs")
    print(f"{'=' * 130}")
    if balanced:
        print(header)
        print("-" * 130)
        for rank, (cfg, stats) in enumerate(balanced[:20], 1):
            print_config_row(rank, cfg, stats, show_exits=True)

    # ═══════════════════════════════════════════════════════════════════
    # RESULTS: Best expectancy ($ per trade)
    # ═══════════════════════════════════════════════════════════════════
    by_exp = sorted(results, key=lambda x: x[1]["expectancy"], reverse=True)

    print(f"\n{'=' * 130}")
    print(f"  TOP 15 BY EXPECTANCY ($ per trade)")
    print(f"{'=' * 130}")
    print(header)
    print("-" * 130)
    for rank, (cfg, stats) in enumerate(by_exp[:15], 1):
        print_config_row(rank, cfg, stats, show_exits=True)

    # ═══════════════════════════════════════════════════════════════════
    # ANALYSIS: Impact of each parameter
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 130}")
    print(f"  PARAMETER IMPACT ANALYSIS (average PF by parameter value)")
    print(f"{'=' * 130}")

    for param_name, param_values in [
        ("tp1_mult", tp1_values), ("sl_atr_min", sl_atr_min_values),
        ("min_rr", min_rr_values),
        ("be_trigger", [0.5, 0.8, 1.0]),
        ("stall_bars", [6, 8, 12]),
        ("trail_dist", [0.8, 1.2, 1.5]),
        ("tp2_mult", [2.0, 2.5, 3.0]),
    ]:
        print(f"\n  {param_name}:")
        for val in param_values:
            matching = [(c, s) for c, s in results if c[param_name] == val]
            if matching:
                avg_pf = np.mean([s["pf"] for _, s in matching])
                avg_wr = np.mean([s["wr"] for _, s in matching])
                avg_pnl = np.mean([s["pnl"] for _, s in matching])
                avg_trades = np.mean([s["trades"] for _, s in matching])
                best_pf = max(s["pf"] for _, s in matching)
                print(f"    {val:>6}: avg PF={avg_pf:.2f} WR={avg_wr:.1f}% "
                      f"PnL=${avg_pnl:+,.0f} trades={avg_trades:.0f} "
                      f"best_PF={best_pf:.2f} (n={len(matching)})")

    # ═══════════════════════════════════════════════════════════════════
    # PHASE 2: Fine-tune top 5 configs
    # ═══════════════════════════════════════════════════════════════════
    if len(results) >= 5:
        print(f"\n{'=' * 130}")
        print(f"  PHASE 2: Fine-tuning top 5 configs")
        print(f"{'=' * 130}")

        fine_results = []
        for base_cfg, base_stats in results[:5]:
            print(f"\n  Fine-tuning: TP1={base_cfg['tp1_mult']} SL_min={base_cfg['sl_atr_min']} "
                  f"RR={base_cfg['min_rr']} (PF={base_stats['pf']:.2f})")

            # Create fine grid around this config
            for tp1_delta in [-0.1, 0, 0.1, 0.2]:
                for sl_delta in [-0.25, 0, 0.25]:
                    for rr_delta in [-0.1, 0, 0.1]:
                        for be_delta in [-0.1, 0, 0.1]:
                            fine_cfg = dict(base_cfg)
                            fine_cfg["tp1_mult"] = max(0.5, base_cfg["tp1_mult"] + tp1_delta)
                            fine_cfg["sl_atr_min"] = max(1.0, base_cfg["sl_atr_min"] + sl_delta)
                            fine_cfg["min_rr"] = max(0.0, base_cfg["min_rr"] + rr_delta)
                            fine_cfg["be_trigger"] = max(0.3, base_cfg["be_trigger"] + be_delta)
                            fine_cfg["micro_mult"] = min(fine_cfg["tp1_mult"], 1.0)

                            stats = run_backtest(symbol_data, fine_cfg)
                            if stats and stats["trades"] >= 100:
                                fine_results.append((fine_cfg, stats))

        if fine_results:
            fine_results.sort(key=lambda x: x[1]["pf"], reverse=True)
            print(f"\n  Fine-tuning found {len(fine_results)} valid configs")
            print(f"\n  TOP 15 FINE-TUNED:")
            print(header)
            print("-" * 130)
            for rank, (cfg, stats) in enumerate(fine_results[:15], 1):
                print_config_row(rank, cfg, stats, show_exits=True)

            # Overall best
            all_results = results + fine_results
            all_results.sort(key=lambda x: x[1]["pf"], reverse=True)
        else:
            all_results = results
    else:
        all_results = results

    # ═══════════════════════════════════════════════════════════════════
    # FINAL RECOMMENDATION
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 130}")
    print(f"  FINAL RECOMMENDATION")
    print(f"{'=' * 130}")

    # Pick best with reasonable trade count
    candidates = [(c, s) for c, s in all_results if s["trades"] >= 200]
    if not candidates:
        candidates = [(c, s) for c, s in all_results if s["trades"] >= 100]

    if candidates:
        # Best by PF
        best_pf = max(candidates, key=lambda x: x[1]["pf"])
        # Best by PnL
        best_pnl = max(candidates, key=lambda x: x[1]["pnl"])
        # Best balanced (PF * WR * sqrt(trades))
        best_balanced = max(candidates,
                          key=lambda x: x[1]["pf"] * (x[1]["wr"]/100) * np.sqrt(x[1]["trades"]))

        for label, (cfg, stats) in [
            ("HIGHEST PF", best_pf),
            ("HIGHEST PNL", best_pnl),
            ("BEST BALANCED", best_balanced),
        ]:
            print(f"\n  {label}:")
            print(f"    TP1={cfg['tp1_mult']:.1f}x  TP2={cfg['tp2_mult']:.1f}x  TP3={cfg['tp3_mult']:.1f}x")
            print(f"    BE_trigger={cfg['be_trigger']:.1f}x  BE_buffer={cfg['be_buffer']:.2f}x")
            print(f"    Trail={cfg['trail_dist']:.1f}x  Stall={cfg['stall_bars']} bars")
            print(f"    Micro={cfg['micro_pct']:.0%} @ {cfg['micro_mult']:.1f}x")
            print(f"    SL_ATR_min={cfg['sl_atr_min']:.1f}x  Min_RR={cfg['min_rr']:.1f}")
            print(f"    → {stats['trades']} trades | {stats['wr']:.1f}% WR | PF {stats['pf']:.2f} "
                  f"| ${stats['pnl']:+,.0f} | Exp ${stats['expectancy']:+,.0f} "
                  f"| DD ${stats['max_dd']:,.0f}")
            print(f"    → SL losses: {stats['sl_count']} | ZP_FLIP: {stats['zp_flip_count']}")

    print(f"\n  Total configs tested: {len(all_results)}")
    print(f"{'=' * 130}")


if __name__ == "__main__":
    main()
