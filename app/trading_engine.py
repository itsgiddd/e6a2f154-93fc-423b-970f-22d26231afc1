#!/usr/bin/env python3
"""
ACI Trading Engine
==================

Professional AI trading engine that integrates neural network predictions
with MT5 trading operations for automated forex trading.

Features:
- Neural network signal generation
- Automated trade execution
- Risk management
- Position monitoring
- Performance tracking
- Real-time trading loop
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import logging
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import re

# Import app modules
from .mt5_connector import MT5Connector
from .model_manager import NeuralModelManager
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from enhanced_tail_risk_protection import TailRiskProtector
from push_structure_analyzer import (
    PushFeatureExtractor,
    SymbolPushProfile,
    infer_direction_from_closes,
)
from pattern_recognition import PatternRecognizer, Pattern, FormingPattern, FORMING_PATTERN_FEATURE_COUNT
from .zeropoint_signal import (
    ZeroPointEngine,
    ZeroPointSignal,
    compute_zeropoint_state,
    extract_zeropoint_bar_features,
    ZEROPOINT_FEATURES_PER_TF,
)

class TradingSignal:
    """Trading signal data structure"""
    
    def __init__(self, symbol: str, action: str, confidence: float, 
                 entry_price: float, stop_loss: float, take_profit: float,
                 position_size: float, reason: str):
        self.symbol = symbol
        self.action = action  # 'BUY', 'SELL', 'HOLD'
        self.confidence = confidence
        self.entry_price = entry_price
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.position_size = position_size
        self.reason = reason
        self.timestamp = datetime.now()
        self.executed = False
        self.order_ticket = None

class Position:
    """Position tracking data structure"""

    def __init__(self, ticket: int, symbol: str, action: str,
                 entry_price: float, stop_loss: float, take_profit: float,
                 position_size: float):
        self.ticket = ticket
        self.symbol = symbol
        self.action = action
        self.entry_price = entry_price
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.position_size = position_size
        self.open_time = datetime.now()
        self.current_price = entry_price
        self.unrealized_pnl = 0.0
        self.status = 'OPEN'  # OPEN, CLOSED, PARTIAL
        self.close_recorded = False
        # Agentic learning metadata (set by _process_signal)
        self.model_confidence: float = 0.0
        self.model_action: str = ''
        self.model_probabilities: str = '{}'
        self.symbol_threshold: float = 0.0
        self.action_mode: str = 'normal'
        self.model_version: str = ''

class TradingEngine:
    """Professional neural trading engine"""
    
    def __init__(self, mt5_connector: MT5Connector, model_manager: NeuralModelManager,
                 risk_per_trade: float = 0.30, confidence_threshold: float = 0.65,
                 trading_pairs: List[str] = None, max_concurrent_positions: int = 8):
        
        self.logger = logging.getLogger(__name__)
        
        # Core components
        self.mt5_connector = mt5_connector
        self.model_manager = model_manager
        
        # Enhanced tail risk protection
        self.tail_risk_protector = TailRiskProtector()
        
        # Advanced performance tracking
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.dirname(__file__)))
        from advanced_performance_tracking import AdvancedPerformanceTracker
        self.performance_tracker = AdvancedPerformanceTracker()
        
        # Trading parameters
        self.risk_per_trade = risk_per_trade  # 30% aggressive for 97.5% WR — double every ~18 days
        self.confidence_threshold = confidence_threshold  # 65% default
        self.trading_pairs = trading_pairs or ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'BTCUSD']
        self.max_concurrent_positions = max_concurrent_positions
        # Conservative startup mode: do not force entries unless explicitly enabled.
        # Immediate execution profile: prioritize taking valid model entries quickly.
        self.immediate_trade_mode = True
        self.profitability_first_mode = False
        self._startup_trade_done: set[str] = set()
        self.model_min_trade_score = 0.20  # EMERGENCY FIX: Was 0.33, too restrictive for profitability
        self.model_min_directional_gap = 0.005
        self.model_pattern_conflict_block = False
        self.mtf_alignment_enabled = False
        self.historical_signal_horizon = 16
        self.historical_min_samples = 30
        self.minimum_symbol_quality_winrate = 0.40  # EMERGENCY FIX: Was 0.50, too restrictive
        self.minimum_symbol_quality_samples = 20  # EMERGENCY FIX: Was 40, too restrictive
        self.minimum_symbol_profit_factor = 0.90  # EMERGENCY FIX: Was 1.05, blocking trades
        self.minimum_symbol_expectancy = 0.00001
        self.minimum_symbol_profitability_samples = 60
        self.live_min_trade_rate = 0.015
        # Symbols explicitly excluded from live trading (weak model performance).
        # NOTE: Cleared for ZeroPoint model — EURUSD is now PF 6.84.
        self.excluded_symbols: set = set()
        # Correlation groups: symbols in the same group move together.
        # Sign indicates direction alignment (+1 = long symbol = long group, -1 = inverted).
        self.correlation_groups = {
            'USD_COMMODITY': {'AUDUSD': +1, 'NZDUSD': +1, 'USDCAD': -1},
            'USD_MAJOR': {'EURUSD': +1, 'GBPUSD': +1, 'USDCAD': -1, 'USDJPY': -1},
            'JPY_CROSSES': {'EURJPY': +1, 'GBPJPY': +1, 'USDJPY': +1},
        }
        self.max_correlated_same_direction = 2
        self.correlated_size_reduction = 0.60
        # Global lot size cap by account balance (prevents oversizing on tight SLs).
        self.global_max_lot_table = [
            (500, 0.10), (1000, 0.20), (3000, 0.50),
            (5000, 1.00), (10000, 2.00), (50000, 5.00),
            (float('inf'), 10.00),
        ]
        # Progressive daily loss tiers (size reduction as losses accumulate).
        self._daily_loss_tiers = [
            (0.05, 0.50),   # 5% daily loss → 50% position size
            (0.08, 0.25),   # 8% → 25%
            (0.12, 0.10),   # 12% → 10%
            (0.15, 0.00),   # 15% → full stop for rest of day
        ]
        self._daily_loss_size_factor = 1.0
        self._symbol_live_profile: Dict[str, Dict[str, float]] = {}
        self._symbol_profile_skip_log_time: Dict[str, datetime] = {}
        self.market_closed_cooldown_seconds = 300
        self._symbol_trade_block_until: Dict[str, datetime] = {}
        self._symbol_market_closed_log_time: Dict[str, datetime] = {}
        self.symbol_entry_cooldown_seconds = 300  # 5 min cooldown between entries on same symbol
        self.max_new_trades_per_hour = 20
        self._symbol_last_entry_time: Dict[str, datetime] = {}
        self._new_trade_timestamps: List[datetime] = []
        # Tail-risk and intraday protection controls (left-tail avoidance).
        self.tail_risk_control_enabled = False
        self.tail_min_weekly_p10_return = 0.0
        self.tail_min_weekly_prob_positive = 0.60
        self.max_daily_loss_pct = 0.15
        self.max_intraday_drawdown_pct = 0.20
        self.loss_pause_until_next_day = False
        self.loss_streak_limit = 5
        self.loss_streak_cooldown_seconds = 900
        self._symbol_loss_streak: Dict[str, int] = {}
        self._symbol_loss_block_until: Dict[str, datetime] = {}
        self._daily_risk_day: Optional[str] = None
        self._daily_start_equity = 0.0
        self._intraday_peak_equity = 0.0
        self._daily_loss_pause_until: Optional[datetime] = None
        
        # Agentic orchestrator (set externally after construction)
        self.orchestrator = None

        # Trading state
        self.is_running = False
        self.trading_thread = None
        self.positions: Dict[int, Position] = {}
        self.signals_history: List[TradingSignal] = []
        self.performance_metrics = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_pnl': 0.0,
            'daily_pnl': 0.0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'max_drawdown': 0.0,
            'current_drawdown': 0.0
        }
        
        # Feature engineering cache
        self.feature_cache = {}
        self.last_update = {}
        
        # Timeframes for analysis
        self.timeframes = {
            'M5': mt5.TIMEFRAME_M5,
            'M15': mt5.TIMEFRAME_M15,
            'H1': mt5.TIMEFRAME_H1,
            'H4': mt5.TIMEFRAME_H4,
            'D1': mt5.TIMEFRAME_D1,
        }
        
        # Performance tracking
        self.start_time = None
        self.last_performance_update = None
        
        # ZeroPoint PRO indicator engine (H4 ATR trailing stop strategy)
        self.zeropoint_engine = ZeroPointEngine()
        self.zeropoint_enabled = True  # Master toggle
        self.zeropoint_standalone_size = 0.65  # 65% size for standalone ZP trades
        self.zeropoint_confluence_boost = 0.10  # Confidence boost when neural + ZP agree

        # ZeroPoint Pure Mode — trades ONLY on ZP H4 flips, fixed lots, no neural/pattern
        self.zeropoint_pure_mode = False
        self.zeropoint_fixed_lot = 0.40  # Fixed lot size for pure mode
        self.zeropoint_skip_symbols: set = set()  # Symbols to skip in ZP pure mode

        # ZeroPoint Trade Monitor settings
        self.zp_max_loss_dollars = 80.0   # Close if losing more than this dollar amount

        # V4 Profit Capture — imported from zeropoint_signal.py constants
        from app.zeropoint_signal import (
            BE_TRIGGER_MULT, BE_BUFFER_MULT, PROFIT_TRAIL_DISTANCE_MULT,
            STALL_BARS, MICRO_TP_MULT, MICRO_TP_PCT,
            TP1_MULT_AGG, TP2_MULT_AGG, TP3_MULT_AGG,
        )
        self.v4_be_trigger = BE_TRIGGER_MULT       # 0.5x ATR -> move SL to BE
        self.v4_be_buffer = BE_BUFFER_MULT          # 0.15x ATR buffer above entry
        self.v4_trail_dist = PROFIT_TRAIL_DISTANCE_MULT  # 0.8x ATR behind max price
        self.v4_stall_bars = STALL_BARS             # 6 H4 bars -> move to BE
        self.v4_micro_mult = MICRO_TP_MULT          # 0.8x ATR micro-partial trigger
        self.v4_micro_pct = MICRO_TP_PCT            # 15% of lot
        self.v4_tp1_mult = TP1_MULT_AGG             # 0.8x ATR
        self.v4_tp2_mult = TP2_MULT_AGG             # 2.0x ATR
        self.v4_tp3_mult = TP3_MULT_AGG             # 5.0x ATR

        # Per-position V4 state tracker:
        # ticket -> {entry, atr, direction, open_time, open_bar_idx,
        #            be_activated, stall_activated, micro_hit, tp1_hit, tp2_hit,
        #            max_favorable_price, profit_lock_sl, remaining_lot, original_lot}
        self._zp_position_tracker: Dict[int, Dict[str, Any]] = {}

        self.logger.info("ACI Trading Engine initialized (ZeroPoint PRO enabled)")
    
    def start(self):
        """Start the trading engine"""
        if self.is_running:
            self.logger.warning("Trading engine already running")
            return
        
        # Validate prerequisites
        if not self.mt5_connector.is_connected():
            raise Exception("MT5 not connected")
        
        if not self.model_manager.is_model_loaded():
            raise Exception("Neural model not loaded")
        
        self.is_running = True
        self.start_time = datetime.now()
        self._startup_trade_done.clear()
        self._symbol_profile_skip_log_time.clear()
        self._symbol_trade_block_until.clear()
        self._symbol_market_closed_log_time.clear()
        self._symbol_last_entry_time.clear()
        self._new_trade_timestamps.clear()
        self._symbol_loss_streak.clear()
        self._symbol_loss_block_until.clear()
        self._daily_loss_pause_until = None
        self._refresh_account_risk_state(force_reset=True)
        self._rebuild_live_symbol_profile()
        if self.profitability_first_mode and self._symbol_live_profile:
            enabled_selected = [s for s in self.trading_pairs if self._is_symbol_live_enabled(s)]
            if enabled_selected:
                self.logger.info(
                    "Profitability-first mode active for selected symbols: "
                    + ", ".join(enabled_selected)
                )
            else:
                self.logger.warning(
                    "Profitability-first mode did not approve any selected symbols; "
                    "no trades will be executed until symbol quality improves"
                )
        
        # Sync existing MT5 positions into internal tracker
        # This prevents duplicate entries on restart
        self._sync_mt5_positions()

        # Start trading thread
        self.trading_thread = threading.Thread(target=self._trading_loop, daemon=True)
        self.trading_thread.start()

        self.logger.info("ACI Trading Engine started")
    
    def stop(self):
        """Stop the trading engine"""
        if not self.is_running:
            return
        
        self.is_running = False
        
        # Wait for trading thread to finish
        if self.trading_thread and self.trading_thread.is_alive():
            self.trading_thread.join(timeout=5)
        
        self.logger.info("ACI Trading Engine stopped")

    def _sync_mt5_positions(self):
        """Load existing MT5 open positions into internal tracker.

        This prevents the engine from opening duplicate trades on symbols
        that already have open positions from a previous session.
        """
        try:
            import MetaTrader5 as mt5_lib
            mt5_positions = mt5_lib.positions_get()
            if not mt5_positions:
                self.logger.info("MT5 position sync: no open positions found")
                return

            synced = 0
            for p in mt5_positions:
                ticket = p.ticket
                if ticket in self.positions:
                    continue  # already tracked

                symbol = p.symbol
                action = "BUY" if p.type == 0 else "SELL"
                pos = Position(
                    ticket=ticket,
                    symbol=symbol,
                    action=action,
                    entry_price=p.price_open,
                    stop_loss=p.sl,
                    take_profit=p.tp,
                    position_size=p.volume,
                )
                pos.current_price = p.price_current
                pos.unrealized_pnl = p.profit
                pos.status = "OPEN"
                self.positions[ticket] = pos

                # Mark symbol as recently entered to prevent immediate re-entry
                sym_key = symbol.upper()
                self._symbol_last_entry_time[sym_key] = datetime.now()
                synced += 1

            self.logger.info(
                f"MT5 position sync: loaded {synced} existing positions "
                f"({len(self.positions)} total tracked)"
            )
        except Exception as e:
            self.logger.warning(f"MT5 position sync failed: {e}")

    def _trading_loop(self):
        """Main trading loop"""
        self.logger.info("Starting trading loop")
        
        try:
            while self.is_running:
                current_time = datetime.now()
                self._refresh_account_risk_state()
                
                # Update positions
                self._update_positions()
                
                # Generate signals for trading pairs
                if self.zeropoint_pure_mode:
                    # ZP Pure Mode: scan ALL symbols, pick best by confidence
                    best_signal = None
                    for symbol in self.trading_pairs:
                        if not self.is_running:
                            break
                        try:
                            signal = self._generate_signal(symbol)
                            if signal and self._can_trade(signal):
                                if best_signal is None or signal.confidence > best_signal.confidence:
                                    best_signal = signal
                        except Exception as e:
                            self.logger.error(f"Error processing {symbol}: {e}")
                    if best_signal:
                        self._process_signal(best_signal)
                else:
                    for symbol in self.trading_pairs:
                        if not self.is_running:
                            break
                        try:
                            signal = self._generate_signal(symbol)
                            if signal:
                                self._process_signal(signal)
                        except Exception as e:
                            self.logger.error(f"Error processing {symbol}: {e}")
                            continue
                
                # Update performance metrics
                if self.last_performance_update is None or \
                   (current_time - self.last_performance_update).seconds >= 60:
                    self._update_performance_metrics()
                    self.last_performance_update = current_time
                
                # Sleep before next iteration
                time.sleep(5)  # 5-second cycles
                
        except Exception as e:
            self.logger.error(f"Critical error in trading loop: {e}")
        finally:
            self.is_running = False

    def _get_symbol_profitability_gate(self, symbol: str) -> Tuple[bool, float, float, int]:
        """
        Return profitability gate status for a symbol.

        Returns:
            Tuple of:
            - is_weak: whether symbol should be treated as weak-profitability
            - avg_trade_return
            - profit_factor
            - sample_count
        """
        try:
            metadata = getattr(self.model_manager, 'model_metadata', {}) or {}
            if not isinstance(metadata, dict):
                return False, 0.0, 0.0, 0

            profitability_map = metadata.get('symbol_profitability_validation', {})
            diagnostics_map = metadata.get('threshold_diagnostics', {})
            if not isinstance(profitability_map, dict):
                profitability_map = {}
            if not isinstance(diagnostics_map, dict):
                diagnostics_map = {}

            normalized = re.sub(r"[^A-Z0-9]", "", str(symbol or "").upper())
            aliases: List[str] = []
            if normalized:
                aliases.append(normalized)
                if normalized.endswith("USD") and len(normalized) > 3:
                    aliases.append(normalized[:-3])

            avg_trade_return = 0.0
            profit_factor = 0.0
            profit_samples = 0
            selected_threshold = 0.0
            weekly_prob_positive = 0.5
            weekly_p10_return = -1.0

            for alias in aliases:
                stats = profitability_map.get(alias)
                if isinstance(stats, dict):
                    avg_trade_return = float(stats.get('avg_trade_return', 0.0) or 0.0)
                    profit_factor = float(stats.get('profit_factor', 0.0) or 0.0)
                    profit_samples = int(
                        round(float(stats.get('trade_count', stats.get('trades', 0.0)) or 0.0))
                    )
                    selected_threshold = float(stats.get('threshold', 0.0) or 0.0)
                    weekly_prob_positive = float(stats.get('weekly_prob_positive', weekly_prob_positive) or weekly_prob_positive)
                    weekly_p10_return = float(stats.get('weekly_p10_return', weekly_p10_return) or weekly_p10_return)
                diag = diagnostics_map.get(alias)
                if isinstance(diag, dict):
                    diag_expectancy = float(diag.get('expectancy', avg_trade_return) or avg_trade_return)
                    diag_pf = float(diag.get('profit_factor', profit_factor) or profit_factor)
                    diag_trades = int(round(float(diag.get('trade_count', profit_samples) or profit_samples)))
                    diag_threshold = float(diag.get('threshold', selected_threshold) or selected_threshold)
                    diag_weekly_prob = float(diag.get('weekly_prob_positive', weekly_prob_positive) or weekly_prob_positive)
                    diag_weekly_p10 = float(diag.get('weekly_p10_return', weekly_p10_return) or weekly_p10_return)
                    if profit_samples < self.minimum_symbol_profitability_samples:
                        avg_trade_return = diag_expectancy
                        profit_factor = diag_pf
                        profit_samples = diag_trades
                        selected_threshold = diag_threshold
                        weekly_prob_positive = diag_weekly_prob
                        weekly_p10_return = diag_weekly_p10
                if profit_samples > 0 or selected_threshold > 0:
                    break

            is_weak = bool(
                (
                    profit_samples >= self.minimum_symbol_profitability_samples
                    or selected_threshold >= 0.85
                )
                and (
                    avg_trade_return <= self.minimum_symbol_expectancy
                    or profit_factor < self.minimum_symbol_profit_factor
                )
            )
            if self.tail_risk_control_enabled and profit_samples > 0:
                if (
                    weekly_p10_return < self.tail_min_weekly_p10_return
                    or weekly_prob_positive < self.tail_min_weekly_prob_positive
                ):
                    is_weak = True
            return is_weak, float(avg_trade_return), float(profit_factor), int(profit_samples)
        except Exception:
            return False, 0.0, 0.0, 0

    def _resolve_symbol_trade_threshold(self, symbol: str) -> float:
        """Resolve symbol confidence threshold from model metadata."""
        try:
            metadata = getattr(self.model_manager, 'model_metadata', {}) or {}
            if not isinstance(metadata, dict):
                return float(self.confidence_threshold)

            normalized = re.sub(r"[^A-Z0-9]", "", str(symbol or "").upper())
            aliases: List[str] = []
            if normalized:
                aliases.append(normalized)
                if normalized.endswith("USD") and len(normalized) > 3:
                    aliases.append(normalized[:-3])

            thresholds = metadata.get('symbol_thresholds', {})
            if isinstance(thresholds, dict):
                for alias in aliases:
                    value = thresholds.get(alias)
                    if value is not None:
                        try:
                            return float(value)
                        except Exception:
                            continue

            global_threshold = metadata.get('global_trade_threshold', self.confidence_threshold)
            return float(global_threshold)
        except Exception:
            return float(self.confidence_threshold)

    def _rebuild_live_symbol_profile(self) -> None:
        """Build symbol-level live trading profile from model validation metadata."""
        try:
            self._symbol_live_profile = {}
            metadata = getattr(self.model_manager, 'model_metadata', {}) or {}
            if not isinstance(metadata, dict):
                return

            profitability_map = metadata.get('symbol_profitability_validation', {})
            diagnostics_map = metadata.get('threshold_diagnostics', {})
            configured_live_profile = metadata.get('symbol_live_profile', {})
            if not isinstance(profitability_map, dict):
                profitability_map = {}
            if not isinstance(diagnostics_map, dict):
                diagnostics_map = {}
            if not isinstance(configured_live_profile, dict):
                configured_live_profile = {}
            if not profitability_map and not diagnostics_map and not configured_live_profile:
                return

            profile_cfg = metadata.get('live_profile_config', {})
            if not isinstance(profile_cfg, dict):
                profile_cfg = {}
            min_samples = int(
                profile_cfg.get(
                    'min_samples',
                    self.minimum_symbol_profitability_samples,
                )
            )
            min_profit_factor = float(
                profile_cfg.get(
                    'min_profit_factor',
                    self.minimum_symbol_profit_factor,
                )
            )
            min_expectancy = float(
                profile_cfg.get(
                    'min_expectancy',
                    self.minimum_symbol_expectancy,
                )
            )
            min_trade_rate = float(
                profile_cfg.get(
                    'min_trade_rate',
                    self.live_min_trade_rate,
                )
            )
            min_weekly_prob_positive = float(
                profile_cfg.get(
                    'min_weekly_prob_positive',
                    self.tail_min_weekly_prob_positive,
                )
            )
            min_weekly_p10_return = float(
                profile_cfg.get(
                    'min_weekly_p10_return',
                    self.tail_min_weekly_p10_return,
                )
            )
            if self.tail_risk_control_enabled:
                min_weekly_prob_positive = max(min_weekly_prob_positive, self.tail_min_weekly_prob_positive)
                min_weekly_p10_return = max(min_weekly_p10_return, self.tail_min_weekly_p10_return)
            self.minimum_symbol_profitability_samples = max(1, min_samples)
            self.minimum_symbol_profit_factor = min_profit_factor
            self.minimum_symbol_expectancy = min_expectancy
            self.live_min_trade_rate = min_trade_rate

            all_symbols = sorted(
                set(profitability_map.keys())
                | set(diagnostics_map.keys())
                | set(configured_live_profile.keys())
            )
            for symbol in all_symbols:
                stats = profitability_map.get(symbol, {})
                diag = diagnostics_map.get(symbol, {})
                configured = configured_live_profile.get(symbol, {})
                if not isinstance(stats, dict):
                    stats = {}
                if not isinstance(diag, dict):
                    diag = {}
                if not isinstance(configured, dict):
                    configured = {}

                trade_count = int(
                    round(
                        float(
                            configured.get(
                                'trade_count',
                                stats.get(
                                    'trade_count',
                                    diag.get('trade_count', 0.0),
                                ),
                            ) or 0.0
                        )
                    )
                )
                samples = int(
                    round(
                        float(
                            configured.get(
                                'samples',
                                stats.get('samples', diag.get('samples', 0.0)),
                            ) or 0.0
                        )
                    )
                )
                expectancy = float(
                    configured.get(
                        'expectancy',
                        stats.get(
                            'avg_trade_return',
                            diag.get('expectancy', 0.0),
                        ),
                    ) or 0.0
                )
                profit_factor = float(
                    configured.get(
                        'profit_factor',
                        stats.get(
                            'profit_factor',
                            diag.get('profit_factor', 0.0),
                        ),
                    ) or 0.0
                )
                trade_rate = float(
                    configured.get(
                        'trade_rate',
                        stats.get('trade_rate', (trade_count / samples) if samples > 0 else 0.0),
                    ) or 0.0
                )
                weekly_prob_positive = float(
                    configured.get(
                        'weekly_prob_positive',
                        stats.get(
                            'weekly_prob_positive',
                            diag.get('weekly_prob_positive', 0.5),
                        ),
                    ) or 0.5
                )
                weekly_p10_return = float(
                    configured.get(
                        'weekly_p10_return',
                        stats.get(
                            'weekly_p10_return',
                            diag.get('weekly_p10_return', -1.0),
                        ),
                    ) or -1.0
                )
                metric_enabled = bool(
                    trade_count >= self.minimum_symbol_profitability_samples
                    and expectancy > self.minimum_symbol_expectancy
                    and profit_factor >= self.minimum_symbol_profit_factor
                    and trade_rate >= self.live_min_trade_rate
                    and weekly_prob_positive >= min_weekly_prob_positive
                    and weekly_p10_return >= min_weekly_p10_return
                )
                configured_enabled = configured.get('enabled', None)
                if configured_enabled is None:
                    enabled = metric_enabled
                else:
                    # Never allow stale metadata to force-enable weak symbols.
                    enabled = bool(configured_enabled) and metric_enabled
                if self.tail_risk_control_enabled and enabled:
                    if (
                        weekly_prob_positive < min_weekly_prob_positive
                        or weekly_p10_return < min_weekly_p10_return
                    ):
                        enabled = False

                configured_risk_multiplier = configured.get('risk_multiplier', None)
                if configured_risk_multiplier is not None:
                    risk_multiplier = float(np.clip(float(configured_risk_multiplier or 0.0), 0.0, 2.0))
                elif enabled:
                    pf_component = np.clip((profit_factor - self.minimum_symbol_profit_factor) / 0.35, 0.0, 1.0)
                    exp_denominator = max(self.minimum_symbol_expectancy * 4.0, 1e-8)
                    exp_component = np.clip((expectancy - self.minimum_symbol_expectancy) / exp_denominator, 0.0, 1.0)
                    sample_component = np.clip(trade_count / 250.0, 0.0, 1.0)
                    risk_multiplier = float(
                        np.clip(
                            0.55 + 0.30 * pf_component + 0.25 * exp_component + 0.20 * sample_component,
                            0.55,
                            1.30,
                        )
                    )
                else:
                    risk_multiplier = 0.0

                self._symbol_live_profile[str(symbol)] = {
                    'enabled': 1.0 if enabled else 0.0,
                    'risk_multiplier': risk_multiplier,
                    'expectancy': expectancy,
                    'profit_factor': profit_factor,
                    'trade_count': float(trade_count),
                    'trade_rate': trade_rate,
                    'weekly_prob_positive': weekly_prob_positive,
                    'weekly_p10_return': weekly_p10_return,
                }

            enabled_symbols = [s for s, v in self._symbol_live_profile.items() if v.get('enabled', 0.0) >= 1.0]
            if enabled_symbols:
                self.logger.info(
                    "Profitability-first live profile enabled symbols: "
                    + ", ".join(enabled_symbols)
                )
            else:
                self.logger.warning("Profitability-first live profile enabled no symbols")
        except Exception as e:
            self.logger.error(f"Failed to build symbol live profile: {e}")
            self._symbol_live_profile = {}

    def _resolve_symbol_live_profile(self, symbol: str) -> Optional[Dict[str, float]]:
        """Resolve per-symbol live profile entry using canonical aliases."""
        if not self._symbol_live_profile:
            return None
        normalized = re.sub(r"[^A-Z0-9]", "", str(symbol or "").upper())
        aliases: List[str] = []
        if normalized:
            aliases.append(normalized)
            if normalized.endswith("USD") and len(normalized) > 3:
                aliases.append(normalized[:-3])
        for alias in aliases:
            entry = self._symbol_live_profile.get(alias)
            if isinstance(entry, dict):
                return entry
        return None

    def _check_correlation_exposure(self, symbol: str, action: str) -> Tuple[bool, float]:
        """Check if opening a new position would breach correlation exposure limits.

        Checks two things per group:
          1. Same-direction stacking (e.g. AUDUSD SELL + NZDUSD SELL = both short AUD/NZD, long USD)
             → reduce size at 1, block at 2
          2. Conflicting direction (e.g. GBPUSD BUY=short USD + USDCAD BUY=long USD)
             → block: contradictory fundamental bets on the same underlying

        Returns (allowed, size_factor):
          - allowed=False means the trade should be blocked
          - size_factor is a multiplier (1.0 = full size, 0.60 = reduced)
        """
        normalized = re.sub(r"[^A-Z0-9]", "", str(symbol or "").upper())
        action_sign = +1 if action == "BUY" else -1
        worst_factor = 1.0

        for group_name, members in self.correlation_groups.items():
            if normalized not in members:
                continue
            # Determine the "net direction" this new trade would represent in the group
            new_net = action_sign * members[normalized]

            # Count existing open positions in this group by net direction
            same_dir_count = 0
            opposite_dir_count = 0
            for pos in self.positions.values():
                if getattr(pos, 'status', 'OPEN') != 'OPEN':
                    continue
                pos_sym = re.sub(r"[^A-Z0-9]", "", str(pos.symbol or "").upper())
                if pos_sym not in members or pos_sym == normalized:
                    continue
                pos_sign = +1 if getattr(pos, 'action', '') == 'BUY' else -1
                pos_net = pos_sign * members[pos_sym]
                if pos_net == new_net:
                    same_dir_count += 1
                elif pos_net == -new_net:
                    opposite_dir_count += 1

            # Block contradictory bets — e.g. betting USD up AND down simultaneously
            if opposite_dir_count >= 1:
                self.logger.info(
                    f"Correlation CONFLICT block: {symbol} {action} (net={'long' if new_net > 0 else 'short'}) "
                    f"contradicts {opposite_dir_count} opposite-direction position(s) in {group_name}"
                )
                return (False, 0.0)

            # Block excessive same-direction stacking
            if same_dir_count >= self.max_correlated_same_direction:
                self.logger.info(
                    f"Correlation STACK block: {symbol} {action} would exceed "
                    f"{self.max_correlated_same_direction} same-direction in {group_name} "
                    f"(already {same_dir_count} open)"
                )
                return (False, 0.0)
            elif same_dir_count >= 1:
                worst_factor = min(worst_factor, self.correlated_size_reduction)

        return (True, worst_factor)

    def _get_global_max_lot(self, balance: float) -> float:
        """Return maximum allowed lot size based on account balance."""
        for threshold, max_lot in self.global_max_lot_table:
            if balance < threshold:
                return max_lot
        return 5.0

    def _is_symbol_live_enabled(self, symbol: str) -> bool:
        """Check if symbol is allowed for live entries by profitability profile."""
        normalized = re.sub(r"[^A-Z0-9]", "", str(symbol or "").upper())
        if normalized in self.excluded_symbols or symbol in self.excluded_symbols:
            return False
        if not self.profitability_first_mode:
            return True
        if not self._symbol_live_profile:
            return True
        entry = self._resolve_symbol_live_profile(symbol)
        if entry is None:
            return False
        return bool(entry.get('enabled', 0.0) >= 1.0)

    def _get_symbol_risk_multiplier(self, symbol: str) -> float:
        """Return symbol risk multiplier derived from profitability profile."""
        entry = self._resolve_symbol_live_profile(symbol)
        if entry is None:
            return 1.0
        if entry.get('enabled', 0.0) < 1.0:
            # When profitability-first mode is off, disabled symbols still get
            # a baseline risk multiplier so they can trade immediately.
            if not self.profitability_first_mode:
                return 0.50
            return 0.0
        return float(np.clip(float(entry.get('risk_multiplier', 1.0) or 1.0), 0.0, 2.0))

    def _log_symbol_profile_skip(self, symbol: str, reason: str, cooldown_seconds: int = 120) -> None:
        """Rate-limit repetitive profile skip logs per symbol."""
        now = datetime.now()
        key = str(symbol or "").upper()
        last = self._symbol_profile_skip_log_time.get(key)
        if last and (now - last).total_seconds() < cooldown_seconds:
            return
        self._symbol_profile_skip_log_time[key] = now
        self.logger.info(f"Skipping {symbol}: {reason}")

    def _refresh_account_risk_state(self, force_reset: bool = False) -> None:
        """Refresh intraday risk state and activate account-level pause when limits are hit."""
        try:
            account_info = self.mt5_connector.get_account_info()
            if not account_info:
                return
            equity = float(account_info.get('equity', 0.0) or 0.0)
            balance = float(account_info.get('balance', 0.0) or 0.0)
            if equity <= 0.0:
                equity = balance
            if equity <= 0.0:
                return

            now = datetime.now()
            day_key = now.strftime("%Y-%m-%d")
            if force_reset or self._daily_risk_day != day_key or self._daily_start_equity <= 0.0:
                self._daily_risk_day = day_key
                self._daily_start_equity = equity
                self._intraday_peak_equity = equity
                self._daily_loss_pause_until = None
                self._daily_loss_size_factor = 1.0
                self.performance_metrics['daily_pnl'] = 0.0
                return

            self._intraday_peak_equity = max(self._intraday_peak_equity, equity)
            daily_pnl = equity - self._daily_start_equity
            self.performance_metrics['daily_pnl'] = daily_pnl

            daily_loss_ratio = (
                (self._daily_start_equity - equity) / self._daily_start_equity
                if self._daily_start_equity > 0
                else 0.0
            )
            intraday_drawdown = (
                (self._intraday_peak_equity - equity) / self._intraday_peak_equity
                if self._intraday_peak_equity > 0
                else 0.0
            )
            self.performance_metrics['current_drawdown'] = max(
                float(self.performance_metrics.get('current_drawdown', 0.0) or 0.0),
                float(intraday_drawdown),
            )

            # Progressive daily loss tiers — reduce size as losses accumulate
            prev_factor = self._daily_loss_size_factor
            new_factor = 1.0
            for tier_pct, tier_factor in self._daily_loss_tiers:
                if daily_loss_ratio >= tier_pct:
                    new_factor = tier_factor
            self._daily_loss_size_factor = new_factor
            if new_factor != prev_factor:
                self.logger.warning(
                    f"Daily loss tier change: size_factor {prev_factor:.2f} → {new_factor:.2f} "
                    f"(daily_loss={daily_loss_ratio:.2%})"
                )

            # Full stop at 15%+ daily loss — pause until next day
            if daily_loss_ratio >= 0.15:
                if not self._daily_loss_pause_until or now >= self._daily_loss_pause_until:
                    self._daily_loss_pause_until = datetime(
                        year=now.year, month=now.month, day=now.day
                    ) + timedelta(days=1)
                    self.logger.warning(
                        f"FULL STOP: 15%+ daily loss ({daily_loss_ratio:.2%}). "
                        f"Paused until {self._daily_loss_pause_until.strftime('%Y-%m-%d %H:%M:%S')}"
                    )
            else:
                hit_daily_loss = daily_loss_ratio >= self.max_daily_loss_pct
                hit_intraday_dd = intraday_drawdown >= self.max_intraday_drawdown_pct
                if hit_daily_loss or hit_intraday_dd:
                    if not self._daily_loss_pause_until or now >= self._daily_loss_pause_until:
                        if self.loss_pause_until_next_day:
                            self._daily_loss_pause_until = datetime(
                                year=now.year, month=now.month, day=now.day
                            ) + timedelta(days=1)
                        else:
                            self._daily_loss_pause_until = now + timedelta(hours=2)
                        self.logger.warning(
                            "Account risk pause activated: "
                            f"daily_loss={daily_loss_ratio:.2%}, "
                            f"intraday_drawdown={intraday_drawdown:.2%}, "
                            f"paused_until={self._daily_loss_pause_until.strftime('%Y-%m-%d %H:%M:%S')}"
                        )
        except Exception as e:
            self.logger.error(f"Failed to refresh account risk state: {e}")

    def _is_account_risk_paused(self) -> bool:
        """Return True if account-level risk pause is active."""
        if not self._daily_loss_pause_until:
            return False
        now = datetime.now()
        if now >= self._daily_loss_pause_until:
            self._daily_loss_pause_until = None
            return False
        return True

    def _register_closed_position(self, position: Position) -> None:
        """Track close events for symbol loss-streak cooldown."""
        try:
            if position.close_recorded:
                return
            position.close_recorded = True

            symbol_key = str(position.symbol or "").upper()
            pnl = float(position.unrealized_pnl or 0.0)
            if pnl < 0.0:
                streak = int(self._symbol_loss_streak.get(symbol_key, 0)) + 1
                self._symbol_loss_streak[symbol_key] = streak
                if streak >= self.loss_streak_limit:
                    block_until = datetime.now() + timedelta(seconds=self.loss_streak_cooldown_seconds)
                    self._symbol_loss_block_until[symbol_key] = block_until
                    self._symbol_loss_streak[symbol_key] = 0
                    self.logger.warning(
                        f"Loss-streak cooldown for {symbol_key}: "
                        f"{self.loss_streak_limit} consecutive losses, "
                        f"paused until {block_until.strftime('%Y-%m-%d %H:%M:%S')}"
                    )
            elif pnl > 0.0:
                self._symbol_loss_streak[symbol_key] = 0

            # Notify agentic orchestrator with full trade context.
            if self.orchestrator is not None:
                try:
                    from trade_journal import TradeRecord

                    entry_val = float(position.entry_price * position.position_size) if position.entry_price else 1.0
                    record = TradeRecord(
                        trade_id=f"{position.ticket}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                        symbol=symbol_key,
                        direction=str(position.action or ''),
                        entry_price=float(position.entry_price or 0.0),
                        exit_price=float(position.current_price or 0.0),
                        pnl=pnl,
                        pnl_pct=pnl / entry_val if entry_val > 0 else 0.0,
                        model_confidence=float(getattr(position, 'model_confidence', 0.0)),
                        model_action=str(getattr(position, 'model_action', '')),
                        model_probabilities=str(getattr(position, 'model_probabilities', '{}')),
                        symbol_threshold=float(getattr(position, 'symbol_threshold', 0.0)),
                        action_mode=str(getattr(position, 'action_mode', 'normal')),
                        entry_time=position.open_time.isoformat() if hasattr(position, 'open_time') else '',
                        exit_time=datetime.now().isoformat(),
                        position_size=float(position.position_size or 0.0),
                        model_version=str(getattr(position, 'model_version', '')),
                        stop_loss=float(position.stop_loss or 0.0),
                        take_profit=float(position.take_profit or 0.0),
                    )
                    self.orchestrator.notify_trade_closed(record)
                except Exception as journal_err:
                    self.logger.debug(f"Trade journal write failed: {journal_err}")
        except Exception as e:
            self.logger.error(f"Failed to register closed position: {e}")
    
    def _generate_zp_pure_signal(self, symbol: str) -> Optional[TradingSignal]:
        """Generate signal using ONLY ZeroPoint H4 ATR trailing stop flips.

        Skips neural model, pattern recognition, profitability gates, and all
        other signal sources.  Used when zeropoint_pure_mode is True.
        """
        norm = symbol.upper().replace(".", "").replace("#", "")
        if norm in self.zeropoint_skip_symbols:
            return None

        symbol_info = self.mt5_connector.get_symbol_info(symbol)
        if not symbol_info:
            return None
        if not self._is_spread_acceptable(symbol_info):
            return None

        # Fetch H4 + H1 data
        h4_rates = self.mt5_connector.get_rates(symbol, mt5.TIMEFRAME_H4, 0, 200)
        h1_rates = self.mt5_connector.get_rates(symbol, mt5.TIMEFRAME_H1, 0, 200)
        df_h4 = self._prepare_ohlc_dataframe(h4_rates) if h4_rates else None
        df_h1 = self._prepare_ohlc_dataframe(h1_rates) if h1_rates else None
        if df_h4 is None or len(df_h4) < 15:
            return None

        # Try standard ZP engine; fall back to raw for non-enabled symbols (e.g. EURUSD)
        zp_signal = None
        if self.zeropoint_engine.is_symbol_enabled(norm):
            zp_signal = self.zeropoint_engine.generate_signal(symbol, df_h4, df_h1)
        if zp_signal is None:
            zp_signal = self._generate_zp_signal_raw(symbol, df_h4, df_h1)
        if zp_signal is None:
            return None

        entry_price = symbol_info['ask'] if zp_signal.direction == 'BUY' else symbol_info['bid']
        digits = int(symbol_info.get('digits', 5))

        return TradingSignal(
            symbol=symbol,
            action=zp_signal.direction,
            confidence=zp_signal.confidence,
            entry_price=entry_price,
            stop_loss=round(zp_signal.stop_loss, digits),
            take_profit=round(zp_signal.tp1, digits),
            position_size=self.zeropoint_fixed_lot,
            reason=(
                f"ZEROPOINT-PURE {zp_signal.direction} "
                f"(Tier={zp_signal.tier}, R:R={zp_signal.risk_reward:.1f}, "
                f"Conf={zp_signal.confidence:.2f})"
            ),
        )

    @staticmethod
    def _generate_zp_signal_raw(symbol, df_h4, df_h1=None):
        """Generate ZP signal for any symbol (bypasses ZEROPOINT_ENABLED_SYMBOLS).

        Produces a signal whenever ZP has an active position (BULL or BEAR),
        not just on flip candles.  If the trailing stop is still alive and
        there's room to TP, it's a valid entry.
        """
        zp = compute_zeropoint_state(df_h4)
        if zp is None or len(zp) < 2:
            return None

        last = zp.iloc[-1]
        pos = int(last.get("pos", 0))

        # Must have an active ZP position
        if pos == 0:
            return None

        direction = "BUY" if pos == 1 else "SELL"

        # Check how fresh the flip is (for confidence scoring)
        buy_sig = bool(last.get("buy_signal", False))
        sell_sig = bool(last.get("sell_signal", False))
        is_fresh_flip = buy_sig or sell_sig
        if not is_fresh_flip and len(zp) >= 2:
            prev = zp.iloc[-2]
            is_fresh_flip = bool(prev.get("buy_signal", False)) or bool(prev.get("sell_signal", False))

        # Count bars since flip
        bars_in_pos = 1
        for idx in range(len(zp) - 2, -1, -1):
            if int(zp.iloc[idx].get("pos", 0)) == pos:
                bars_in_pos += 1
            else:
                break

        entry = float(last["close"])
        atr_val = float(last["atr"])
        if atr_val <= 0 or not np.isfinite(atr_val):
            return None

        trailing_stop = float(last.get("xATRTrailingStop", 0))
        if trailing_stop <= 0:
            return None

        from .zeropoint_signal import (
            SWING_LOOKBACK, SL_BUFFER_PCT, SL_ATR_MIN_MULT, TP1_MULT,
        )

        # SL = ZP trailing stop (the actual ATR stop the indicator uses)
        sl = trailing_stop
        # Add small buffer beyond the trailing stop
        buffer = atr_val * SL_BUFFER_PCT
        if direction == "BUY":
            sl = sl - buffer
            tp1 = entry + atr_val * TP1_MULT
        else:
            sl = sl + buffer
            tp1 = entry - atr_val * TP1_MULT

        sl_dist = abs(entry - sl)
        tp_dist = abs(tp1 - entry)
        rr = tp_dist / sl_dist if sl_dist > 0 else 0

        # If there's no room left to TP (price already past TP), skip
        if direction == "BUY" and entry >= tp1:
            return None
        if direction == "SELL" and entry <= tp1:
            return None

        # H1 confirmation
        h1_conf = False
        if df_h1 is not None:
            zp_h1 = compute_zeropoint_state(df_h1)
            if zp_h1 is not None and len(zp_h1) > 0:
                h1_pos = int(zp_h1.iloc[-1].get("pos", 0))
                if direction == "BUY" and h1_pos == 1:
                    h1_conf = True
                elif direction == "SELL" and h1_pos == -1:
                    h1_conf = True

        # Confidence: fresh flips get highest, ongoing positions get moderate
        if is_fresh_flip:
            conf = 0.70 + (0.15 if h1_conf else 0.0) + min(rr * 0.05, 0.10)
        else:
            # Ongoing position — slightly lower confidence, decays with age
            age_penalty = min(bars_in_pos * 0.02, 0.15)  # -2% per bar, max -15%
            conf = 0.65 + (0.12 if h1_conf else 0.0) + min(rr * 0.05, 0.08) - age_penalty
        conf = max(0.40, min(conf, 0.98))

        # Determine tier based on freshness + H1 alignment
        if is_fresh_flip and h1_conf:
            tier = "S"
        elif is_fresh_flip:
            tier = "A"
        elif h1_conf and bars_in_pos <= 5:
            tier = "B"
        else:
            tier = "C"

        return ZeroPointSignal(
            symbol=symbol, direction=direction, entry_price=entry,
            stop_loss=sl, tp1=tp1, tp2=tp1, tp3=tp1,
            atr_value=atr_val, confidence=conf,
            signal_time=datetime.now(), timeframe="H4",
            tier=tier, trailing_stop=trailing_stop,
            risk_reward=rr,
        )

    def _generate_signal(self, symbol: str) -> Optional[TradingSignal]:
        """
        Generate neural trading signal for a symbol

        Args:
            symbol: Trading symbol

        Returns:
            TradingSignal object or None
        """
        try:
            # ZeroPoint Pure Mode — skip neural/pattern, only ZP H4 flips
            if self.zeropoint_pure_mode:
                return self._generate_zp_pure_signal(symbol)

            # Get market data
            market_data = self._get_market_data(symbol)
            if not market_data:
                return None

            # Get symbol info for trading
            symbol_info = self.mt5_connector.get_symbol_info(symbol)
            if not symbol_info:
                return None

            # Enhanced tail risk protection - analyze market conditions
            try:
                # Convert market_data to DataFrame for analysis
                if isinstance(market_data.get('rates'), list):
                    df = pd.DataFrame(market_data['rates'])
                    if not df.empty:
                        df['datetime'] = pd.to_datetime(df['time'], unit='s')
                        df.set_index('datetime', inplace=True)
                        market_conditions = self.tail_risk_protector.analyze_market_conditions(df)
                    else:
                        market_conditions = self.tail_risk_protector._get_default_risk_indicators()
                else:
                    market_conditions = self.tail_risk_protector._get_default_risk_indicators()
            except Exception as e:
                self.logger.warning(f"Failed to analyze market conditions for {symbol}: {e}")
                market_conditions = self.tail_risk_protector._get_default_risk_indicators()
            if not self._is_spread_acceptable(symbol_info):
                self.logger.debug(f"Spread too wide for {symbol}; skipping signal generation this cycle")
                return None
            if not self._is_symbol_live_enabled(symbol):
                self._log_symbol_profile_skip(symbol, "blocked by profitability-first live profile")
                return None
            risk_multiplier = self._get_symbol_risk_multiplier(symbol)
            if risk_multiplier <= 0.0:
                self._log_symbol_profile_skip(symbol, "non-positive live risk multiplier")
                return None

            profitability_weak, avg_trade_return, profit_factor, profit_samples = (
                self._get_symbol_profitability_gate(symbol)
            )
            quality_degraded_mode = False

            # If model validation quality is weak for this symbol, only allow
            # historical MTF entries with explicit positive edge.
            quality = self._get_model_symbol_quality(symbol)
            if quality:
                quality_win_rate, quality_samples = quality
                if (
                    quality_win_rate < self.minimum_symbol_quality_winrate
                    or quality_samples < self.minimum_symbol_quality_samples
                ):
                    self.logger.debug(
                        f"Model quality weak for {symbol} "
                        f"(dir_win={quality_win_rate:.3f}, samples={quality_samples}); "
                        "using historical MTF gate only"
                    )
                    if profitability_weak:
                        self.logger.debug(
                            f"Skipping {symbol}: weak directional quality and weak profitability "
                            f"(avg={avg_trade_return:.6f}, pf={profit_factor:.3f}, samples={profit_samples})"
                        )
                        return None
                    historical_signal = self._generate_historical_mtf_signal(symbol, symbol_info, market_data)
                    if historical_signal is not None:
                        return historical_signal

                    # Historical MTF fallback had no edge-valid setup.
                    # Allow neural path with stricter edge/confidence requirements.
                    quality_degraded_mode = True

            if profitability_weak and self.profitability_first_mode:
                self.logger.debug(
                    f"Model profitability weak for {symbol} "
                    f"(avg={avg_trade_return:.6f}, pf={profit_factor:.3f}, samples={profit_samples}); "
                    "skipping symbol this cycle"
                )
                return None
            
            # Extract features
            features = self._extract_features(symbol, market_data)
            if features is None:
                return None
            
            # Get neural prediction
            prediction = self.model_manager.predict(features, symbol=symbol)
            if not prediction:
                return self._generate_startup_fallback_signal(symbol, symbol_info, market_data)

            action = str(prediction.get('action', 'HOLD')).upper()
            confidence = float(prediction.get('confidence', 0.0))
            model_threshold = float(prediction.get('trade_threshold', self.confidence_threshold))
            if self.profitability_first_mode or self.immediate_trade_mode:
                required_confidence = model_threshold
            else:
                required_confidence = max(self.confidence_threshold, model_threshold)
            should_trade = bool(prediction.get('should_trade', True))
            probabilities = prediction.get('probabilities', {}) if isinstance(prediction, dict) else {}
            buy_prob = float(probabilities.get('BUY', 0.0) or 0.0)
            sell_prob = float(probabilities.get('SELL', 0.0) or 0.0)
            directional_gap = abs(buy_prob - sell_prob)
            trade_score = float(prediction.get('trade_score', max(buy_prob, sell_prob)) or 0.0)

            # Track whether the neural model produced a valid directional signal.
            # If not, we still allow pattern-only trades downstream.
            neural_valid = True

            if action not in ('BUY', 'SELL'):
                neural_valid = False

            if neural_valid and (not should_trade or confidence < required_confidence):
                neural_valid = False

            min_trade_score = self.model_min_trade_score
            min_directional_gap = self.model_min_directional_gap

            if neural_valid and (
                trade_score < min_trade_score
                or directional_gap < min_directional_gap
            ):
                self.logger.debug(
                    f"Skipping {symbol}: weak model edge "
                    f"(score={trade_score:.3f}, gap={directional_gap:.3f})"
                )
                neural_valid = False

            # M15 candle pattern checks & MTF alignment — only when neural
            # produced a valid directional signal.
            pattern_description = ""
            pattern_confirmed = False

            if neural_valid:
                pattern_action, pattern_rule, pattern_streak = self._get_recent_candle_pattern_signal(
                    market_data,
                    timeframe='M15',
                )
                if pattern_action in ('BUY', 'SELL'):
                    if pattern_rule == 'continuation_4plus':
                        pattern_description = (
                            f"M15 continuation: {pattern_streak} same-direction candles -> {pattern_action}"
                        )
                    elif pattern_rule == 'reversal_3':
                        pattern_description = (
                            f"M15 reversal: 3 same-direction candles -> {pattern_action}"
                        )
                    else:
                        pattern_description = f"M15 pattern signal -> {pattern_action}"

                    if self.model_pattern_conflict_block and action != pattern_action:
                        self.logger.debug(
                            f"Skipping {symbol}: model={action} conflicts with M15 pattern={pattern_action} "
                            f"(rule={pattern_rule}, streak={pattern_streak})"
                        )
                        return None
                    if action == pattern_action:
                        pattern_confirmed = True
                        confidence = float(np.clip(max(confidence, required_confidence), 0.0, 1.0))

                # Enforce multi-timeframe alignment for live model signals.
                if self.mtf_alignment_enabled:
                    mtf_action, mtf_score, _ = self._get_current_mtf_action(market_data)
                    if mtf_action in ('BUY', 'SELL') and mtf_action != action:
                        self.logger.debug(
                            f"Skipping {symbol} model signal {action}: MTF alignment is {mtf_action} (score {mtf_score})"
                        )
                        return None
            
            # ----------------------------------------------------------
            # Chart-pattern confluence detection
            # ----------------------------------------------------------
            pattern_result = None
            pattern_sl = None
            pattern_tp = None
            pattern_source = ""
            pattern_size_mult = 1.0   # multiplier for position sizing

            try:
                pattern_result = self._detect_pattern_signal(symbol, market_data)
            except Exception as e:
                self.logger.debug(f"Pattern scan error {symbol}: {e}")

            if neural_valid and pattern_result is not None:
                pat, pat_tf, pat_score = pattern_result
                pat_dir = pat.direction.lower()  # "bullish" / "bearish"

                # Map pattern direction to BUY/SELL
                pat_action = "BUY" if pat_dir == "bullish" else "SELL" if pat_dir == "bearish" else None

                if pat_action is not None:
                    # Extract pattern-derived SL / TP
                    pat_sl_raw = pat.details.get("stop_loss")
                    pat_tp_raw = pat.details.get("target_price")

                    if pat_sl_raw and pat_tp_raw:
                        pattern_sl = float(pat_sl_raw)
                        pattern_tp = float(pat_tp_raw)

                    if pat_action == action:
                        # ---- CONFLUENCE: Neural + Pattern agree ----
                        confidence = float(np.clip(confidence + 0.05, 0.0, 1.0))
                        pattern_source = (
                            f"CONFLUENCE Neural+{pat.name}({pat_tf}, "
                            f"conf={pat.confidence:.2f}, score={pat_score:.3f})"
                        )
                        self.logger.info(
                            f"{symbol} confluence: neural {action} + {pat.name} on {pat_tf}"
                        )
                    else:
                        # Neural and pattern disagree — ignore pattern for SL/TP
                        pattern_sl = None
                        pattern_tp = None
                        pattern_source = ""

            # ----------------------------------------------------------
            # Pattern-only trade (neural HOLD / weak / invalid but pattern is strong)
            # ----------------------------------------------------------
            if not neural_valid and pattern_result is not None:
                pat, pat_tf, pat_score = pattern_result
                pat_dir = pat.direction.lower()
                pat_action = "BUY" if pat_dir == "bullish" else "SELL" if pat_dir == "bearish" else None

                if (
                    pat_action is not None
                    and pat.confidence >= 0.75
                    and pat.details.get("target_price")
                    and pat.details.get("stop_loss")
                ):
                    pat_entry = symbol_info['ask'] if pat_action == 'BUY' else symbol_info['bid']
                    _p_sl = float(pat.details["stop_loss"])
                    _p_tp = float(pat.details["target_price"])
                    _p_risk = abs(pat_entry - _p_sl)
                    _p_reward = abs(_p_tp - pat_entry)
                    _p_rr = _p_reward / _p_risk if _p_risk > 0 else 0.0

                    if _p_rr >= 2.0:
                        # Override neural HOLD with pattern-only signal at 50 % size
                        action = pat_action
                        confidence = float(np.clip(pat.confidence, 0.0, 1.0))
                        pattern_sl = _p_sl
                        pattern_tp = _p_tp
                        pattern_size_mult = 0.50
                        pattern_source = (
                            f"PATTERN-ONLY {pat.name}({pat_tf}, "
                            f"conf={pat.confidence:.2f}, R:R={_p_rr:.1f})"
                        )
                        self.logger.info(
                            f"{symbol} pattern-only trade: {pat.name} on {pat_tf} "
                            f"(R:R={_p_rr:.1f}, 50%% size)"
                        )

            # ----------------------------------------------------------
            # Multi-TF forming pattern detection — AGENTIC approach
            # The neural model was trained WITH forming-pattern features,
            # so when it outputs BUY/SELL it already incorporates them.
            # We scan forming patterns here ONLY to:
            #   a) Provide structural SL/TP when the model agrees
            #   b) Boost confidence when neural + forming align
            #   c) If neural is HOLD, let a very strong forming pattern
            #      override at reduced size (the model's features still
            #      feed into this — it just means the model sees something
            #      forming but doesn't have enough historical confidence yet)
            # ----------------------------------------------------------
            forming_result = None
            try:
                forming_result = self._detect_forming_pattern_signal(symbol, market_data)
            except Exception as e:
                self.logger.debug(f"Forming pattern scan error {symbol}: {e}")

            if forming_result is not None:
                fp, fp_tf, fp_score = forming_result
                fp_action = "BUY" if fp.predicted_direction == "bullish" else "SELL"
                fp_entry = symbol_info['ask'] if fp_action == 'BUY' else symbol_info['bid']
                fp_risk = abs(fp_entry - fp.stop_loss)
                fp_reward = abs(fp.target_price - fp_entry)
                fp_rr = fp_reward / fp_risk if fp_risk > 0 else 0.0

                if neural_valid and action in ('BUY', 'SELL'):
                    # --- Neural already decided (BUY/SELL) ---
                    if fp_action == action:
                        # Neural + forming pattern agree → use structural SL/TP + boost
                        confidence = float(np.clip(confidence * 1.15, 0.0, 1.0))
                        if fp_rr >= 1.5:
                            pattern_sl = fp.stop_loss
                            pattern_tp = fp.target_price
                        pattern_source = pattern_source or (
                            f"NEURAL+FORMING {fp.name}({fp_tf}, "
                            f"complete={fp.completion_pct:.0%}, R:R={fp_rr:.1f})"
                        )
                        self.logger.info(
                            f"{symbol} neural+forming confluence: {action} + {fp.name} on {fp_tf} "
                            f"(completion={fp.completion_pct:.0%}, R:R={fp_rr:.1f})"
                        )
                    # If they disagree, neural's trained judgment wins — ignore forming pattern

                elif action not in ('BUY', 'SELL'):
                    # --- Neural said HOLD, but forming pattern detected ---
                    # Only override if R:R is acceptable — no hardcoded completion/confidence gates.
                    # The model already saw the forming features and chose HOLD, so we use
                    # reduced sizing.  The agentic retrainer will learn from these outcomes.
                    if fp_rr >= 1.5:
                        action = fp_action
                        confidence = float(np.clip(fp.confidence * fp.completion_pct, 0.0, 1.0))
                        pattern_sl = fp.stop_loss
                        pattern_tp = fp.target_price
                        # Scale size by completion: more complete = more size (35-65%)
                        pattern_size_mult = float(np.clip(
                            0.35 + 0.30 * fp.completion_pct, 0.35, 0.65
                        ))
                        pattern_source = (
                            f"PREDICTIVE {fp.name}({fp_tf}, "
                            f"complete={fp.completion_pct:.0%}, conf={fp.confidence:.2f}, "
                            f"R:R={fp_rr:.1f}, size={pattern_size_mult:.0%})"
                        )
                        self.logger.info(
                            f"{symbol} PREDICTIVE trade: {fp.name} on {fp_tf} "
                            f"(completion={fp.completion_pct:.0%}, R:R={fp_rr:.1f}, "
                            f"size={pattern_size_mult:.0%})"
                        )

            # ----------------------------------------------------------
            # ZeroPoint PRO H4 ATR trailing stop signals
            # ----------------------------------------------------------
            # Integration modes:
            #   1. Neural BUY/SELL + ZeroPoint same direction = CONFLUENCE (boost conf + use ZP SL/TP)
            #   2. Neural HOLD + ZeroPoint signal = STANDALONE ZP trade at reduced size
            #   3. Neural vs ZeroPoint disagree = neural wins (ignore ZP)
            # ----------------------------------------------------------
            zp_signal = None
            if self.zeropoint_enabled and self.zeropoint_engine.is_symbol_enabled(symbol):
                try:
                    h4_rates = market_data.get('H4')
                    h1_rates = market_data.get('H1')
                    df_h4 = self._prepare_ohlc_dataframe(h4_rates) if h4_rates else None
                    df_h1 = self._prepare_ohlc_dataframe(h1_rates) if h1_rates else None

                    if df_h4 is not None and len(df_h4) >= 15:
                        zp_signal = self.zeropoint_engine.generate_signal(symbol, df_h4, df_h1)
                except Exception as e:
                    self.logger.debug(f"ZeroPoint signal error {symbol}: {e}")

            if zp_signal is not None:
                if neural_valid and action in ('BUY', 'SELL'):
                    # Mode 1: CONFLUENCE — neural and ZeroPoint agree
                    if zp_signal.direction == action:
                        confidence = float(np.clip(
                            confidence + self.zeropoint_confluence_boost, 0.0, 1.0
                        ))
                        # Use ZeroPoint's structural SL/TP (backtest-validated)
                        if pattern_sl is None:
                            pattern_sl = zp_signal.stop_loss
                            pattern_tp = zp_signal.tp1  # TP1 exit is most profitable
                        pattern_source = pattern_source or (
                            f"NEURAL+ZEROPOINT {zp_signal.direction} "
                            f"(Tier={zp_signal.tier}, R:R={zp_signal.risk_reward:.1f}, "
                            f"ATR={zp_signal.atr_value:.5f})"
                        )
                        self.logger.info(
                            f"{symbol} NEURAL+ZEROPOINT confluence: {action} "
                            f"(ZP tier={zp_signal.tier}, R:R={zp_signal.risk_reward:.1f})"
                        )
                    # If neural and ZP disagree, neural wins — ignore ZP
                    else:
                        self.logger.debug(
                            f"{symbol} neural={action} vs ZeroPoint={zp_signal.direction} -- neural wins"
                        )

                elif action not in ('BUY', 'SELL'):
                    # Mode 2: STANDALONE ZeroPoint trade (neural HOLD)
                    if zp_signal.risk_reward >= 1.3:
                        action = zp_signal.direction
                        confidence = float(np.clip(zp_signal.confidence, 0.0, 1.0))
                        pattern_sl = zp_signal.stop_loss
                        pattern_tp = zp_signal.tp1  # Exit at TP1 (backtest-validated)
                        pattern_size_mult = self.zeropoint_standalone_size
                        pattern_source = (
                            f"ZEROPOINT-H4 {zp_signal.direction} "
                            f"(Tier={zp_signal.tier}, R:R={zp_signal.risk_reward:.1f}, "
                            f"Conf={zp_signal.confidence:.2f}, ATR={zp_signal.atr_value:.5f}, "
                            f"size={self.zeropoint_standalone_size:.0%})"
                        )
                        self.logger.info(
                            f"{symbol} ZEROPOINT standalone: {action} "
                            f"(tier={zp_signal.tier}, R:R={zp_signal.risk_reward:.1f}, "
                            f"{self.zeropoint_standalone_size:.0%} size)"
                        )

            # If action is still HOLD after pattern-only + predictive + ZeroPoint checks, fall through
            if action not in ('BUY', 'SELL'):
                return self._generate_startup_fallback_signal(symbol, symbol_info, market_data)

            # ----------------------------------------------------------
            # Calculate trading parameters (SL / TP)
            # ----------------------------------------------------------
            entry_price = symbol_info['ask'] if action == 'BUY' else symbol_info['bid']
            digits = int(symbol_info.get('digits', 5))

            if pattern_sl is not None and pattern_tp is not None:
                # Use pattern-derived structural SL/TP (or ZeroPoint SL/TP)
                stop_loss = round(pattern_sl, digits)
                take_profit = round(pattern_tp, digits)
            else:
                # Fallback: ATR-based SL/TP
                stop_loss, take_profit = self._calculate_sl_tp(
                    symbol, action, entry_price, symbol_info, market_data
                )

            # ----------------------------------------------------------
            # R:R ≥ 2.0 filter (pattern-based trades only)
            # ----------------------------------------------------------
            if pattern_sl is not None and pattern_tp is not None:
                risk_dist = abs(entry_price - stop_loss)
                reward_dist = abs(take_profit - entry_price)
                rr_ratio = reward_dist / risk_dist if risk_dist > 0 else 0.0
                if rr_ratio < 2.0:
                    self.logger.debug(
                        f"Skipping {symbol}: pattern R:R too low ({rr_ratio:.2f} < 2.0)"
                    )
                    return None

            # ----------------------------------------------------------
            # Candle close confirmation (pattern-based trades)
            # ----------------------------------------------------------
            if pattern_result is not None and pattern_sl is not None:
                pat, pat_tf, _ = pattern_result
                try:
                    tf_rates = market_data.get(pat_tf)
                    if tf_rates:
                        df_check = self._prepare_ohlc_dataframe(tf_rates)
                        if df_check is not None:
                            # Pattern breakout candle must be a CLOSED candle.
                            # MT5 get_rates returns *completed* candles, so index
                            # len(df)-1 is the last completed candle.  If the
                            # pattern's breakout index equals that last candle we
                            # accept it; if index_end == len(df) (would be
                            # current live candle), we wait.
                            last_closed_idx = len(df_check) - 1
                            if pat.index_end > last_closed_idx:
                                self.logger.debug(
                                    f"Skipping {symbol}: pattern breakout candle not yet closed "
                                    f"on {pat_tf}"
                                )
                                return None
                except Exception as e:
                    self.logger.debug(f"Candle-close check error {symbol}: {e}")

            # ----------------------------------------------------------
            # Retest detection — boost confidence if retest confirmed
            # ----------------------------------------------------------
            if pattern_result is not None and pattern_sl is not None:
                pat, pat_tf, _ = pattern_result
                try:
                    tf_rates = market_data.get(pat_tf)
                    if tf_rates:
                        df_retest = self._prepare_ohlc_dataframe(tf_rates)
                        if df_retest is not None and len(df_retest) > 0:
                            current_close = float(df_retest['close'].iloc[-1])
                            retest_ok = PatternRecognizer.detect_retest(pat, current_close)
                            if retest_ok:
                                confidence = float(np.clip(confidence + 0.05, 0.0, 1.0))
                                pattern_source += " [retest confirmed]"
                                self.logger.info(
                                    f"{symbol} retest confirmed for {pat.name}"
                                )
                            else:
                                # No retest — reduce position size by 25 %
                                pattern_size_mult *= 0.75
                except Exception as e:
                    self.logger.debug(f"Retest check error {symbol}: {e}")

            # ----------------------------------------------------------
            # Enhanced tail risk protection — validate signal
            # ----------------------------------------------------------
            if self.tail_risk_control_enabled:
                try:
                    recent_performance = [trade.get('pnl', 0.0) for trade in
                                        self.performance_metrics.get('recent_trades', [])[-10:]]

                    validation_result = self.tail_risk_protector.validate_trade_signal(
                        signal_confidence=confidence,
                        market_conditions=market_conditions,
                        recent_performance=recent_performance
                    )

                    if not validation_result['approved']:
                        rejection_reasons = '; '.join(validation_result['rejection_reasons'])
                        self.logger.debug(
                            f"Tail risk protection rejected {symbol} {action} signal: {rejection_reasons}"
                        )
                        if not self.immediate_trade_mode:
                            return None
                    risk_multiplier *= float(validation_result.get('position_size_multiplier', 1.0) or 1.0)

                except Exception as e:
                    self.logger.warning(f"Tail risk validation failed for {symbol}: {e}")

            # ----------------------------------------------------------
            # UNIFIED position sizing pipeline — single multiplier, hard-capped
            # ----------------------------------------------------------
            # risk_multiplier already includes symbol profile + tail risk
            effective_mult = risk_multiplier
            effective_mult *= min(pattern_size_mult, 1.25)       # cap pattern bonus

            # Correlation-based size reduction
            _, corr_factor = self._check_correlation_exposure(symbol, action)
            effective_mult *= corr_factor

            # Progressive daily loss reduction
            effective_mult *= self._daily_loss_size_factor

            # HARD CAP — no compound stacking beyond 1.5x
            effective_mult = min(effective_mult, 1.5)

            self.logger.debug(
                f"{symbol} sizing pipeline: risk_mult={risk_multiplier:.2f}, "
                f"pat_mult={pattern_size_mult:.2f}→{min(pattern_size_mult, 1.25):.2f}, "
                f"corr={corr_factor:.2f}, daily_loss={self._daily_loss_size_factor:.2f}, "
                f"effective={effective_mult:.2f}"
            )

            position_size = self._calculate_position_size(
                symbol,
                entry_price,
                stop_loss,
                symbol_info,
                risk_multiplier=effective_mult,
            )

            if position_size <= 0:
                return None

            # ----------------------------------------------------------
            # Build reason string
            # ----------------------------------------------------------
            if pattern_source:
                reason = (
                    f"{pattern_source} | {action} "
                    f"({confidence:.1%} confidence)"
                )
            elif pattern_confirmed:
                reason = (
                    f"Neural + M15 candle confirmation: {action} "
                    f"({confidence:.1%} confidence, threshold {required_confidence:.1%})"
                    + (f" | {pattern_description}" if pattern_description else "")
                )
            else:
                reason = (
                    f"Neural prediction: {action} "
                    f"({confidence:.1%} confidence, threshold {required_confidence:.1%})"
                    + (f" | {pattern_description}" if pattern_description else "")
                )

            # Create signal
            signal = TradingSignal(
                symbol=symbol,
                action=action,
                confidence=confidence,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                position_size=position_size,
                reason=reason,
            )

            return signal
            
        except Exception as e:
            self.logger.error(f"Error generating signal for {symbol}: {e}")
            return None

    def _generate_startup_fallback_signal(
        self,
        symbol: str,
        symbol_info: Dict[str, Any],
        market_data: Dict[str, Any]
    ) -> Optional[TradingSignal]:
        """
        Generate conservative fallback entries only when explicitly enabled.
        Fallback is restricted to historical MTF edge-validated setups.
        """
        try:
            if not self.immediate_trade_mode:
                return None
            if symbol in self._startup_trade_done:
                return None
            if not self._is_symbol_live_enabled(symbol):
                return None

            # Conservative fallback path: only historical MTF edge-validated entries.
            historical_signal = self._generate_historical_mtf_signal(symbol, symbol_info, market_data)
            if historical_signal is not None:
                return historical_signal

            return None
        except Exception as e:
            self.logger.error(f"Error generating startup fallback signal for {symbol}: {e}")
            return None
    
    def _get_market_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get comprehensive market data for a symbol.

        Core timeframes (M5, M15, H1) are mandatory — if any is missing the
        method returns None.  Higher timeframes (H4, D1) are optional and used
        only for chart-pattern detection; their absence is tolerated.
        """
        CORE_TIMEFRAMES = {"M5", "M15", "H1"}
        try:
            market_data = {}

            for tf_name, tf_constant in self.timeframes.items():
                rates = self.mt5_connector.get_rates(symbol, tf_constant, 0, 100)
                if rates:
                    market_data[tf_name] = rates
                else:
                    if tf_name in CORE_TIMEFRAMES:
                        self.logger.warning(f"No {tf_name} data for {symbol}")
                        return None
                    else:
                        # H4 / D1 missing is non-fatal — pattern detection
                        # will simply skip that timeframe.
                        self.logger.debug(f"No {tf_name} data for {symbol} (optional)")

            # Get symbol info
            symbol_info = self.mt5_connector.get_symbol_info(symbol)
            if symbol_info:
                market_data['symbol_info'] = symbol_info

            return market_data

        except Exception as e:
            self.logger.error(f"Error getting market data for {symbol}: {e}")
            return None
    
    def _prepare_ohlc_dataframe(self, rates: List[Dict[str, Any]]) -> Optional[pd.DataFrame]:
        """Normalize MT5 rate dictionaries into a sorted OHLCV dataframe."""
        try:
            if not rates:
                return None
            df = pd.DataFrame(rates).copy()
            if df.empty:
                return None
            if 'time' not in df.columns:
                return None
            df['time'] = pd.to_datetime(df['time'])
            df.sort_values('time', inplace=True)
            df.drop_duplicates(subset='time', keep='last', inplace=True)
            df.set_index('time', inplace=True)
            for col in ('open', 'high', 'low', 'close', 'tick_volume'):
                if col not in df.columns:
                    if col == 'tick_volume':
                        df[col] = 0.0
                    else:
                        return None
            return df
        except Exception:
            return None

    def _get_recent_candle_pattern_signal(
        self,
        market_data: Dict[str, Any],
        timeframe: str = 'M15',
        lookback: int = 8,
    ) -> Tuple[str, str, int]:
        """
        Derive candle-pattern action from recent closed candles.

        Rule implemented:
        - 4+ candles same direction -> continuation
        - exactly 3 candles same direction -> reversal
        """
        try:
            rates = market_data.get(timeframe) or []
            if len(rates) < 6:
                return 'HOLD', '', 0

            # Exclude current forming candle; use only closed candles.
            closed = rates[:-1] if len(rates) > 1 else rates
            if len(closed) < 4:
                return 'HOLD', '', 0

            recent = closed[-max(4, int(lookback)):]
            directions: List[int] = []
            for candle in recent:
                open_price = float(candle.get('open', 0.0) or 0.0)
                close_price = float(candle.get('close', 0.0) or 0.0)
                if close_price > open_price:
                    directions.append(1)
                elif close_price < open_price:
                    directions.append(-1)
                else:
                    directions.append(0)

            if not directions or directions[-1] == 0:
                return 'HOLD', '', 0

            last_dir = directions[-1]
            streak = 1
            for idx in range(len(directions) - 2, -1, -1):
                if directions[idx] == last_dir:
                    streak += 1
                else:
                    break

            if streak >= 4:
                action = 'BUY' if last_dir > 0 else 'SELL'
                return action, 'continuation_4plus', int(streak)
            if streak == 3:
                action = 'SELL' if last_dir > 0 else 'BUY'
                return action, 'reversal_3', int(streak)
            return 'HOLD', '', int(streak)
        except Exception:
            return 'HOLD', '', 0

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0.0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
        rs = gain / (loss + 1e-12)
        return 100 - (100 / (1 + rs))

    def _calculate_atr_series(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate ATR series from OHLC dataframe."""
        prev_close = df['close'].shift(1)
        tr = pd.concat(
            [
                (df['high'] - df['low']).abs(),
                (df['high'] - prev_close).abs(),
                (df['low'] - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        return tr.rolling(period).mean()

    def _enrich_indicator_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add trainer-aligned indicators for runtime features and MTF logic."""
        out = df.copy()
        out['sma_5'] = out['close'].rolling(5).mean()
        out['sma_20'] = out['close'].rolling(20).mean()
        out['sma_50'] = out['close'].rolling(50).mean()
        out['ema_12'] = out['close'].ewm(span=12, adjust=False).mean()
        out['ema_26'] = out['close'].ewm(span=26, adjust=False).mean()
        out['rsi'] = self._calculate_rsi(out['close'])
        out['returns'] = out['close'].pct_change()
        out['volatility'] = out['returns'].rolling(20).std()
        out['atr_14'] = self._calculate_atr_series(out, period=14)
        bb_std = out['close'].rolling(20).std()
        out['bb_upper'] = out['sma_20'] + 2.0 * bb_std
        out['bb_lower'] = out['sma_20'] - 2.0 * bb_std
        out['volume_z'] = (
            (out['tick_volume'] - out['tick_volume'].rolling(20).mean())
            / (out['tick_volume'].rolling(20).std() + 1e-8)
        )
        candle_range = (out['high'] - out['low']).replace(0, np.nan)
        out['body_ratio'] = (out['close'] - out['open']) / (candle_range + 1e-8)
        return out.dropna()

    def _compute_timeframe_vote(self, row: pd.Series) -> int:
        """Compute simple directional vote from a timeframe indicator row."""
        vote = 0
        ema_fast = float(row.get('ema_12', 0.0))
        ema_slow = float(row.get('ema_26', 0.0))
        close = float(row.get('close', 0.0))
        sma_20 = float(row.get('sma_20', 0.0))
        rsi = float(row.get('rsi', 50.0))

        if ema_fast > ema_slow:
            vote += 1
        elif ema_fast < ema_slow:
            vote -= 1

        if close > sma_20:
            vote += 1
        elif close < sma_20:
            vote -= 1

        if rsi >= 55:
            vote += 1
        elif rsi <= 45:
            vote -= 1

        return vote

    def _get_current_mtf_action(self, market_data: Dict[str, Any]) -> Tuple[str, int, Dict[str, int]]:
        """Derive current multi-timeframe directional action from M5/M15/H1."""
        try:
            frames: Dict[str, pd.DataFrame] = {}
            for tf in ('M5', 'M15', 'H1'):
                rates = market_data.get(tf) or []
                frame = self._prepare_ohlc_dataframe(rates)
                if frame is None or len(frame) < 30:
                    return 'HOLD', 0, {}
                enriched = self._enrich_indicator_frame(frame)
                if enriched.empty:
                    return 'HOLD', 0, {}
                frames[tf] = enriched

            votes = {
                'M5': self._compute_timeframe_vote(frames['M5'].iloc[-1]),
                'M15': self._compute_timeframe_vote(frames['M15'].iloc[-1]),
                'H1': self._compute_timeframe_vote(frames['H1'].iloc[-1]),
            }
            total_vote = int(votes['M5'] + votes['M15'] + votes['H1'])

            # Require at least medium alignment and avoid H1 against the direction.
            if total_vote >= 3 and votes['M15'] > 0 and votes['H1'] >= 0:
                return 'BUY', total_vote, votes
            if total_vote <= -3 and votes['M15'] < 0 and votes['H1'] <= 0:
                return 'SELL', total_vote, votes
            return 'HOLD', total_vote, votes
        except Exception:
            return 'HOLD', 0, {}

    def _generate_historical_mtf_signal(
        self,
        symbol: str,
        symbol_info: Dict[str, Any],
        market_data: Dict[str, Any]
    ) -> Optional[TradingSignal]:
        """
        Generate startup signal from historical M15 pattern quality plus current MTF alignment.
        """
        try:
            mtf_action, mtf_score, votes = self._get_current_mtf_action(market_data)
            if mtf_action not in ('BUY', 'SELL'):
                return None

            m15_rates = market_data.get('M15') or []
            m15_frame = self._prepare_ohlc_dataframe(m15_rates)
            if m15_frame is None or len(m15_frame) < 80:
                return None
            m15 = self._enrich_indicator_frame(m15_frame)
            if len(m15) < 60:
                return None

            horizon = max(4, int(self.historical_signal_horizon))
            wins = 0
            trades = 0
            edge_sum = 0.0
            point = float(symbol_info.get("point", 0.0001) or 0.0001)
            live_spread_points = float(symbol_info.get("spread", 0.0) or 0.0)

            # Backtest the same M15-pattern direction logic on recent history.
            start_idx = 25
            end_idx = len(m15) - horizon
            for i in range(start_idx, end_idx):
                row = m15.iloc[i]
                vote = self._compute_timeframe_vote(row)

                if mtf_action == 'BUY' and vote < 2:
                    continue
                if mtf_action == 'SELL' and vote > -2:
                    continue

                entry = float(m15['close'].iloc[i])
                exit_price = float(m15['close'].iloc[i + horizon])
                future_return = (exit_price / (entry + 1e-12)) - 1.0

                pnl_directional = future_return if mtf_action == 'BUY' else -future_return
                spread_cost_ratio = (live_spread_points * point) / (entry + 1e-12)
                net_directional = pnl_directional - (spread_cost_ratio * 1.1)
                trades += 1
                edge_sum += net_directional
                if net_directional > 0:
                    wins += 1

            if trades < self.historical_min_samples:
                return None

            win_rate = wins / trades
            avg_edge = edge_sum / trades

            quality = self._get_model_symbol_quality(symbol)
            min_win_rate = 0.52
            min_avg_edge = 0.0
            if quality:
                quality_win_rate, quality_samples = quality
                if (
                    quality_win_rate < self.minimum_symbol_quality_winrate
                    or quality_samples < self.minimum_symbol_quality_samples
                ):
                    min_win_rate = 0.56
                    min_avg_edge = 0.00005

            # Require positive historical edge before forcing an immediate startup trade.
            if win_rate < min_win_rate or avg_edge <= min_avg_edge:
                return None

            startup_threshold = self._resolve_symbol_trade_threshold(symbol)
            confidence = float(
                np.clip(
                    0.56 + (win_rate - 0.5) * 1.4 + np.clip(avg_edge * 100.0, -0.04, 0.10),
                    max(startup_threshold + 0.01, 0.60),
                    0.92,
                )
            )

            entry_price = symbol_info['ask'] if mtf_action == 'BUY' else symbol_info['bid']
            stop_loss, take_profit = self._calculate_sl_tp(
                symbol, mtf_action, entry_price, symbol_info, market_data
            )
            position_size = self._calculate_position_size(
                symbol,
                entry_price,
                stop_loss,
                symbol_info,
                risk_multiplier=self._get_symbol_risk_multiplier(symbol),
            )
            if position_size <= 0:
                return None

            signal = TradingSignal(
                symbol=symbol,
                action=mtf_action,
                confidence=confidence,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                position_size=position_size,
                reason=(
                    f"Historical MTF startup entry ({mtf_action}, "
                    f"win={win_rate:.1%}, samples={trades}, votes={votes}, score={mtf_score})"
                ),
            )
            return signal
        except Exception as e:
            self.logger.error(f"Error building historical MTF startup signal for {symbol}: {e}")
            return None

    def _compute_zeropoint_tf_features(self, ohlcv_data) -> np.ndarray:
        """
        Compute 5 ZeroPoint features from one timeframe's OHLCV data.

        Args:
            ohlcv_data: Raw MT5 rates (list of dicts) or pandas DataFrame

        Returns:
            np.ndarray of shape (5,) with ZeroPoint features, or zeros on error.
        """
        zero = np.zeros(ZEROPOINT_FEATURES_PER_TF, dtype=np.float32)
        try:
            if ohlcv_data is None:
                return zero

            # Convert to DataFrame if needed
            if isinstance(ohlcv_data, pd.DataFrame):
                df = ohlcv_data
            else:
                df = self._prepare_ohlc_dataframe(ohlcv_data)

            if df is None or len(df) < 15:
                return zero

            # Compute ZeroPoint state
            zp_df = compute_zeropoint_state(df)
            if zp_df is None or len(zp_df) == 0:
                return zero

            # Extract features from the last bar
            return extract_zeropoint_bar_features(zp_df.iloc[-1])
        except Exception as e:
            self.logger.debug(f"ZeroPoint TF feature error: {e}")
            return zero

    def _extract_features(self, symbol: str, market_data: Dict[str, Any]) -> Optional[np.ndarray]:
        """
        Extract runtime features aligned with trainer feature schema.
        Base feature order mirrors `simple_neural_trainer.py`.

        Backwards-compatible: If model has zeropoint_feature_count > 0 in metadata,
        15 ZeroPoint features (5 per TF x M15/H1/H4) are inserted between
        forming pattern features and symbol one-hot. Otherwise, same 47-dim vector.
        """
        try:
            if 'M15' not in market_data:
                return None

            m15_data = market_data['M15']
            if len(m15_data) < 80:
                return None

            df_raw = self._prepare_ohlc_dataframe(m15_data)
            if df_raw is None:
                return None
            df = self._enrich_indicator_frame(df_raw)
            if len(df) < 25:
                return None

            window = df.iloc[-20:]
            current = df.iloc[-1]
            prev = df.iloc[-2]

            current_price = float(current['close'])
            prev_price = float(prev['close'])
            price_std = float(window['close'].std())
            returns_std = float(window['returns'].std())

            if not np.isfinite(price_std) or price_std < 1e-12:
                price_std = 1e-8
            if not np.isfinite(returns_std) or returns_std < 1e-12:
                returns_std = 1e-8

            close_3 = float(df.iloc[-4]['close']) if len(df) >= 4 else prev_price
            close_6 = float(df.iloc[-7]['close']) if len(df) >= 7 else prev_price
            close_12 = float(df.iloc[-13]['close']) if len(df) >= 13 else prev_price

            atr_14 = float(current['atr_14']) if np.isfinite(current['atr_14']) else 0.0
            bb_upper = float(current['bb_upper']) if np.isfinite(current['bb_upper']) else current_price
            bb_lower = float(current['bb_lower']) if np.isfinite(current['bb_lower']) else current_price
            bb_width = max(bb_upper - bb_lower, 1e-8)
            bb_pos = (current_price - bb_lower) / bb_width
            volume_z = float(current['volume_z']) if np.isfinite(current['volume_z']) else 0.0
            body_ratio = float(current['body_ratio']) if np.isfinite(current['body_ratio']) else 0.0

            # Spread features (indices 16-17, matching trainer).
            symbol_info = self.mt5_connector.get_symbol_info(symbol) if hasattr(self, 'mt5_connector') else None
            symbol_point = float(getattr(symbol_info, 'point', 0.0001)) if symbol_info else 0.0001
            spread_points = float(getattr(symbol_info, 'spread', 0.0) if symbol_info else 0.0)
            spread_cost = float(np.clip((spread_points * symbol_point) / (current_price + 1e-12), 0.0, 0.02))
            atr_ratio = atr_14 / (current_price + 1e-12)
            spread_pressure = spread_cost / (atr_ratio + 1e-12)

            base_features = np.array(
                [
                    (current_price - prev_price) / (prev_price + 1e-12),
                    (current_price - float(window['close'].mean())) / (price_std + 1e-12),
                    float(current['sma_5']) / (current_price + 1e-12) - 1.0,
                    float(current['sma_20']) / (current_price + 1e-12) - 1.0,
                    float(current['rsi']) / 100.0,
                    returns_std * 100.0,
                    (current_price - close_3) / (close_3 + 1e-12),
                    (current_price - close_6) / (close_6 + 1e-12),
                    (current_price - close_12) / (close_12 + 1e-12),
                    float(current['sma_50']) / (current_price + 1e-12) - 1.0,
                    float(current['ema_12']) / (current_price + 1e-12) - 1.0,
                    float(current['ema_26']) / (current_price + 1e-12) - 1.0,
                    atr_14 / (current_price + 1e-12),
                    bb_pos,
                    volume_z,
                    body_ratio,
                    spread_cost,
                    spread_pressure,
                ],
                dtype=np.float32,
            )

            # Push structure features (8 features from swing analysis).
            push_profile = self.model_manager.get_push_profile(symbol)
            if push_profile is None:
                push_profile = SymbolPushProfile.default(symbol)
            push_window = max(60, len(df))
            pw_start = max(0, len(df) - push_window)
            pw_highs = df['high'].astype(float).values[pw_start:]
            pw_lows = df['low'].astype(float).values[pw_start:]
            pw_closes = df['close'].astype(float).values[pw_start:]
            direction = infer_direction_from_closes(pw_closes, lookback=10)
            push_features = PushFeatureExtractor.extract_push_features(
                highs=pw_highs,
                lows=pw_lows,
                closes=pw_closes,
                profile=push_profile,
                point=symbol_point,
                direction=direction,
            )

            # Forming pattern features (12 features — puzzle-piece detection).
            # Scan ALL timeframes (M15, H1, H4, D1) for forming patterns and
            # pick the strongest one.  This makes the neural model multi-TF aware
            # so it can decide to trade immediately when it sees a strong forming
            # structure on any timeframe.
            # TIME-AWARE: Only feed patterns likely to resolve within our ~24hr
            # profit window.  Higher TFs need higher completion to qualify.
            try:
                MTF_PAT_CONFIG = {
                    "M15": {"weight": 0.85, "min_completion": 0.40},
                    "H1":  {"weight": 1.0,  "min_completion": 0.55},
                    "H4":  {"weight": 1.15, "min_completion": 0.75},
                    "D1":  {"weight": 1.3,  "min_completion": 0.90},
                }
                best_forming_score = -1.0
                best_forming_feats = np.zeros(FORMING_PATTERN_FEATURE_COUNT, dtype=np.float32)

                for pat_tf, cfg in MTF_PAT_CONFIG.items():
                    try:
                        if pat_tf == "M15":
                            # Already have enriched M15 df
                            pat_df = df
                        else:
                            pat_rates = market_data.get(pat_tf)
                            if not pat_rates:
                                continue
                            pat_df_raw = self._prepare_ohlc_dataframe(pat_rates)
                            if pat_df_raw is None or len(pat_df_raw) < 30:
                                continue
                            pat_df = pat_df_raw  # No need to enrich; PatternRecognizer uses OHLCV

                        pat_window = min(80, len(pat_df))
                        pat_slice = pat_df.iloc[-pat_window:].copy()
                        if len(pat_slice) < 25:
                            continue

                        recognizer = PatternRecognizer(pat_slice)
                        forming = recognizer.detect_forming_patterns()
                        if not forming:
                            continue

                        min_comp = cfg["min_completion"]
                        tf_weight = cfg["weight"]

                        # Filter: only keep patterns that meet time-aware completion threshold
                        time_filtered = [fp for fp in forming if fp.completion_pct >= min_comp]
                        if not time_filtered:
                            continue

                        # Score: best pattern's confidence * completion * TF weight + time bonus
                        for fp in time_filtered:
                            time_bonus = (fp.completion_pct - min_comp) / (1.0 - min_comp + 1e-8)
                            fp_score = fp.confidence * fp.completion_pct * tf_weight * (1.0 + 0.3 * time_bonus)
                            if fp_score > best_forming_score:
                                best_forming_score = fp_score
                                best_forming_feats = PatternRecognizer.forming_pattern_features(time_filtered)
                    except Exception:
                        continue

                pattern_features = best_forming_feats
            except Exception:
                pattern_features = np.zeros(FORMING_PATTERN_FEATURE_COUNT, dtype=np.float32)

            symbol_features = self.model_manager.get_symbol_features(symbol)

            # Check if model expects ZeroPoint features (backwards-compatible)
            zp_feature_count = 0
            if hasattr(self, 'model_manager') and self.model_manager is not None:
                model_meta = getattr(self.model_manager, 'metadata', None) or {}
                zp_feature_count = int(model_meta.get("zeropoint_feature_count", 0))

            if zp_feature_count > 0:
                # Model was trained with ZeroPoint features — compute 15 ZP features
                # (5 per timeframe x M15/H1/H4)
                zp_m15_feats = self._compute_zeropoint_tf_features(df_raw)
                zp_h1_feats = self._compute_zeropoint_tf_features(market_data.get('H1'))
                zp_h4_feats = self._compute_zeropoint_tf_features(market_data.get('H4'))

                features = np.concatenate([
                    base_features,       # 18
                    push_features,       # 8
                    pattern_features,    # 12
                    zp_m15_feats,        # 5
                    zp_h1_feats,         # 5
                    zp_h4_feats,         # 5
                    symbol_features,     # 9
                ]).astype(np.float32)
            else:
                # Old model (47 dims) — no ZeroPoint features
                features = np.concatenate([
                    base_features,
                    push_features,
                    pattern_features,
                    symbol_features,
                ]).astype(np.float32)

            return features
        except Exception as e:
            self.logger.error(f"Error extracting features: {e}")
            return None
    
    def _estimate_atr(self, rates: List[Dict[str, Any]], period: int = 14) -> Optional[float]:
        """Estimate ATR from rate dictionaries."""
        try:
            if not rates or len(rates) < period + 1:
                return None

            highs = np.array([float(r['high']) for r in rates], dtype=float)
            lows = np.array([float(r['low']) for r in rates], dtype=float)
            closes = np.array([float(r['close']) for r in rates], dtype=float)

            tr_values = []
            start_idx = max(1, len(rates) - period)
            for i in range(start_idx, len(rates)):
                high_low = highs[i] - lows[i]
                high_close = abs(highs[i] - closes[i - 1])
                low_close = abs(lows[i] - closes[i - 1])
                tr_values.append(max(high_low, high_close, low_close))

            if not tr_values:
                return None
            return float(np.mean(tr_values))
        except Exception:
            return None

    def _calculate_sl_tp(
        self,
        symbol: str,
        action: str,
        entry_price: float,
        symbol_info: Dict[str, Any],
        market_data: Optional[Dict[str, Any]] = None
    ) -> Tuple[float, float]:
        """Calculate stop loss and take profit levels for FX and non-FX symbols."""
        try:
            digits = int(symbol_info.get('digits', 5))
            point = float(symbol_info.get('point', 0.00001) or 0.00001)
            spread_points = float(symbol_info.get('spread', 0.0) or 0.0)
            spread_price = spread_points * point

            atr = None
            if market_data and 'M15' in market_data:
                atr = self._estimate_atr(market_data['M15'], period=14)

            # ATR-based distances are symbol-agnostic (works for BTCUSD too).
            if atr and atr > 0:
                base_distance = max(atr, spread_price * 4.0, point * 20.0)
            else:
                base_distance = max(spread_price * 30.0, point * 20.0, entry_price * 0.0005)

            sl_distance = base_distance * 1.2
            tp_distance = base_distance * 2.2

            if action == 'BUY':
                stop_loss = entry_price - sl_distance
                take_profit = entry_price + tp_distance
            else:  # SELL
                stop_loss = entry_price + sl_distance
                take_profit = entry_price - tp_distance

            return round(stop_loss, digits), round(take_profit, digits)

        except Exception as e:
            self.logger.error(f"Error calculating SL/TP for {symbol}: {e}")
            return 0.0, 0.0

    # ------------------------------------------------------------------ #
    # Chart pattern detection + confluence                                #
    # ------------------------------------------------------------------ #

    def _detect_pattern_signal(self, symbol: str, market_data: Dict[str, Any]) -> Optional[Tuple[Pattern, str, float]]:
        """Scan H1/H4/D1 for fresh chart patterns.

        Returns ``(best_pattern, timeframe_key, pattern_score)`` or None.
        """
        TF_WEIGHTS = {"H1": 1.0, "H4": 1.15, "D1": 1.3}
        # Max pattern age varies by timeframe (higher TF patterns take longer
        # to play out their measured moves).
        TF_MAX_AGE = {"H1": 5, "H4": 10, "D1": 20}
        best: Optional[Tuple[Pattern, str, float]] = None

        for tf_key, tf_weight in TF_WEIGHTS.items():
            rates = market_data.get(tf_key)
            if not rates:
                continue
            df = self._prepare_ohlc_dataframe(rates)
            if df is None or len(df) < 45:
                continue
            max_age = TF_MAX_AGE.get(tf_key, 10)
            try:
                recognizer = PatternRecognizer(df)
                patterns = recognizer.detect_all(max_age=max_age)
            except Exception as e:
                self.logger.debug(f"Pattern detection error on {symbol} {tf_key}: {e}")
                continue

            for pat in patterns:
                # Must have a valid target_price (should always be true
                # after _validate_and_filter but guard anyway)
                if not pat.details.get("target_price"):
                    continue
                # Score = confidence * volume boost * timeframe weight
                score = pat.confidence * (1.0 + 0.2 * pat.volume_score) * tf_weight
                if best is None or score > best[2]:
                    best = (pat, tf_key, score)

        if best is not None:
            pat, tf, sc = best
            self.logger.info(
                f"Pattern detected for {symbol}: {pat.name} on {tf} "
                f"(conf={pat.confidence:.2f}, score={sc:.3f}, dir={pat.direction})"
            )
        return best

    def _detect_forming_pattern_signal(
        self, symbol: str, market_data: Dict[str, Any],
    ) -> Optional[Tuple[FormingPattern, str, float]]:
        """Scan H1/H4/D1 for FORMING (incomplete) patterns — the 'puzzle pieces'.

        TIME-AWARE: Only considers patterns likely to resolve within our
        profit window (~24 hours).  Higher TF patterns need higher completion
        to qualify because they take longer to play out:
          - M15: >=40% complete (resolves in hours)
          - H1:  >=55% complete (resolves in hours–1 day)
          - H4:  >=75% complete (resolves in 1–3 days, need near-apex)
          - D1:  >=90% complete (resolves in days–weeks, only if imminent)

        Returns ``(best_forming, timeframe_key, score)`` or None.
        """
        # Timeframe weights + minimum completion to be time-actionable.
        # Higher TF patterns are more powerful BUT only if they'll resolve
        # within our ~24hr profit window.
        TF_CONFIG = {
            "M15": {"weight": 0.85, "min_completion": 0.40},
            "H1":  {"weight": 1.0,  "min_completion": 0.55},
            "H4":  {"weight": 1.15, "min_completion": 0.75},
            "D1":  {"weight": 1.3,  "min_completion": 0.90},
        }
        best: Optional[Tuple[FormingPattern, str, float]] = None

        for tf_key, cfg in TF_CONFIG.items():
            rates = market_data.get(tf_key)
            if not rates:
                continue
            df = self._prepare_ohlc_dataframe(rates)
            if df is None or len(df) < 30:
                continue
            try:
                recognizer = PatternRecognizer(df)
                forming = recognizer.detect_forming_patterns()
            except Exception as e:
                self.logger.debug(f"Forming pattern error on {symbol} {tf_key}: {e}")
                continue

            min_comp = cfg["min_completion"]
            tf_weight = cfg["weight"]

            for fp in forming:
                # TIME GATE: pattern must be complete enough for this TF
                # to resolve within our profit window
                if fp.completion_pct < min_comp:
                    continue
                # Must have valid R:R
                entry_est = fp.breakout_level
                risk = abs(entry_est - fp.stop_loss)
                reward = abs(fp.target_price - entry_est)
                if risk <= 0 or reward / risk < 1.5:
                    continue

                # Time-adjusted score: bonus for patterns near completion
                # on faster timeframes (more likely to resolve in our window)
                time_bonus = (fp.completion_pct - min_comp) / (1.0 - min_comp + 1e-8)
                score = fp.confidence * fp.completion_pct * tf_weight * (1.0 + 0.3 * time_bonus)
                if best is None or score > best[2]:
                    best = (fp, tf_key, score)

        if best is not None:
            fp, tf, sc = best
            self.logger.info(
                f"FORMING pattern for {symbol}: {fp.name} on {tf} "
                f"(completion={fp.completion_pct:.0%}, conf={fp.confidence:.2f}, "
                f"score={sc:.3f}, dir={fp.predicted_direction})"
            )
        return best

    def _calculate_position_size(self, symbol: str, entry_price: float, stop_loss: float,
                               symbol_info: Dict[str, Any], risk_multiplier: float = 1.0) -> float:
        """Calculate position size based on risk management"""
        try:
            # Get account info
            account_info = self.mt5_connector.get_account_info()
            if not account_info:
                return 0.0
            
            balance = float(account_info['balance'])
            risk_scale = float(np.clip(risk_multiplier, 0.0, 2.0))
            if risk_scale <= 0.0:
                return 0.0
            risk_amount = balance * self.risk_per_trade * risk_scale

            tick_size = float(symbol_info.get('trade_tick_size', 0.0) or 0.0)
            point = float(symbol_info.get('point', 0.0) or 0.0)
            if tick_size <= 0:
                tick_size = point if point > 0 else 1e-6

            tick_value = float(symbol_info.get('trade_tick_value', 0.0) or 0.0)
            if tick_value <= 0:
                contract_size = float(symbol_info.get('trade_contract_size', 1.0) or 1.0)
                tick_value = max(contract_size * tick_size, 1e-6)

            sl_distance = abs(entry_price - stop_loss)
            sl_ticks = sl_distance / tick_size

            if sl_ticks <= 0:
                return 0.0

            loss_per_lot = sl_ticks * tick_value
            if loss_per_lot <= 0:
                return 0.0

            # Calculate position size using symbol tick economics.
            position_size = risk_amount / loss_per_lot
            
            # Apply symbol constraints
            volume_min = float(symbol_info.get('volume_min', 0.01) or 0.01)
            volume_max = float(symbol_info.get('volume_max', 100.0) or 100.0)
            volume_step = float(symbol_info.get('volume_step', 0.01) or 0.01)
            
            # Round to step size
            position_size = round(position_size / volume_step) * volume_step
            
            # Ensure within limits
            position_size = max(volume_min, min(volume_max, position_size))

            # Global balance-based lot cap — prevents oversized positions
            # on small accounts regardless of SL tightness
            global_max = self._get_global_max_lot(balance)
            if position_size > global_max:
                self.logger.info(
                    f"Global lot cap: {position_size:.2f} -> {global_max:.2f} "
                    f"(balance=${balance:.0f})"
                )
                position_size = global_max

            return float(round(position_size, 2))
            
        except Exception as e:
            self.logger.error(f"Error calculating position size for {symbol}: {e}")
            return 0.0

    def _is_spread_acceptable(self, symbol_info: Dict[str, Any]) -> bool:
        """
        Validate quote quality before signal generation/execution.
        Uses tighter limits for FX and percentage limits for crypto/CFDs.
        """
        try:
            symbol = str(
                symbol_info.get('requested_name')
                or symbol_info.get('name')
                or ''
            ).upper()
            bid = float(symbol_info.get('bid', 0.0) or 0.0)
            ask = float(symbol_info.get('ask', 0.0) or 0.0)

            # If broker quote is not live, skip.
            if bid <= 0 or ask <= 0 or ask < bid:
                return False

            point = float(symbol_info.get('point', 0.0) or 0.0)
            spread_points = float(symbol_info.get('spread', 0.0) or 0.0)
            spread_price = spread_points * point if point > 0 else 0.0
            if spread_price <= 0:
                spread_price = ask - bid
            if spread_price <= 0:
                return False

            mid = (ask + bid) / 2.0
            if mid <= 0:
                return False
            spread_pct = spread_price / mid

            normalized = ''.join(ch for ch in symbol if ch.isalnum())
            fx_ccy = {"USD", "EUR", "JPY", "GBP", "AUD", "NZD", "CAD", "CHF"}
            is_crypto = any(token in normalized for token in ('BTC', 'ETH', 'XRP', 'LTC', 'SOL'))
            is_forex = (
                not is_crypto
                and len(normalized) >= 6
                and normalized[:3].isalpha()
                and normalized[3:6].isalpha()
                and normalized[:3] in fx_ccy
                and normalized[3:6] in fx_ccy
            )

            if is_forex:
                quote_ccy = normalized[3:6]
                pip_size = 0.01 if quote_ccy == 'JPY' else 0.0001
                spread_pips = spread_price / pip_size if pip_size > 0 else 999.0
                max_pips = 4.0 if quote_ccy == 'JPY' else 3.0
                if spread_pips > max_pips:
                    return False
                if spread_pct > 0.00035:
                    return False
                return True

            # Crypto/CFDs/metals: use percentage limits.
            if is_crypto:
                return spread_pct <= 0.0025
            return spread_pct <= 0.0015
        except Exception as e:
            self.logger.debug(f"Spread validation failed: {e}")
            return False

    def _get_model_symbol_quality(self, symbol: str) -> Optional[Tuple[float, int]]:
        """Return (directional_win_rate, directional_trade_count) from model metadata."""
        try:
            metadata = getattr(self.model_manager, 'model_metadata', {}) or {}
            if not isinstance(metadata, dict):
                return None

            symbol_validation = metadata.get('symbol_validation', {})
            if not isinstance(symbol_validation, dict) or not symbol_validation:
                return None

            normalized = re.sub(r"[^A-Z0-9]", "", str(symbol or "").upper())
            aliases = []
            if normalized:
                aliases.append(normalized)
                if normalized.endswith("USD") and len(normalized) > 3:
                    aliases.append(normalized[:-3])

            for alias in aliases:
                stats = symbol_validation.get(alias)
                if not isinstance(stats, dict):
                    continue
                win_rate = float(
                    stats.get('directional_win_rate', stats.get('win_rate', 0.0)) or 0.0
                )
                trade_count = int(
                    round(float(stats.get('directional_trade_count', stats.get('trade_count', 0.0)) or 0.0))
                )
                return win_rate, trade_count
            return None
        except Exception:
            return None

    def _get_model_symbol_profitability(self, symbol: str) -> Optional[Tuple[float, float, int]]:
        """Return (avg_trade_return, profit_factor, trade_count) from model metadata."""
        try:
            metadata = getattr(self.model_manager, 'model_metadata', {}) or {}
            if not isinstance(metadata, dict):
                return None

            profitability = metadata.get('symbol_profitability_validation', {})
            if not isinstance(profitability, dict) or not profitability:
                return None

            normalized = re.sub(r"[^A-Z0-9]", "", str(symbol or "").upper())
            aliases = []
            if normalized:
                aliases.append(normalized)
                if normalized.endswith("USD") and len(normalized) > 3:
                    aliases.append(normalized[:-3])

            for alias in aliases:
                stats = profitability.get(alias)
                if not isinstance(stats, dict):
                    continue

                avg_trade_return = float(stats.get('avg_trade_return', 0.0) or 0.0)
                profit_factor = float(stats.get('profit_factor', 0.0) or 0.0)
                trade_count = int(
                    round(float(stats.get('trade_count', stats.get('trades', 0.0)) or 0.0))
                )
                return avg_trade_return, profit_factor, trade_count

            return None
        except Exception:
            return None
    
    def _process_signal(self, signal: TradingSignal):
        """Process a trading signal"""
        try:
            if signal.action not in ('BUY', 'SELL'):
                self.logger.info(
                    f"Skipping non-directional signal for {signal.symbol}: {signal.action}"
                )
                return

            # Check if we can take this trade
            if not self._can_trade(signal):
                return
            
            # Execute trade
            order_result = self._execute_trade(signal)
            if order_result:
                signal.executed = True
                signal.order_ticket = order_result.get('order')
                now = datetime.now()
                
                # Create position tracking
                position = Position(
                    ticket=order_result['order'],
                    symbol=signal.symbol,
                    action=signal.action,
                    entry_price=signal.entry_price,
                    stop_loss=signal.stop_loss,
                    take_profit=signal.take_profit,
                    position_size=signal.position_size
                )
                
                # Attach model context for agentic learning journal.
                position.model_confidence = float(signal.confidence or 0.0)
                position.model_action = str(signal.action or '')
                try:
                    _threshold = self._resolve_symbol_trade_threshold(signal.symbol)
                    position.symbol_threshold = float(_threshold)
                except Exception:
                    position.symbol_threshold = 0.0
                try:
                    _metadata = getattr(self.model_manager, 'model_metadata', {}) or {}
                    _action_modes = _metadata.get('symbol_action_modes', {}) or {}
                    position.action_mode = str(_action_modes.get(signal.symbol.upper(), 'normal'))
                    position.model_version = str(_metadata.get('training_date', ''))
                except Exception:
                    pass

                self.positions[position.ticket] = position
                self.signals_history.append(signal)
                self._symbol_last_entry_time[signal.symbol.upper()] = now
                self._new_trade_timestamps.append(now)
                if "startup" in str(signal.reason).lower():
                    self._startup_trade_done.add(signal.symbol)
                
                # Record trade for performance tracking
                try:
                    trade_data = {
                        'symbol': signal.symbol,
                        'action': signal.action,
                        'entry_price': signal.entry_price,
                        'position_size': signal.position_size,
                        'confidence': signal.confidence,
                        'reason': signal.reason,
                        'pnl': 0,  # Will be updated when trade closes
                        'timestamp': datetime.now()
                    }
                    self.performance_tracker.record_trade(trade_data)
                except Exception as e:
                    self.logger.warning(f"Failed to record trade for performance tracking: {e}")
                
                self.logger.info(f"Trade executed: {signal.symbol} {signal.action} @ {signal.entry_price}")
                self.logger.info(f"SL: {signal.stop_loss}, TP: {signal.take_profit}")
            
        except Exception as e:
            self.logger.error(f"Error processing signal: {e}")
    
    def _can_trade(self, signal: TradingSignal) -> bool:
        """Check if we can execute the trade"""
        if signal.action not in ('BUY', 'SELL'):
            return False

        self._refresh_account_risk_state()
        if self._is_account_risk_paused():
            return False

        now = datetime.now()
        symbol_key = signal.symbol.upper()
        loss_block_until = self._symbol_loss_block_until.get(symbol_key)
        if loss_block_until:
            if now < loss_block_until:
                return False
            self._symbol_loss_block_until.pop(symbol_key, None)

        blocked_until = self._symbol_trade_block_until.get(symbol_key)
        if blocked_until:
            if now < blocked_until:
                return False
            # Cooldown expired.
            self._symbol_trade_block_until.pop(symbol_key, None)
            self._symbol_market_closed_log_time.pop(symbol_key, None)

        last_entry = self._symbol_last_entry_time.get(symbol_key)
        if last_entry and (now - last_entry).total_seconds() < self.symbol_entry_cooldown_seconds:
            return False

        cutoff = now - timedelta(hours=1)
        self._new_trade_timestamps = [ts for ts in self._new_trade_timestamps if ts >= cutoff]
        if len(self._new_trade_timestamps) >= self.max_new_trades_per_hour:
            return False

        # Check maximum concurrent positions
        if len(self.positions) >= self.max_concurrent_positions:
            return False
        
        # Check if we already have a position in this symbol (internal tracker)
        for position in self.positions.values():
            if position.symbol == signal.symbol and position.status == 'OPEN':
                return False

        # SAFETY NET: also check MT5 directly for existing positions on this symbol.
        # This catches positions from previous sessions that may not be in our tracker.
        try:
            import MetaTrader5 as mt5_lib
            mt5_sym_positions = mt5_lib.positions_get(symbol=signal.symbol)
            if mt5_sym_positions and len(mt5_sym_positions) > 0:
                self.logger.info(
                    f"MT5 duplicate guard: {signal.symbol} already has "
                    f"{len(mt5_sym_positions)} open position(s) on MT5 - blocking new entry"
                )
                return False
        except Exception:
            pass  # If MT5 query fails, rely on internal tracker

        # ZeroPoint Pure Mode — bypass confidence threshold and correlation guard
        if self.zeropoint_pure_mode:
            return True

        # Check confidence using symbol-level threshold when available.
        reason_text = str(getattr(signal, 'reason', '') or '').lower()
        symbol_threshold = self._resolve_symbol_trade_threshold(signal.symbol)
        if self.profitability_first_mode or self.immediate_trade_mode:
            required_confidence = symbol_threshold
        else:
            required_confidence = max(self.confidence_threshold, symbol_threshold)

        # Startup/historical fallback entries are generated from stricter
        # profitability and MTF gates, so do not over-block them with UI threshold.
        if "startup" in reason_text or "historical mtf" in reason_text:
            required_confidence = min(required_confidence, max(symbol_threshold, 0.58))

        if signal.confidence < required_confidence:
            return False

        # Correlation exposure guard — block if too many same-direction correlated positions
        corr_allowed, _ = self._check_correlation_exposure(signal.symbol, signal.action)
        if not corr_allowed:
            self.logger.info(
                f"Correlation guard blocked {signal.symbol} {signal.action}: "
                f"too many same-direction correlated positions"
            )
            return False

        return True
    
    def _execute_trade(self, signal: TradingSignal) -> Optional[Dict[str, Any]]:
        """Execute the actual trade"""
        try:
            # Prepare order request
            symbol_info = self.mt5_connector.get_symbol_info(signal.symbol)
            if not symbol_info:
                return None
            if not self._is_spread_acceptable(symbol_info):
                self.logger.debug(f"Execution skipped for {signal.symbol}: spread/quote quality failed")
                return None
            
            # Determine order type and price
            if signal.action == 'BUY':
                order_type = mt5.ORDER_TYPE_BUY
                price = symbol_info['ask']
            else:  # SELL
                order_type = mt5.ORDER_TYPE_SELL
                price = symbol_info['bid']
            if price <= 0:
                self.logger.warning(f"Execution skipped for {signal.symbol}: invalid market price {price}")
                return None
            
            # Base order request; filling mode is selected with runtime fallback.
            base_order_request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": signal.symbol,
                "volume": signal.position_size,
                "type": order_type,
                "price": price,
                "sl": signal.stop_loss,
                "tp": signal.take_profit,
                "deviation": 20,
                "magic": 123456,
                "comment": f"Neural-{signal.confidence:.1%}",
                "type_time": mt5.ORDER_TIME_GTC,
            }

            invalid_fill_retcode = getattr(mt5, 'TRADE_RETCODE_INVALID_FILL', 10030)
            market_closed_retcode = getattr(mt5, 'TRADE_RETCODE_MARKET_CLOSED', 10018)
            fill_modes = [mt5.ORDER_FILLING_FOK, mt5.ORDER_FILLING_IOC, mt5.ORDER_FILLING_RETURN]
            tried = []

            for fill_mode in fill_modes:
                order_request = dict(base_order_request)
                order_request["type_filling"] = fill_mode
                tried.append(fill_mode)

                result = self.mt5_connector.send_order(order_request)
                if result and result.get('retcode') == mt5.TRADE_RETCODE_DONE:
                    return result

                if result and result.get('retcode') == market_closed_retcode:
                    symbol_key = signal.symbol.upper()
                    now = datetime.now()
                    blocked_until = now + timedelta(seconds=self.market_closed_cooldown_seconds)
                    self._symbol_trade_block_until[symbol_key] = blocked_until

                    last_log = self._symbol_market_closed_log_time.get(symbol_key)
                    if not last_log or (now - last_log).total_seconds() >= 60:
                        self._symbol_market_closed_log_time[symbol_key] = now
                        self.logger.warning(
                            f"Market closed for {signal.symbol}; pausing new attempts until "
                            f"{blocked_until.strftime('%Y-%m-%d %H:%M:%S')}"
                        )
                    return None

                # Retry with another filling mode only when broker rejects filling policy.
                if not result or result.get('retcode') != invalid_fill_retcode:
                    self.logger.error(f"Trade execution failed: {result}")
                    return None

            self.logger.error(
                f"Trade execution failed after trying filling modes {tried}: invalid fill policy"
            )
            return None
            
        except Exception as e:
            self.logger.error(f"Error executing trade: {e}")
            return None
    
    # ------------------------------------------------------------------
    # ZeroPoint Trade Monitor — active position management
    # ------------------------------------------------------------------
    # 1. Trail SL using live ZeroPoint ATR trailing stop
    # 2. Max loss cutoff — close if losing more than threshold
    # 3. Break-even move — move SL to entry once in profit by threshold
    # ------------------------------------------------------------------

    def _zp_manage_positions(self):
        """Active management for open ZP positions.

        Called every trading loop cycle.  Updates SL via ZP trailing stop,
        enforces max-loss cutoff, and moves to break-even when in profit.
        """
        if not self.zeropoint_pure_mode:
            return

        import MetaTrader5 as mt5_lib
        mt5_positions = mt5_lib.positions_get()
        if not mt5_positions:
            # No open positions — clear tracker
            self._zp_position_tracker.clear()
            return

        # Clean up tracker for positions that no longer exist
        open_tickets = {p.ticket for p in mt5_positions}
        stale = [t for t in self._zp_position_tracker if t not in open_tickets]
        for t in stale:
            self._zp_position_tracker.pop(t, None)

        for pos in mt5_positions:
            try:
                self._zp_manage_single_position(pos)
            except Exception as e:
                self.logger.debug(f"ZP manage error {pos.symbol}: {e}")

    def _zp_manage_single_position(self, mt5_pos):
        """V4 Profit Capture — full 5-layer trade management.

        Layers:
        1. Early BE: Move SL to BE after 0.5x ATR favorable move
        2. Stall Exit: Move to BE after 6 H4 bars if TP1 not hit
        3. Micro-Partial: Close 15% of lot at 0.8x ATR profit
        4. TP1/TP2 Partials: Close 1/3 at TP1 (0.8x), 1/3 at TP2 (2.0x)
        5. Post-TP1 Trail: Trail SL 0.8x ATR behind max favorable price
        Plus: ZP flip exit, max loss cutoff
        """
        import MetaTrader5 as mt5_lib

        symbol = mt5_pos.symbol
        ticket = mt5_pos.ticket
        action = "BUY" if mt5_pos.type == 0 else "SELL"
        entry = mt5_pos.price_open
        current_sl = mt5_pos.sl
        current_tp = mt5_pos.tp
        current_price = mt5_pos.price_current
        pnl = mt5_pos.profit
        volume = mt5_pos.volume

        sym_info = mt5_lib.symbol_info(symbol)
        if sym_info is None:
            return
        point = sym_info.point
        digits = sym_info.digits

        # --- 0. MAX LOSS CUTOFF ---
        if pnl <= -self.zp_max_loss_dollars:
            self.logger.warning(
                f"V4 MAX LOSS: {symbol} {action} losing ${abs(pnl):.2f} — CLOSING"
            )
            self._mt5_close_position(ticket, symbol, action, volume)
            self._zp_position_tracker.pop(ticket, None)
            return

        # --- Initialize V4 tracker for new positions ---
        now = datetime.now()
        tracker = self._zp_position_tracker.get(ticket)
        if tracker is None:
            # Get ATR at entry from live H4 data
            atr_at_entry = 0.0
            try:
                h4_rates = self.mt5_connector.get_rates(symbol, mt5.TIMEFRAME_H4, 0, 200)
                if h4_rates:
                    df_h4 = self._prepare_ohlc_dataframe(h4_rates)
                    if df_h4 is not None and len(df_h4) >= 15:
                        zp_state = compute_zeropoint_state(df_h4)
                        if zp_state is not None and len(zp_state) > 0:
                            atr_at_entry = float(zp_state.iloc[-1].get("atr", 0))
            except Exception:
                pass

            if atr_at_entry <= 0:
                # Fallback: estimate ATR from SL distance
                atr_at_entry = abs(entry - current_sl) / 3.0 if current_sl > 0 else 0.001

            tracker = {
                "entry": entry,
                "atr": atr_at_entry,
                "direction": action,
                "open_time": now,
                "h4_bars_at_open": 0,
                "be_activated": False,
                "stall_activated": False,
                "micro_hit": False,
                "tp1_hit": False,
                "tp2_hit": False,
                "max_favorable_price": entry,
                "profit_lock_sl": None,
                "remaining_lot": volume,
                "original_lot": volume,
            }
            self._zp_position_tracker[ticket] = tracker

        atr = tracker["atr"]
        is_buy = (action == "BUY")

        # --- Track max favorable excursion ---
        if is_buy:
            if current_price > tracker["max_favorable_price"]:
                tracker["max_favorable_price"] = current_price
            max_profit_raw = tracker["max_favorable_price"] - entry
        else:
            if current_price < tracker["max_favorable_price"]:
                tracker["max_favorable_price"] = current_price
            max_profit_raw = entry - tracker["max_favorable_price"]

        # --- Count H4 bars elapsed ---
        h4_bars_elapsed = 0
        try:
            h4_rates = self.mt5_connector.get_rates(symbol, mt5.TIMEFRAME_H4, 0, 200)
            if h4_rates:
                df_h4 = self._prepare_ohlc_dataframe(h4_rates)
                if df_h4 is not None and len(df_h4) > 0:
                    # Count bars since entry time
                    import pandas as pd
                    entry_time = tracker["open_time"]
                    bars_after = df_h4[df_h4["time"] > entry_time]
                    h4_bars_elapsed = len(bars_after)

                    # Also get live ZP state for flip detection
                    zp_state = compute_zeropoint_state(df_h4)
        except Exception:
            zp_state = None

        # --- 1. ZP FLIP EXIT (highest priority) ---
        try:
            if zp_state is not None and len(zp_state) > 0:
                zp_pos = int(zp_state.iloc[-1].get("pos", 0))
                if (is_buy and zp_pos == -1) or (not is_buy and zp_pos == 1):
                    self.logger.warning(
                        f"V4 ZP_FLIP: {symbol} {action} — ZP flipped against us, CLOSING"
                    )
                    self._mt5_close_position(ticket, symbol, action, tracker["remaining_lot"])
                    self._zp_position_tracker.pop(ticket, None)
                    return
        except Exception:
            pass

        new_sl = current_sl
        new_tp = current_tp
        reason_parts = []

        # --- 2. EARLY BREAKEVEN (0.5x ATR favorable move) ---
        if not tracker["be_activated"] and atr > 0:
            if max_profit_raw >= self.v4_be_trigger * atr:
                be_buffer = self.v4_be_buffer * atr
                if is_buy:
                    be_level = round(entry + be_buffer, digits)
                    if be_level > new_sl:
                        new_sl = be_level
                        tracker["be_activated"] = True
                        reason_parts.append("V4-BE")
                        self.logger.info(
                            f"V4 BE ACTIVATED: {symbol} {action} "
                            f"max_profit={max_profit_raw/atr:.2f}x ATR -> SL to {be_level}"
                        )
                else:
                    be_level = round(entry - be_buffer, digits)
                    if new_sl == 0 or be_level < new_sl:
                        new_sl = be_level
                        tracker["be_activated"] = True
                        reason_parts.append("V4-BE")
                        self.logger.info(
                            f"V4 BE ACTIVATED: {symbol} {action} "
                            f"max_profit={max_profit_raw/atr:.2f}x ATR -> SL to {be_level}"
                        )

        # --- 3. STALL EXIT (6 H4 bars, no TP1 -> move to BE) ---
        if not tracker["tp1_hit"] and not tracker["stall_activated"] and atr > 0:
            if h4_bars_elapsed >= self.v4_stall_bars:
                be_buffer = self.v4_be_buffer * atr
                if is_buy:
                    stall_sl = round(entry + be_buffer, digits)
                    if stall_sl > new_sl:
                        new_sl = stall_sl
                        tracker["stall_activated"] = True
                        tracker["be_activated"] = True
                        reason_parts.append("V4-STALL")
                        self.logger.info(
                            f"V4 STALL EXIT: {symbol} {action} "
                            f"{h4_bars_elapsed} bars, no TP1 -> SL to BE {stall_sl}"
                        )
                else:
                    stall_sl = round(entry - be_buffer, digits)
                    if new_sl == 0 or stall_sl < new_sl:
                        new_sl = stall_sl
                        tracker["stall_activated"] = True
                        tracker["be_activated"] = True
                        reason_parts.append("V4-STALL")
                        self.logger.info(
                            f"V4 STALL EXIT: {symbol} {action} "
                            f"{h4_bars_elapsed} bars, no TP1 -> SL to BE {stall_sl}"
                        )

        # --- 4. MICRO-PARTIAL (15% at 0.8x ATR) ---
        if not tracker["micro_hit"] and not tracker["tp1_hit"] and atr > 0:
            micro_price = entry + self.v4_micro_mult * atr if is_buy else entry - self.v4_micro_mult * atr
            micro_triggered = (current_price >= micro_price) if is_buy else (current_price <= micro_price)

            if micro_triggered and tracker["remaining_lot"] > sym_info.volume_min:
                micro_lot = round(tracker["original_lot"] * self.v4_micro_pct, 2)
                micro_lot = max(sym_info.volume_min, micro_lot)
                # Snap to volume_step
                vol_step = sym_info.volume_step
                micro_lot = round(micro_lot / vol_step) * vol_step
                micro_lot = min(micro_lot, tracker["remaining_lot"] - sym_info.volume_min)

                if micro_lot >= sym_info.volume_min:
                    self.logger.info(
                        f"V4 MICRO-TP: {symbol} {action} closing {micro_lot:.2f} lots "
                        f"(15% at {self.v4_micro_mult}x ATR)"
                    )
                    self._mt5_close_partial(ticket, symbol, action, micro_lot, "V4-micro-TP")
                    tracker["remaining_lot"] = round(tracker["remaining_lot"] - micro_lot, 2)
                    tracker["micro_hit"] = True

        # --- 5. TP1 PARTIAL (33% at 0.8x ATR) ---
        if not tracker["tp1_hit"] and atr > 0:
            tp1_price = entry + self.v4_tp1_mult * atr if is_buy else entry - self.v4_tp1_mult * atr
            tp1_triggered = (current_price >= tp1_price) if is_buy else (current_price <= tp1_price)

            if tp1_triggered and tracker["remaining_lot"] > sym_info.volume_min:
                tp1_lot = round(tracker["original_lot"] * 0.333, 2)
                tp1_lot = max(sym_info.volume_min, tp1_lot)
                vol_step = sym_info.volume_step
                tp1_lot = round(tp1_lot / vol_step) * vol_step
                tp1_lot = min(tp1_lot, tracker["remaining_lot"] - sym_info.volume_min)

                if tp1_lot >= sym_info.volume_min:
                    self.logger.info(
                        f"V4 TP1 HIT: {symbol} {action} closing {tp1_lot:.2f} lots "
                        f"(33% at {self.v4_tp1_mult}x ATR = {tp1_price:.{digits}f})"
                    )
                    self._mt5_close_partial(ticket, symbol, action, tp1_lot, "V4-TP1")
                    tracker["remaining_lot"] = round(tracker["remaining_lot"] - tp1_lot, 2)
                    tracker["tp1_hit"] = True

        # --- 6. TP2 PARTIAL (33% at 2.0x ATR) ---
        if tracker["tp1_hit"] and not tracker["tp2_hit"] and atr > 0:
            tp2_price = entry + self.v4_tp2_mult * atr if is_buy else entry - self.v4_tp2_mult * atr
            tp2_triggered = (current_price >= tp2_price) if is_buy else (current_price <= tp2_price)

            if tp2_triggered and tracker["remaining_lot"] > sym_info.volume_min:
                tp2_lot = round(tracker["original_lot"] * 0.333, 2)
                tp2_lot = max(sym_info.volume_min, tp2_lot)
                vol_step = sym_info.volume_step
                tp2_lot = round(tp2_lot / vol_step) * vol_step
                tp2_lot = min(tp2_lot, tracker["remaining_lot"] - sym_info.volume_min)

                if tp2_lot >= sym_info.volume_min:
                    self.logger.info(
                        f"V4 TP2 HIT: {symbol} {action} closing {tp2_lot:.2f} lots "
                        f"(33% at {self.v4_tp2_mult}x ATR = {tp2_price:.{digits}f})"
                    )
                    self._mt5_close_partial(ticket, symbol, action, tp2_lot, "V4-TP2")
                    tracker["remaining_lot"] = round(tracker["remaining_lot"] - tp2_lot, 2)
                    tracker["tp2_hit"] = True
                    # Also move SL to BE on TP2
                    if is_buy:
                        if entry > new_sl:
                            new_sl = round(entry, digits)
                            reason_parts.append("V4-TP2-BE")
                    else:
                        if new_sl == 0 or entry < new_sl:
                            new_sl = round(entry, digits)
                            reason_parts.append("V4-TP2-BE")

        # --- 7. POST-TP1 TRAILING STOP (0.8x ATR behind max favorable price) ---
        if tracker["tp1_hit"] and atr > 0:
            trail_dist = self.v4_trail_dist * atr
            if is_buy:
                lock_sl = round(tracker["max_favorable_price"] - trail_dist, digits)
                if lock_sl > entry and lock_sl > new_sl:
                    new_sl = lock_sl
                    tracker["profit_lock_sl"] = lock_sl
                    reason_parts.append("V4-TRAIL")
            else:
                lock_sl = round(tracker["max_favorable_price"] + trail_dist, digits)
                if lock_sl < entry and (new_sl == 0 or lock_sl < new_sl):
                    new_sl = lock_sl
                    tracker["profit_lock_sl"] = lock_sl
                    reason_parts.append("V4-TRAIL")

        # --- 8. ZP ATR TRAILING STOP (original — tighten SL from indicator) ---
        try:
            if zp_state is not None and len(zp_state) > 0:
                zp_stop = float(zp_state.iloc[-1].get("xATRTrailingStop", 0))
                zp_pos = int(zp_state.iloc[-1].get("pos", 0))
                if zp_stop > 0:
                    if is_buy and zp_pos == 1:
                        zp_sl = round(zp_stop, digits)
                        if zp_sl > new_sl:
                            new_sl = zp_sl
                            reason_parts.append("ZP-trail")
                    elif not is_buy and zp_pos == -1:
                        zp_sl = round(zp_stop, digits)
                        if new_sl == 0 or zp_sl < new_sl:
                            new_sl = zp_sl
                            reason_parts.append("ZP-trail")
        except Exception:
            pass

        # --- Apply SL if changed ---
        if new_sl != current_sl and new_sl > 0:
            reason_str = "+".join(reason_parts) or "V4-tighten"
            self._mt5_modify_sl(ticket, symbol, new_sl, current_tp, reason_str)

    def _mt5_close_position(self, ticket: int, symbol: str, action: str, volume: float):
        """Send MT5 close order for a position."""
        import MetaTrader5 as mt5_lib

        sym_info = mt5_lib.symbol_info(symbol)
        if sym_info is None:
            return

        if action == "BUY":
            close_type = mt5.ORDER_TYPE_SELL
            price = sym_info.bid
        else:
            close_type = mt5.ORDER_TYPE_BUY
            price = sym_info.ask

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume,
            "type": close_type,
            "position": ticket,
            "price": price,
            "deviation": 20,
            "magic": 123456,
            "comment": "ZP-monitor-exit",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_FOK,
        }

        result = mt5_lib.order_send(request)
        if result and result.retcode == mt5_lib.TRADE_RETCODE_DONE:
            self.logger.info(f"ZP CLOSED {symbol} {action} ticket={ticket} @ {price}")
            # Update internal tracker
            if ticket in self.positions:
                self.positions[ticket].status = 'CLOSED'
                self.positions[ticket].current_price = price
                self._register_closed_position(self.positions[ticket])
        else:
            # Try other fill modes
            for fill in [mt5.ORDER_FILLING_IOC, mt5.ORDER_FILLING_RETURN]:
                request["type_filling"] = fill
                result = mt5_lib.order_send(request)
                if result and result.retcode == mt5_lib.TRADE_RETCODE_DONE:
                    self.logger.info(f"ZP CLOSED {symbol} {action} ticket={ticket} @ {price}")
                    if ticket in self.positions:
                        self.positions[ticket].status = 'CLOSED'
                        self.positions[ticket].current_price = price
                        self._register_closed_position(self.positions[ticket])
                    return
            self.logger.error(f"ZP close FAILED {symbol}: {result}")

    def _mt5_close_partial(self, ticket: int, symbol: str, action: str, volume: float, comment: str = "V4-partial"):
        """Close a PARTIAL volume of an open MT5 position (for V4 partials)."""
        import MetaTrader5 as mt5_lib

        sym_info = mt5_lib.symbol_info(symbol)
        if sym_info is None:
            return

        # Snap volume to step
        vol_step = sym_info.volume_step
        volume = round(volume / vol_step) * vol_step
        volume = max(sym_info.volume_min, min(volume, sym_info.volume_max))

        if action == "BUY":
            close_type = mt5.ORDER_TYPE_SELL
            price = sym_info.bid
        else:
            close_type = mt5.ORDER_TYPE_BUY
            price = sym_info.ask

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume,
            "type": close_type,
            "position": ticket,
            "price": price,
            "deviation": 20,
            "magic": 123456,
            "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_FOK,
        }

        result = mt5_lib.order_send(request)
        if result and result.retcode == mt5_lib.TRADE_RETCODE_DONE:
            self.logger.info(
                f"V4 PARTIAL CLOSE: {symbol} {action} {volume:.2f} lots ({comment}) @ {price}"
            )
        else:
            # Try other fill modes
            for fill in [mt5.ORDER_FILLING_IOC, mt5.ORDER_FILLING_RETURN]:
                request["type_filling"] = fill
                result = mt5_lib.order_send(request)
                if result and result.retcode == mt5_lib.TRADE_RETCODE_DONE:
                    self.logger.info(
                        f"V4 PARTIAL CLOSE: {symbol} {action} {volume:.2f} lots ({comment}) @ {price}"
                    )
                    return
            self.logger.error(f"V4 partial close FAILED {symbol}: {result}")

    def _mt5_modify_sl(self, ticket: int, symbol: str, new_sl: float, tp: float, reason: str):
        """Modify SL on an open MT5 position."""
        import MetaTrader5 as mt5_lib

        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "symbol": symbol,
            "position": ticket,
            "sl": new_sl,
            "tp": tp,
        }

        result = mt5_lib.order_send(request)
        if result and result.retcode == mt5_lib.TRADE_RETCODE_DONE:
            self.logger.info(
                f"ZP SL updated ({reason}): {symbol} ticket={ticket} new_SL={new_sl}"
            )
        else:
            self.logger.debug(f"ZP SL modify failed {symbol}: {result}")

    def _update_positions(self):
        """Update position information and check for exits"""
        try:
            # ZeroPoint active management (trail SL, max loss, break-even)
            self._zp_manage_positions()

            # Get current positions from MT5
            mt5_positions = self.mt5_connector.get_positions()

            # Update our position tracking
            for ticket, position in list(self.positions.items()):
                # Find position in MT5
                mt5_pos = None
                for pos in mt5_positions:
                    if pos['ticket'] == ticket:
                        mt5_pos = pos
                        break

                if mt5_pos:
                    # Update position data
                    position.current_price = mt5_pos['price_current']
                    position.unrealized_pnl = mt5_pos['profit']

                    # Check exit conditions
                    if self._should_close_position(position, mt5_pos):
                        self._close_position(position)
                else:
                    # Position no longer exists in MT5, mark as closed
                    self._register_closed_position(position)
                    position.status = 'CLOSED'
                    self.logger.info(f"Position {ticket} closed")

        except Exception as e:
            self.logger.error(f"Error updating positions: {e}")

    def _should_close_position(self, position: Position, mt5_pos: Dict[str, Any]) -> bool:
        """Determine if position should be closed"""
        try:
            current_price = mt5_pos['price_current']

            if position.action == 'BUY':
                # Check stop loss
                if current_price <= position.stop_loss:
                    return True
                # Check take profit
                if current_price >= position.take_profit:
                    return True
            else:  # SELL
                # Check stop loss
                if current_price >= position.stop_loss:
                    return True
                # Check take profit
                if current_price <= position.take_profit:
                    return True

            return False

        except Exception as e:
            self.logger.error(f"Error checking exit conditions: {e}")
            return False

    def _close_position(self, position: Position):
        """Close a position"""
        try:
            self._mt5_close_position(
                position.ticket, position.symbol, position.action, position.position_size
            )
        except Exception as e:
            self.logger.error(f"Error closing position: {e}")
    
    def _update_performance_metrics(self):
        """Update trading performance metrics"""
        try:
            # Calculate metrics from positions and signals
            closed_positions = [p for p in self.positions.values() if p.status == 'CLOSED']
            
            if closed_positions:
                winning_trades = sum(1 for p in closed_positions if p.unrealized_pnl > 0)
                losing_trades = sum(1 for p in closed_positions if p.unrealized_pnl < 0)
                total_trades = len(closed_positions)
                
                self.performance_metrics.update({
                    'total_trades': total_trades,
                    'winning_trades': winning_trades,
                    'losing_trades': losing_trades,
                    'win_rate': winning_trades / total_trades if total_trades > 0 else 0,
                    'total_pnl': sum(p.unrealized_pnl for p in closed_positions),
                    'current_drawdown': self._calculate_drawdown(closed_positions)
                })
                
        except Exception as e:
            self.logger.error(f"Error updating performance metrics: {e}")
    
    def _calculate_drawdown(self, positions: List[Position]) -> float:
        """Calculate current drawdown"""
        # Simplified drawdown calculation
        if not positions:
            return 0.0
        
        profits = [p.unrealized_pnl for p in positions]
        if not profits:
            return 0.0
        
        peak = max(profits)
        current = sum(profits)
        
        if peak <= 0:
            return 0.0
        
        return max(0, (peak - current) / peak)
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        return self.performance_metrics.copy()
    
    def get_active_signals(self) -> List[Dict[str, Any]]:
        """Get list of recent signals"""
        recent_signals = [s for s in self.signals_history[-20:]]  # Last 20 signals
        
        return [
            {
                'timestamp': signal.timestamp.isoformat(),
                'symbol': signal.symbol,
                'action': signal.action,
                'confidence': signal.confidence,
                'entry_price': signal.entry_price,
                'stop_loss': signal.stop_loss,
                'take_profit': signal.take_profit,
                'position_size': signal.position_size,
                'reason': signal.reason,
                'executed': signal.executed
            }
            for signal in recent_signals
        ]
    
    def get_active_positions(self) -> List[Dict[str, Any]]:
        """Get list of active positions"""
        active_positions = [p for p in self.positions.values() if p.status == 'OPEN']
        
        return [
            {
                'ticket': position.ticket,
                'symbol': position.symbol,
                'action': position.action,
                'entry_price': position.entry_price,
                'current_price': position.current_price,
                'stop_loss': position.stop_loss,
                'take_profit': position.take_profit,
                'position_size': position.position_size,
                'unrealized_pnl': position.unrealized_pnl,
                'open_time': position.open_time.isoformat(),
                'status': position.status
            }
            for position in active_positions
        ]
