from market_context import MarketContextAnalyzer
from pattern_recognition import PatternRecognizer
from trade_validator import TradeValidator, TradeDecision
from adaptive_risk import AdaptiveRiskManager
from trading_memory import TradingMemory
import pandas as pd

class AIBrain:
    """
    Coordinator mimicking human brain layers:
    1. Context (Occipital/Parietal)
    2. Pattern (Temporal)
    3. Validation (Pre-frontal)
    4. Memory (Hippocampus)
    5. Risk (Amygdala regulation)
    """
    def __init__(self):
        self.market_analyzer = MarketContextAnalyzer()
        # PatternRecognizer is instantiated per data slice usually, or we can keep a persistent one?
        # Usually per data. We will instantiate it inside 'think'.
        self.trade_validator = TradeValidator()
        self.risk_manager = AdaptiveRiskManager()
        self.memory = TradingMemory()
        self.reasoning_engine = ReasoningEngine()
        self.required_columns = {"open", "high", "low", "close"}

    def _sanitize_data(self, data: pd.DataFrame, name: str) -> pd.DataFrame:
        if data is None or data.empty:
            raise ValueError(f"{name} data is empty")
        missing = self.required_columns - set(data.columns)
        if missing:
            raise ValueError(f"{name} data missing columns: {', '.join(sorted(missing))}")
        cleaned = data.dropna(subset=list(self.required_columns)).copy()
        if cleaned.empty:
            raise ValueError(f"{name} data has no usable rows after cleanup")
        if "time" in cleaned.columns:
            cleaned = cleaned.sort_values("time")
        return cleaned

    def _validate_symbol_info(self, symbol_info) -> None:
        required_attrs = ("point", "volume_step", "volume_min", "volume_max", "trade_tick_value")
        missing = [attr for attr in required_attrs if not hasattr(symbol_info, attr)]
        if missing:
            raise ValueError(f"symbol_info missing fields: {', '.join(missing)}")

    def _fallback_stop_loss(self, pattern, data_h1: pd.DataFrame, price: float) -> float:
        lookback = data_h1.tail(20)
        if pattern.direction == "bullish":
            return float(lookback["low"].min())
        return float(lookback["high"].max())

    def _fallback_target_distance(self, data_h1: pd.DataFrame, risk: float) -> float:
        lookback = data_h1.tail(20)
        recent_range = float(lookback["high"].max() - lookback["low"].min())
        return max(recent_range, risk * 2)
        
    def think(self, symbol: str, data_h1: pd.DataFrame, data_h4: pd.DataFrame, data_d1: pd.DataFrame, account_info, symbol_info) -> dict:
        """
        Returns a dict with decision and execution details.
        """
        try:
            data_h1 = self._sanitize_data(data_h1, "H1")
            data_h4 = self._sanitize_data(data_h4, "H4")
            data_d1 = self._sanitize_data(data_d1, "D1")
            self._validate_symbol_info(symbol_info)
        except ValueError as exc:
            return {"decision": "REJECT", "reason": str(exc)}

        # 0. Check Memory (Revenge Trading / Kill Switch)
        if not self.memory.can_trade(symbol):
            return {"decision": "REJECT", "reason": "Memory Block (Loss Streak or Cooldown)"}

        if len(data_h1) < 60 or len(data_h4) < 50 or len(data_d1) < 20:
            return {"decision": "WAIT", "reason": "Insufficient market history"}
            
        # 1. Market Context
        market_state = self.market_analyzer.get_market_state(symbol, data_h1, data_h4, data_d1)
        
        # 2. Pattern Recognition (on Data H1/H4? Usually patterns on H1 or H4)
        # We assume H1 for patterns for now, or we iterate. 
        # For simplicity, let's say we check H1 patterns.
        recognizer = PatternRecognizer(data_h1)
        patterns = recognizer.detect_all()
        
        # Filter freshness
        current_len = len(data_h1)
        fresh_patterns = [p for p in patterns if p.index_end >= current_len - 2]
        
        if not fresh_patterns:
            return {"decision": "WAIT", "reason": "No fresh patterns"}
            
        # 3. Validation & Selection
        best_decision = None
        best_pattern = None
        best_feat = None
        best_rationale = []
        
        for pattern in fresh_patterns:
            # Extract features for this pattern (Stub for features needed by validator)
            # In live_trader we extract complex features. 
            # Here let's pass dummy features or minimal needed.
            # Validator needs: vol_anomaly -> pattern.volume_score
            features = {"vol_anomaly": pattern.volume_score * 2} # Approximate mapping
            
            decision = self.trade_validator.validate(pattern, market_state, features)
            
            if decision.should_trade:
                if best_decision is None or decision.confluence_score > best_decision.confluence_score:
                    best_decision = decision
                    best_pattern = pattern
                    best_rationale = decision.rationale
        
        if not best_decision or not best_decision.should_trade:
             # Log the rejection of the last one?
             msg = best_decision.rejection_reason if best_decision else "All patterns rejected"
             rationale = best_decision.rationale if best_decision else []
             return {"decision": "REJECT", "reason": msg, "reasoning": rationale}
             
        # 4. Risk Calculation
        price = data_h1['close'].iloc[-1]
        sl = best_pattern.details.get('stop_loss')
        if sl is None:
            sl = self._fallback_stop_loss(best_pattern, data_h1, price)
        sl = float(sl)
        target_distance = best_pattern.details.get('height', 0)

        risk = abs(price - sl)
        if risk == 0:
            return {"decision": "REJECT", "reason": "Zero risk distance", "reasoning": best_rationale + ["Rejected: zero risk distance"]}

        if target_distance <= 0:
            target_distance = self._fallback_target_distance(data_h1, risk)
            if target_distance <= 0:
                return {"decision": "REJECT", "reason": "No measured move target", "reasoning": best_rationale + ["Rejected: missing measured move"]}

        if best_pattern.direction == "bullish" and sl >= price:
            return {"decision": "REJECT", "reason": "Invalid SL for bullish setup", "reasoning": best_rationale + ["Rejected: SL above price"]}
        if best_pattern.direction == "bearish" and sl <= price:
            return {"decision": "REJECT", "reason": "Invalid SL for bearish setup", "reasoning": best_rationale + ["Rejected: SL below price"]}

        tp = price + target_distance if best_pattern.direction == "bullish" else price - target_distance
        reward = abs(tp - price)
        rr = reward / risk
        if rr < 2.0:
            return {"decision": "REJECT", "reason": f"RR below 1:2 ({rr:.2f})", "reasoning": best_rationale + [f"Rejected: RR {rr:.2f} < 2.0"]}

        reasoning_notes = self.reasoning_engine.build_reasoning(
            best_pattern,
            market_state,
            rr,
            best_rationale,
            price,
            sl,
            tp
        )
        
        lot = self.risk_manager.calculate_lot_size(
            symbol, 
            price, 
            sl, 
            best_decision.confluence_score,
            account_info,
            symbol_info
        )
        
        if lot <= 0:
             return {"decision": "REJECT", "reason": "Risk Calc = 0 Lot", "reasoning": best_rationale + ["Rejected: lot size below minimum"]}
             
        return {
            "decision": "TRADE",
            "pattern": best_pattern,
            "lot": lot,
            "sl": sl,
            "tp": tp,
            "reason": f"Score {best_decision.confluence_score} | {best_decision.rejection_reason} | RR {rr:.2f}",
            "reasoning": reasoning_notes,
            "confidence": best_decision.confidence,
            "market_state": market_state
        }
        
    def log_result(self, symbol, result, profit):
        self.memory.close_trade(symbol, profit)


class ReasoningEngine:
    def build_reasoning(self, pattern, market_state: dict, rr: float, validator_notes: list, price: float, sl: float, tp: float) -> list:
        notes = list(validator_notes)
        notes.append(f"Pattern: {pattern.name} ({pattern.direction})")
        notes.append(f"Pattern height: {pattern.details.get('height', 0):.5f}")
        notes.append(f"SL distance: {abs(price - sl):.5f}")
        notes.append(f"TP distance: {abs(tp - price):.5f}")
        notes.append(f"RR check: {rr:.2f} >= 2.0")
        notes.append(f"Session score: {market_state.get('session', 0):.2f}")
        notes.append(f"Strength score: {market_state.get('strength', 0):.2f}")
        return notes
