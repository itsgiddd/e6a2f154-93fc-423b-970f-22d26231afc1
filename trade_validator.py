from dataclasses import dataclass, field
from typing import List

@dataclass
class TradeDecision:
    should_trade: bool
    rejection_reason: str = ""
    confluence_score: int = 0
    confidence: float = 0.0
    rationale: List[str] = field(default_factory=list)

class TradeValidator:
    """
    The 'Pre-Frontal Cortex' - Performs sanity checks and confluence scoring.
    """
    
    def validate(self, pattern, market_state: dict, features: dict) -> TradeDecision:
        reasons = []
        score = 0
        
        # 1. Trend Alignment (Critical)
        # If D1 trend exists, MUST align.
        global_trend = market_state.get('global_trend', 0)
        trend_score = 0
        if global_trend == 1 and pattern.direction == "bullish": trend_score = 1
        elif global_trend == -1 and pattern.direction == "bearish": trend_score = 1
        elif global_trend == 0: trend_score = 0 # Range
        
        # If strong trend against us -> Reject
        if global_trend == 1 and pattern.direction == "bearish":
            return TradeDecision(False, "Against D1 Uptrend", rationale=["Rejected: trend against trade"])
        if global_trend == -1 and pattern.direction == "bullish":
            return TradeDecision(False, "Against D1 Downtrend", rationale=["Rejected: trend against trade"])
            
        if trend_score == 1:
            score += 2 # Strong weight for trend
            reasons.append("Aligned with D1 trend (+2)")
        else:
            reasons.append("No D1 trend alignment (+0)")
        
        # 2. Push Count (Pattern Strength)
        push_count = getattr(pattern, 'push_count', 1) 
        if push_count >= 2: score += 1
        if push_count >= 3: score += 1 # Bonus
        
        if push_count < 1: 
             return TradeDecision(False, "Push Count < 1", rationale=["Rejected: insufficient pushes"])

        if push_count >= 4:
            return TradeDecision(False, "Push Exhaustion (>=4 pushes)", rationale=["Rejected: push exhaustion"])
        reasons.append(f"Push count {push_count} (+{1 if push_count >= 2 else 0}{'+1' if push_count >= 3 else ''})")
             
        # 3. Volume Confirmation
        vol_score = getattr(pattern, 'volume_score', 0.5)
        # Or use feature 'vol_anomaly'
        vol_anomaly = features.get('vol_anomaly', 1.0)
        if vol_anomaly > 1.5:
            score += 1
            reasons.append("Volume expansion (+1)")
        else:
            reasons.append("No volume expansion (+0)")
        
        # 4. Pattern Confidence (AI Probability)
        ai_conf = pattern.confidence
        if ai_conf > 0.6:
            score += 1
            reasons.append("Pattern confidence > 0.6 (+1)")
        else:
            reasons.append("Pattern confidence <= 0.6 (+0)")
        
        # 5. Session
        if market_state.get('session', 0) > 0.6:
            score += 1
            reasons.append("Session quality high (+1)")
        else:
            reasons.append("Session quality low (+0)")
        
        # 6. Market Strength
        if market_state.get('strength', 0) > 20:
            score += 1
            reasons.append("Market strength strong (+1)")
        else:
            reasons.append("Market strength weak (+0)")
        
        # Decision
        # REQUIRE 5+ (Strict)
        if score >= 5:
            if pattern.direction == "neutral":
                 return TradeDecision(False, "Rejected Neutral Pattern", score, ai_conf, reasons + ["Rejected: neutral direction"])
            return TradeDecision(True, "High Confluence", score, ai_conf, reasons + ["Approved: confluence threshold met"])
        else:
            reasons.append("Rejected: confluence below threshold")
            return TradeDecision(False, f"Low Confluence (Score {score}/8)", score, ai_conf, reasons)
