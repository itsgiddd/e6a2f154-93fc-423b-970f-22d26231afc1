from dataclasses import dataclass

@dataclass
class TradeDecision:
    should_trade: bool
    rejection_reason: str = ""
    confluence_score: int = 0
    confidence: float = 0.0

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
            return TradeDecision(False, "Against D1 Uptrend")
        if global_trend == -1 and pattern.direction == "bullish":
            return TradeDecision(False, "Against D1 Downtrend")
            
        if trend_score == 1: score += 2 # Strong weight for trend
        
        # 2. Push Count (Pattern Strength)
        push_count = getattr(pattern, 'push_count', 1) 
        if push_count >= 2: score += 1
        if push_count >= 3: score += 1 # Bonus
        
        if push_count < 1: 
             return TradeDecision(False, "Push Count < 1")
             
        # 3. Volume Confirmation
        vol_score = getattr(pattern, 'volume_score', 0.5)
        # Or use feature 'vol_anomaly'
        vol_anomaly = features.get('vol_anomaly', 1.0)
        if vol_anomaly > 1.5: score += 1
        
        # 4. Pattern Confidence (AI Probability)
        ai_conf = pattern.confidence
        if ai_conf > 0.6: score += 1
        
        # 5. Session
        if market_state.get('session', 0) > 0.6: score += 1
        
        # 6. Market Strength
        if market_state.get('strength', 0) > 20: score += 1
        
        # Decision
        # REQUIRE 5+ (Strict)
        if score >= 5:
            if pattern.direction == "neutral":
                 return TradeDecision(False, "Rejected Neutral Pattern")
            return TradeDecision(True, "High Confluence", score, ai_conf)
        else:
            return TradeDecision(False, f"Low Confluence (Score {score}/8)", score, ai_conf)
