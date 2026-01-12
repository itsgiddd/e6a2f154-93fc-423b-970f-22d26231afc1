import numpy as np
import pandas as pd
from scipy.signal import argrelextrema
from dataclasses import dataclass
from typing import List, Optional, Tuple

@dataclass
class Pattern:
    name: str # "Rising Wedge", "Double Top", etc.
    index_start: int
    index_end: int
    confidence: float
    direction: str # "bullish" or "bearish"
    details: dict
    push_count: int = 1
    volume_score: float = 0.5
    quality_grade: str = "B"

class PatternRecognizer:
    def __init__(self, data: pd.DataFrame):
        """
        data must have columns: ['open', 'high', 'low', 'close', 'tick_volume']
        """
        self.data = data
        self.highs = data['high'].values
        self.lows = data['low'].values
        self.close = data['close'].values
        # Ensure volume exists, handle if missing
        self.vol = data['tick_volume'].values if 'tick_volume' in data.columns else np.zeros(len(data))
        
    def find_peaks(self, order=5):
        return argrelextrema(self.highs, np.greater, order=order)[0]

    def find_troughs(self, order=5):
        return argrelextrema(self.lows, np.less, order=order)[0]

    def get_slope(self, idx1, idx2, val1, val2):
        if idx2 == idx1: return 0
        return (val2 - val1) / (idx2 - idx1)

    def _calc_volume_score(self, start_idx, end_idx) -> float:
        """
        Check if there is a volume spike during the pattern formation or breakout.
        Avg volume in pattern vs Avg volume before.
        """
        if start_idx <= 0: return 0.5
        
        pattern_vol = np.mean(self.vol[start_idx:end_idx+1])
        # Lookback 50 bars
        lookback = max(0, start_idx - 50)
        baseline_vol = np.mean(self.vol[lookback:start_idx])
        
        if baseline_vol == 0: return 1.0
        
        ratio = pattern_vol / baseline_vol
        
        # Normalize to 0-1 score
        # 1.0 ratio -> 0.5 score
        # 2.0 ratio -> 1.0 score
        # 0.5 ratio -> 0.2 score
        score = min(1.0, ratio / 2.0)
        return score

    def detect_all(self) -> List[Pattern]:
        patterns = []
        patterns.extend(self.detect_double_top_bottom())
        patterns.extend(self.detect_head_and_shoulders())
        patterns.extend(self.detect_triangles_and_wedges())
        return patterns

    def detect_double_top_bottom(self) -> List[Pattern]:
        patterns = []
        peaks = self.find_peaks(order=5)
        troughs = self.find_troughs(order=5)
        
        # Double Top
        for i in range(len(peaks) - 1):
            p1 = peaks[i]
            p2 = peaks[i+1]
            if abs(self.highs[p1] - self.highs[p2]) / self.highs[p1] < 0.01:
                inter_troughs = [t for t in troughs if p1 < t < p2]
                if inter_troughs:
                    t_idx = inter_troughs[0]
                    neckline = self.lows[t_idx]
                    pattern_height = max(self.highs[p1], self.highs[p2]) - neckline
                    
                    vol_score = self._calc_volume_score(p1, p2)
                    
                    patterns.append(Pattern(
                        name="Double Top",
                        index_start=p1,
                        index_end=p2,
                        confidence=0.85,
                        direction="bearish",
                        details={
                            "neckline": neckline, 
                            "height": pattern_height,
                            "stop_loss": max(self.highs[p1], self.highs[p2]) + (pattern_height * 0.1) 
                        },
                        push_count=2, # 2 Peaks
                        volume_score=vol_score,
                        quality_grade="A"
                    ))

        # Double Bottom
        for i in range(len(troughs) - 1):
            t1 = troughs[i]
            t2 = troughs[i+1]
            if abs(self.lows[t1] - self.lows[t2]) / self.lows[t1] < 0.01:
                inter_peaks = [p for p in peaks if t1 < p < t2]
                if inter_peaks:
                    p_idx = inter_peaks[0]
                    neckline = self.highs[p_idx]
                    pattern_height = neckline - min(self.lows[t1], self.lows[t2])
                    
                    vol_score = self._calc_volume_score(t1, t2)

                    patterns.append(Pattern(
                        name="Double Bottom",
                        index_start=t1,
                        index_end=t2,
                        confidence=0.85,
                        direction="bullish",
                        details={
                            "neckline": neckline,
                            "height": pattern_height,
                            "stop_loss": min(self.lows[t1], self.lows[t2]) - (pattern_height * 0.1)
                        },
                        push_count=2,
                        volume_score=vol_score,
                        quality_grade="A"
                    ))
        return patterns

    def detect_head_and_shoulders(self) -> List[Pattern]:
        patterns = []
        peaks = self.find_peaks(order=5)
        # ... (Abbreviated for token limits, assuming previous logic + upgrades)
        # For brevity, I'll include the full logic but compressed or similar to previous
        if len(peaks) >= 3:
            for i in range(len(peaks) - 2):
                l_shoulder = peaks[i]
                head = peaks[i+1]
                r_shoulder = peaks[i+2]
                h_val = self.highs[head]
                l_val = self.highs[l_shoulder]
                r_val = self.highs[r_shoulder]
                
                if h_val > l_val and h_val > r_val:
                    if abs(l_val - r_val) / l_val < 0.05:
                        troughs = self.find_troughs(order=5)
                        related_troughs = [t for t in troughs if l_shoulder < t < r_shoulder]
                        if not related_troughs: continue
                        neckline = np.mean([self.lows[t] for t in related_troughs])
                        pattern_height = h_val - neckline
                        
                        vol_score = self._calc_volume_score(l_shoulder, r_shoulder)
                        
                        patterns.append(Pattern(
                            name="Head and Shoulders (Top)",
                            index_start=l_shoulder,
                            index_end=r_shoulder,
                            confidence=0.8,
                            direction="bearish",
                            details={
                                "neckline": neckline,
                                "height": pattern_height,
                                "stop_loss": r_val + (pattern_height * 0.1)
                            },
                            push_count=3,
                            volume_score=vol_score,
                            quality_grade="A"
                        ))
        
        # Inverted H&S
        troughs = self.find_troughs(order=5)
        if len(troughs) >= 3:
             for i in range(len(troughs) - 2):
                l_shoulder = troughs[i]
                head = troughs[i+1]
                r_shoulder = troughs[i+2]
                h_val = self.lows[head]
                l_val = self.lows[l_shoulder]
                r_val = self.lows[r_shoulder]
                if h_val < l_val and h_val < r_val:
                    if abs(l_val - r_val) / l_val < 0.05:
                        peaks = self.find_peaks(order=5)
                        related_peaks = [p for p in peaks if l_shoulder < p < r_shoulder]
                        if not related_peaks: continue
                        neckline = np.mean([self.highs[p] for p in related_peaks])
                        pattern_height = neckline - h_val
                        
                        vol_score = self._calc_volume_score(l_shoulder, r_shoulder)
                        
                        patterns.append(Pattern(
                            name="Head and Shoulders (Bottom)",
                            index_start=l_shoulder,
                            index_end=r_shoulder,
                            confidence=0.8,
                            direction="bullish",
                            details={
                                "neckline": neckline,
                                "height": pattern_height,
                                "stop_loss": r_val - (pattern_height * 0.1)
                            },
                            push_count=3,
                            volume_score=vol_score,
                            quality_grade="A"
                        ))
        return patterns

    def detect_triangles_and_wedges(self) -> List[Pattern]:
        patterns = []
        peaks = self.find_peaks(order=3) # Use smaller order for triangles
        troughs = self.find_troughs(order=3)
        if len(peaks) < 3 or len(troughs) < 3: return []
        
        # Look at last few
        recent_peaks = peaks[-3:]
        recent_troughs = troughs[-3:]
        
        if len(recent_peaks) < 2 or len(recent_troughs) < 2: return []
        
        start_idx = min(recent_peaks[0], recent_troughs[0])
        end_idx = max(recent_peaks[-1], recent_troughs[-1])
        vol_score = self._calc_volume_score(start_idx, end_idx)
        
        res_slope = self.get_slope(recent_peaks[0], recent_peaks[-1], self.highs[recent_peaks[0]], self.highs[recent_peaks[-1]])
        sup_slope = self.get_slope(recent_troughs[0], recent_troughs[-1], self.lows[recent_troughs[0]], self.lows[recent_troughs[-1]])
        
        pattern_high = max(self.highs[start_idx:end_idx+1])
        pattern_low = min(self.lows[start_idx:end_idx+1])
        pattern_height = pattern_high - pattern_low

        details = {"height": pattern_height}
        # Pushes = number of touches (peaks + troughs) in the window
        # Approximate
        pushes = len([p for p in recent_peaks if p >= start_idx]) + len([t for t in recent_troughs if t >= start_idx])
        
        if abs(res_slope) < 0.0001 and sup_slope > 0.0001:
             details["stop_loss"] = self.lows[recent_troughs[-1]]
             patterns.append(Pattern("Ascending Triangle", start_idx, end_idx, 0.75, "bullish", details, pushes, vol_score))
        elif res_slope < -0.0001 and abs(sup_slope) < 0.0001:
             details["stop_loss"] = self.highs[recent_peaks[-1]]
             patterns.append(Pattern("Descending Triangle", start_idx, end_idx, 0.75, "bearish", details, pushes, vol_score))
        elif res_slope < 0 and sup_slope > 0:
            details["stop_loss"] = self.lows[recent_troughs[-1]]
            patterns.append(Pattern("Symmetrical Triangle", start_idx, end_idx, 0.7, "neutral", details, pushes, vol_score))
        elif res_slope > 0 and sup_slope > 0:
            if sup_slope > res_slope:
                details["stop_loss"] = self.highs[recent_peaks[-1]]
                patterns.append(Pattern("Rising Wedge", start_idx, end_idx, 0.8, "bearish", details, pushes, vol_score))
        elif res_slope < 0 and sup_slope < 0:
            if res_slope < sup_slope:
                 details["stop_loss"] = self.lows[recent_troughs[-1]]
                 patterns.append(Pattern("Falling Wedge", start_idx, end_idx, 0.8, "bullish", details, pushes, vol_score))
        return patterns
