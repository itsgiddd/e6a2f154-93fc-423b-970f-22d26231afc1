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

    def _linear_fit(self, indices: np.ndarray, values: np.ndarray) -> Optional[Tuple[float, float, float]]:
        if len(indices) < 2:
            return None
        slope, intercept = np.polyfit(indices, values, 1)
        fitted = slope * indices + intercept
        ss_res = np.sum((values - fitted) ** 2)
        ss_tot = np.sum((values - np.mean(values)) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot != 0 else 0.0
        return slope, intercept, r2

    def _line_value(self, slope: float, intercept: float, idx: int) -> float:
        return slope * idx + intercept

    def _breakout_confirmed(self, direction: str, idx: int, slope: float, intercept: float) -> bool:
        level = self._line_value(slope, intercept, idx)
        close = self.close[idx]
        if direction == "bullish":
            return close > level
        if direction == "bearish":
            return close < level
        return False

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
        patterns.extend(self.detect_flags_pennants_rectangles())
        patterns.extend(self.detect_rounding_and_cup())
        patterns.extend(self.detect_diamonds())
        return patterns

    def detect_double_top_bottom(self) -> List[Pattern]:
        patterns = []
        peaks = self.find_peaks(order=5)
        troughs = self.find_troughs(order=5)
        last_idx = len(self.data) - 1
        
        # Double Top
        for i in range(len(peaks) - 1):
            p1 = peaks[i]
            p2 = peaks[i+1]
            if abs(self.highs[p1] - self.highs[p2]) / self.highs[p1] < 0.01:
                inter_troughs = [t for t in troughs if p1 < t < p2]
                if inter_troughs:
                    t_idx = inter_troughs[0]
                    neckline = self.lows[t_idx]
                    if self.close[last_idx] > neckline:
                        continue
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
                    if self.close[last_idx] < neckline:
                        continue
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
        last_idx = len(self.data) - 1
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
                        if self.close[last_idx] > neckline:
                            continue
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
                        if self.close[last_idx] < neckline:
                            continue
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
        peaks = self.find_peaks(order=3)
        troughs = self.find_troughs(order=3)
        if len(peaks) < 3 or len(troughs) < 3:
            return []

        window = 60
        end_idx = len(self.data) - 1
        start_idx = max(0, end_idx - window)
        recent_peaks = np.array([p for p in peaks if p >= start_idx])
        recent_troughs = np.array([t for t in troughs if t >= start_idx])
        if len(recent_peaks) < 2 or len(recent_troughs) < 2:
            return []

        res_fit = self._linear_fit(recent_peaks, self.highs[recent_peaks])
        sup_fit = self._linear_fit(recent_troughs, self.lows[recent_troughs])
        if not res_fit or not sup_fit:
            return []

        res_slope, res_intercept, res_r2 = res_fit
        sup_slope, sup_intercept, sup_r2 = sup_fit
        if res_r2 < 0.6 or sup_r2 < 0.6:
            return []

        vol_score = self._calc_volume_score(start_idx, end_idx)
        pattern_high = max(self.highs[start_idx:end_idx+1])
        pattern_low = min(self.lows[start_idx:end_idx+1])
        pattern_height = pattern_high - pattern_low
        details = {"height": pattern_height}
        pushes = min(len(recent_peaks), len(recent_troughs))

        res_level = self._line_value(res_slope, res_intercept, end_idx)
        sup_level = self._line_value(sup_slope, sup_intercept, end_idx)
        converging = res_level - sup_level < (pattern_height * 0.6)

        if abs(res_slope) < 0.0001 and sup_slope > 0.0001 and converging:
            if self._breakout_confirmed("bullish", end_idx, res_slope, res_intercept):
                details["stop_loss"] = self.lows[recent_troughs[-1]]
                patterns.append(Pattern("Ascending Triangle", start_idx, end_idx, 0.78, "bullish", details, pushes, vol_score))
        elif res_slope < -0.0001 and abs(sup_slope) < 0.0001 and converging:
            if self._breakout_confirmed("bearish", end_idx, sup_slope, sup_intercept):
                details["stop_loss"] = self.highs[recent_peaks[-1]]
                patterns.append(Pattern("Descending Triangle", start_idx, end_idx, 0.78, "bearish", details, pushes, vol_score))
        elif res_slope < 0 and sup_slope > 0 and converging:
            if self._breakout_confirmed("bullish", end_idx, res_slope, res_intercept):
                details["stop_loss"] = self.lows[recent_troughs[-1]]
                patterns.append(Pattern("Symmetrical Triangle", start_idx, end_idx, 0.74, "bullish", details, pushes, vol_score))
            elif self._breakout_confirmed("bearish", end_idx, sup_slope, sup_intercept):
                details["stop_loss"] = self.highs[recent_peaks[-1]]
                patterns.append(Pattern("Symmetrical Triangle", start_idx, end_idx, 0.74, "bearish", details, pushes, vol_score))
        elif res_slope > 0 and sup_slope > 0 and converging:
            if sup_slope > res_slope and self._breakout_confirmed("bearish", end_idx, sup_slope, sup_intercept):
                details["stop_loss"] = self.highs[recent_peaks[-1]]
                patterns.append(Pattern("Rising Wedge", start_idx, end_idx, 0.8, "bearish", details, pushes, vol_score))
        elif res_slope < 0 and sup_slope < 0 and converging:
            if res_slope < sup_slope and self._breakout_confirmed("bullish", end_idx, res_slope, res_intercept):
                details["stop_loss"] = self.lows[recent_troughs[-1]]
                patterns.append(Pattern("Falling Wedge", start_idx, end_idx, 0.8, "bullish", details, pushes, vol_score))
        return patterns

    def detect_flags_pennants_rectangles(self) -> List[Pattern]:
        patterns = []
        if len(self.data) < 40:
            return patterns

        window = 30
        pole_window = 20
        consolidation_window = 10
        end_idx = len(self.data) - 1
        start_idx = max(0, end_idx - window)

        pole_start = max(0, end_idx - pole_window)
        pole_high = np.max(self.highs[pole_start:end_idx + 1])
        pole_low = np.min(self.lows[pole_start:end_idx + 1])
        pole_length = pole_high - pole_low

        if pole_length <= 0:
            return patterns

        trend_dir = "bullish" if self.close[end_idx] >= self.close[pole_start] else "bearish"
        consolidation_start = max(0, end_idx - consolidation_window)

        cons_highs = self.highs[consolidation_start:end_idx + 1]
        cons_lows = self.lows[consolidation_start:end_idx + 1]
        cons_range = np.max(cons_highs) - np.min(cons_lows)

        # Require tight consolidation relative to the pole.
        if cons_range > pole_length * 0.5:
            return patterns

        res_slope = self.get_slope(
            consolidation_start,
            end_idx,
            cons_highs[0],
            cons_highs[-1]
        )
        sup_slope = self.get_slope(
            consolidation_start,
            end_idx,
            cons_lows[0],
            cons_lows[-1]
        )
        vol_score = self._calc_volume_score(consolidation_start, end_idx)
        details = {"height": pole_length}

        if abs(res_slope) < 0.0001 and abs(sup_slope) < 0.0001:
            name = "Bullish Rectangle" if trend_dir == "bullish" else "Bearish Rectangle"
            details["stop_loss"] = np.min(cons_lows) if trend_dir == "bullish" else np.max(cons_highs)
            breakout = self.close[end_idx] > np.max(cons_highs) if trend_dir == "bullish" else self.close[end_idx] < np.min(cons_lows)
            if breakout:
                patterns.append(Pattern(name, consolidation_start, end_idx, 0.7, trend_dir, details, 2, vol_score))
            return patterns

        if res_slope < 0 and sup_slope > 0:
            name = "Bullish Pennant" if trend_dir == "bullish" else "Bearish Pennant"
            details["stop_loss"] = np.min(cons_lows) if trend_dir == "bullish" else np.max(cons_highs)
            breakout = self.close[end_idx] > np.max(cons_highs) if trend_dir == "bullish" else self.close[end_idx] < np.min(cons_lows)
            if breakout:
                patterns.append(Pattern(name, consolidation_start, end_idx, 0.75, trend_dir, details, 2, vol_score))
            return patterns

        # Flag if both slopes tilt against the trend.
        if trend_dir == "bullish" and res_slope < 0 and sup_slope < 0:
            details["stop_loss"] = np.min(cons_lows)
            breakout = self.close[end_idx] > np.max(cons_highs)
            if breakout:
                patterns.append(Pattern("Bull Flag", consolidation_start, end_idx, 0.72, "bullish", details, 2, vol_score))
        elif trend_dir == "bearish" and res_slope > 0 and sup_slope > 0:
            details["stop_loss"] = np.max(cons_highs)
            breakout = self.close[end_idx] < np.min(cons_lows)
            if breakout:
                patterns.append(Pattern("Bear Flag", consolidation_start, end_idx, 0.72, "bearish", details, 2, vol_score))
        return patterns

    def detect_rounding_and_cup(self) -> List[Pattern]:
        patterns = []
        if len(self.data) < 60:
            return patterns

        window = 50
        end_idx = len(self.data) - 1
        start_idx = max(0, end_idx - window)
        idx = np.arange(start_idx, end_idx + 1)
        prices = self.close[start_idx:end_idx + 1]

        if len(prices) < 10:
            return patterns

        coeffs = np.polyfit(idx, prices, 2)
        curvature = coeffs[0]
        vol_score = self._calc_volume_score(start_idx, end_idx)
        height = np.max(self.highs[start_idx:end_idx + 1]) - np.min(self.lows[start_idx:end_idx + 1])
        details = {"height": height}

        # Handle detection for teacup: small pullback at the end.
        handle_window = 10
        handle_start = max(start_idx, end_idx - handle_window)
        handle_high = np.max(self.highs[handle_start:end_idx + 1])
        handle_low = np.min(self.lows[handle_start:end_idx + 1])
        handle_depth = handle_high - handle_low

        if curvature > 0:
            details["stop_loss"] = np.min(self.lows[start_idx:end_idx + 1])
            neckline = np.max(self.highs[start_idx:end_idx + 1])
            if self.close[end_idx] > neckline:
                if handle_depth < height * 0.25:
                    patterns.append(Pattern("Teacup", start_idx, end_idx, 0.7, "bullish", details, 3, vol_score))
                else:
                    patterns.append(Pattern("Rounding Bottom", start_idx, end_idx, 0.68, "bullish", details, 3, vol_score))
        elif curvature < 0:
            details["stop_loss"] = np.max(self.highs[start_idx:end_idx + 1])
            neckline = np.min(self.lows[start_idx:end_idx + 1])
            if self.close[end_idx] < neckline:
                patterns.append(Pattern("Rounding Top", start_idx, end_idx, 0.68, "bearish", details, 3, vol_score))

        return patterns

    def detect_diamonds(self) -> List[Pattern]:
        patterns = []
        if len(self.data) < 60:
            return patterns

        window = 40
        end_idx = len(self.data) - 1
        start_idx = max(0, end_idx - window)
        midpoint = start_idx + (window // 2)

        first_high = np.max(self.highs[start_idx:midpoint + 1])
        first_low = np.min(self.lows[start_idx:midpoint + 1])
        second_high = np.max(self.highs[midpoint:end_idx + 1])
        second_low = np.min(self.lows[midpoint:end_idx + 1])

        first_range = first_high - first_low
        second_range = second_high - second_low

        if first_range <= 0 or second_range <= 0:
            return patterns

        # Diamond: expand then contract.
        if first_range < second_range:
            return patterns

        vol_score = self._calc_volume_score(start_idx, end_idx)
        height = max(first_high, second_high) - min(first_low, second_low)
        details = {"height": height}

        upper_line = max(first_high, second_high)
        lower_line = min(first_low, second_low)
        if self.close[end_idx] > upper_line:
            details["stop_loss"] = lower_line
            patterns.append(Pattern("Bullish Diamond", start_idx, end_idx, 0.65, "bullish", details, 3, vol_score))
        elif self.close[end_idx] < lower_line:
            details["stop_loss"] = upper_line
            patterns.append(Pattern("Bearish Diamond", start_idx, end_idx, 0.65, "bearish", details, 3, vol_score))

        return patterns
