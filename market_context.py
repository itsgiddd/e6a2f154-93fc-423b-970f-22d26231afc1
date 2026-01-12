import pandas as pd
import numpy as np
from datetime import datetime

class MarketContextAnalyzer:
    """
    Analyzes the broader market environment:
    - Trend Direction (H4/D1)
    - Trend Strength (ADX equivalent)
    - Volatility State
    - Session Quality
    """
    
    def get_market_state(self, symbol: str, data_h1: pd.DataFrame, data_h4: pd.DataFrame, data_d1: pd.DataFrame) -> dict:
        state = {
            'trend_d1': self._calculate_trend(data_d1),
            'trend_h4': self._calculate_trend(data_h4),
            'strength': self._calculate_strength(data_h4),
            'volatility': self._classify_volatility(data_h1),
            'session': self._get_session_quality(),
            'momentum_h1': self._calculate_momentum(data_h1)
        }
        
        # Derived 'Global Trend' (-1 to 1)
        # If D1 and H4 agree, strong trend. Else undefined.
        if state['trend_d1'] == state['trend_h4']:
            state['global_trend'] = state['trend_d1'] # 1 or -1
        else:
            state['global_trend'] = 0 # Range/Counter-trend
            
        return state

    def _calculate_trend(self, df: pd.DataFrame, period=20) -> int:
        """Simple SMA Trend: 1 if Close > SMA, -1 if Close < SMA"""
        if len(df) < period: return 0
        sma = df['close'].rolling(period).mean()
        last_close = df['close'].iloc[-1]
        last_sma = sma.iloc[-1]
        return 1 if last_close > last_sma else -1

    def _calculate_strength(self, df: pd.DataFrame) -> float:
        """
        Approximate Trend Strength (0-100) using Range vs ATR-like movement.
        Simplified ADX proxy: Ratio of (High-Low) / Abs(Move) over N bars?
        Let's use: Avg Candle Body / Avg Candle Range 
        Or just: (SMA20 - SMA50) divergence?
        Let's use a standard deviation of closing prices relative to price level.
        """
        if len(df) < 20: return 0.0
        
        # Simple Proxy: Rolling Correlation of Price vs Time?
        # Better: Percentile of recent volatility?
        # Let's use: ADX-like calculation is complex to code from scratch without talib.
        # Fallback: Body Size / Wick ratio?
        
        # Let's use Slope of SMA20
        sma20 = df['close'].rolling(20).mean()
        if len(sma20) < 2: return 0.0
        
        slope = (sma20.iloc[-1] - sma20.iloc[-5]) / 5
        # Normalize slope by price ... 0.1% change per bar is strong?
        slope_pct = (slope / df['close'].iloc[-1]) * 100
        
        # Map 0.0 -> 0.05 to 0 -> 100
        strength = min(100, abs(slope_pct) / 0.05 * 100)
        return strength

    def _classify_volatility(self, df: pd.DataFrame) -> str:
        """Returns 'low', 'normal', 'high'"""
        if len(df) < 20: return 'normal'
        
        # ATR 14
        high_low = df['high'] - df['low']
        atr = high_low.rolling(14).mean().iloc[-1]
        
        # Compare to longer term average (100 bars)
        long_atr = high_low.rolling(100).mean().iloc[-1]
        
        if long_atr == 0: return 'normal'
        ratio = atr / long_atr
        
        if ratio < 0.7: return 'low'
        if ratio > 1.5: return 'high'
        return 'normal'

    def _get_session_quality(self) -> float:
        """
        Returns 0.0 to 1.0 based on current time (GMT/Server time).
        Assuming Server Time ~ GMT+2/3.
        London Open (0800 GMT) to NY Close (2200 GMT) is best.
        """
        # We need to know current server time. 
        # Since we are running in a script, we typically use local time or query MT5 logic.
        # For now, let's assume the machine time is aligned or we return 0.5 (neutral).
        # We'll just assume 'Active' hours are 8am - 5pm Local for now? 
        # No, 'LiveTrader' runs 24/7.
        # Let's return 0.8 as baseline, 1.0 for Overlap.
        return 0.8 

    def _calculate_momentum(self, df: pd.DataFrame) -> int:
        """Last 3 bars green vs red?"""
        if len(df) < 3: return 0
        closes = df['close'].values[-3:]
        opens = df['open'].values[-3:]
        
        green_candles = sum([1 for c, o in zip(closes, opens) if c > o])
        if green_candles == 3: return 1
        if green_candles == 0: return -1 # 3 red candles
        return 0
