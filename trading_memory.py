from dataclasses import dataclass, field
from typing import List, Dict
import time

@dataclass
class TradeRecord:
    symbol: str
    action: str # BUY/SELL
    entry_time: float
    result: str = "OPEN" # OPEN, WIN, LOSS
    profit: float = 0.0

class TradingMemory:
    """
    Remembers recent trading activity to:
    - Prevent over-trading (max trades per day/session)
    - Detect losing streaks (halt trading)
    - Prevent revenge trading (trading same symbol immediately after loss)
    """
    def __init__(self):
        self.history: List[TradeRecord] = []
        self.daily_pnl = 0.0
        self.loss_streak = 0
        self.win_streak = 0
        
    def log_trade(self, symbol: str, action: str):
        self.history.append(TradeRecord(symbol, action, time.time()))
        
    def close_trade(self, symbol: str, profit: float):
        # Find most recent open trade for symbol (simplified)
        for trade in reversed(self.history):
            if trade.symbol == symbol and trade.result == "OPEN":
                trade.result = "WIN" if profit > 0 else "LOSS"
                trade.profit = profit
                self.daily_pnl += profit
                
                if profit > 0:
                    self.win_streak += 1
                    self.loss_streak = 0
                else:
                    self.loss_streak += 1
                    self.win_streak = 0
                break
                
    def can_trade(self, symbol: str) -> bool:
        """Sanity checks based on recent history"""
        # 1. Kill switch: Max daily loss?
        if self.daily_pnl < -100: # Example limit
            return False
            
        # 2. Too many consecutive losses?
        if self.loss_streak >= 3:
            return False 
            
        # 3. Just lost on this symbol? (Cool off)
        recent_symbol_trades = [t for t in self.history if t.symbol == symbol][-1:]
        if recent_symbol_trades:
            last = recent_symbol_trades[0]
            if last.result == "LOSS" and (time.time() - last.entry_time < 3600):
                # Lost within last hour? Don't trade again.
                return False
                
        return True
