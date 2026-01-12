import MetaTrader5 as mt5
import pandas as pd
from ai_brain import AIBrain
from dataclasses import dataclass
import time

# Mock classes for Backtest context
@dataclass
class MockAccountInfo:
    balance: float = 1000.0
    equity: float = 1000.0
    margin_free: float = 1000.0
    leverage: int = 100

@dataclass
class MockSymbolInfo:
    point: float = 0.00001
    trade_tick_value: float = 1.0
    volume_step: float = 0.01
    volume_min: float = 0.01
    volume_max: float = 100.0

class Backtester:
    def __init__(self, symbol="GBPJPY", days=30):
        self.symbol = symbol
        self.days = days
        self.brain = AIBrain()
        self.account = MockAccountInfo()
        self.symbol_info = MockSymbolInfo()
        # Custom point adjustments
        if "JPY" in symbol:
             self.symbol_info.point = 0.001
             self.symbol_info.trade_tick_value = 0.68 # Approx for standard lot
        else:
             self.symbol_info.point = 0.00001
             self.symbol_info.trade_tick_value = 1.0

    def fetch_history(self):
        if not mt5.initialize():
            print("MT5 Init Failed")
            return None
            
        print(f"Fetching {self.days} days of data for {self.symbol}...")
        
        # We need H1 for patterns, H4/D1 for context
        # Fetch MORE than needed to simulate stepping through time
        count_h1 = self.days * 24
        
        h1_rates = mt5.copy_rates_from_pos(self.symbol, mt5.TIMEFRAME_H1, 0, count_h1 + 200) # Buffer
        h4_rates = mt5.copy_rates_from_pos(self.symbol, mt5.TIMEFRAME_H4, 0, (self.days * 6) + 100)
        d1_rates = mt5.copy_rates_from_pos(self.symbol, mt5.TIMEFRAME_D1, 0, self.days + 50)
        
        if h1_rates is None: return None
        
        df_h1 = pd.DataFrame(h1_rates)
        df_h1['time'] = pd.to_datetime(df_h1['time'], unit='s')
        
        df_h4 = pd.DataFrame(h4_rates)
        df_h4['time'] = pd.to_datetime(df_h4['time'], unit='s')
        
        df_d1 = pd.DataFrame(d1_rates)
        df_d1['time'] = pd.to_datetime(df_d1['time'], unit='s')
        
        return df_h1, df_h4, df_d1

    def run(self):
        data = self.fetch_history()
        if not data: return
        
        full_h1, full_h4, full_d1 = data
        
        trades = []
        wins = 0
        losses = 0
        total_pnl = 0.0
        
        # Simulation Loop
        # We step through H1 bars, pretending that is "Now"
        # Start after buffer
        start_idx = 200
        
        print(f"Starting Backtest on {len(full_h1) - start_idx} Hourly Bars...")
        
        for i in range(start_idx, len(full_h1)):
            # Slice data to represent "past up to now"
            current_h1 = full_h1.iloc[:i+1] # Include current bar as "just closed" usually, or strictly past?
            # Pattern recognizer looks at closed bars usually.
            
            current_time = current_h1.iloc[-1]['time']
            
            # Filter H4/D1 to be <= current_time
            current_h4 = full_h4[full_h4['time'] <= current_time]
            current_d1 = full_d1[full_d1['time'] <= current_time]
            
            if len(current_h4) < 50 or len(current_d1) < 20: continue
            
            # --- BRAIN THINK ---
            # We ignore memory cooldowns for backtest simplicity mostly? 
            # Or we keep them to test logic.
            
            decision = self.brain.think(self.symbol, current_h1, current_h4, current_d1, self.account, self.symbol_info)
            
            if decision['decision'] == 'TRADE':
                # trade executed at close of bar 'i' (open of 'i+1')
                entry_price = decision['pattern'].details.get('entry', current_h1.iloc[-1]['close']) # Or Open of next?
                # Using Close of current is realistic enough for "Open of next"
                
                direction = decision['pattern'].direction
                sl = decision['sl']
                tp = decision['tp']
                
                # Check Outcome in FUTURE bars
                outcome = "OPEN"
                pnl = 0
                
                # Look ahead
                for f in range(i+1, len(full_h1)):
                    future_bar = full_h1.iloc[f]
                    high = future_bar['high']
                    low = future_bar['low']
                    
                    if direction == 'bullish':
                        if low <= sl:
                            outcome = "LOSS"
                            pnl = -1 * (entry_price - sl) # In points approx
                            break
                        if high >= tp:
                            outcome = "WIN"
                            pnl = (tp - entry_price)
                            break
                    else: # bearish
                        if high >= sl:
                            outcome = "LOSS"
                            pnl = -1 * (sl - entry_price)
                            break
                        if low <= tp:
                            outcome = "WIN"
                            pnl = (entry_price - tp)
                            break
                            
                    # Timeout? (e.g. 5 days)
                    if (future_bar['time'] - current_time).days > 5:
                        outcome = "TIMEOUT"
                        pnl = 0 
                        break
                
                # Log
                if outcome != "OPEN":
                    trades.append({
                        "time": current_time,
                        "type": direction,
                        "result": outcome,
                        "pnl": pnl,
                        "reason": decision['reason']
                    })
                    if outcome == "WIN": wins += 1
                    if outcome == "LOSS": losses += 1
                    
                    print(f"[{current_time}] {direction.upper()} -> {outcome} ({decision['reason']})")
                    
                    # Update Memory (Crucial for streaks)
                    self.brain.log_result(self.symbol, outcome, pnl)
        
        # Summary
        total = wins + losses
        win_rate = (wins / total * 100) if total > 0 else 0
        
        print("\n=== BACKTEST RESULTS ===")
        print(f"Symbol: {self.symbol} | Period: {self.days} Days")
        print(f"Total Trades: {total}")
        print(f"Wins: {wins}")
        print(f"Losses: {losses}")
        print(f"Win Rate: {win_rate:.1f}%")

if __name__ == "__main__":
    bt = Backtester("GBPJPY", days=30)
    bt.run()
