import MetaTrader5 as mt5
import pandas as pd
import joblib
import time
import sys
import numpy as np
from ai_brain import AIBrain

# --- CONFIGURATION ---
SYMBOLS = [
    "EURUSD", "GBPUSD", "USDJPY", "USDCHF", "USDCAD", "AUDUSD", "NZDUSD",
    "EURGBP", "EURJPY", "GBPJPY", "AUDJPY", "CADJPY", "CHFJPY"
]
MAGIC_NUMBER = 123456

class LiveTrader:
    def __init__(self):
        self.brain = AIBrain()
        
    def connect(self):
        if not mt5.initialize():
            print(f"MT5 Init Failed: {mt5.last_error()}")
            return False
        # Login is automatic if terminal is open
        print(f"[INIT] Connected to Account: {mt5.account_info().login}, Server: {mt5.account_info().server}")
        return True

    def fetch_data(self, symbol, timeframe, num_bars=1000):
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, num_bars)
        if rates is None or len(rates) == 0:
            return None
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        return df

    def has_open_position(self, symbol):
        positions = mt5.positions_get(symbol=symbol)
        return positions is not None and len(positions) > 0

    def execute_trade_from_brain(self, symbol, decision):
        if self.has_open_position(symbol):
            print(f"Skipping {symbol}: Position already open.")
            return

        pattern = decision['pattern']
        action = mt5.TRADE_ACTION_DEAL
        order_type = mt5.ORDER_TYPE_BUY if pattern.direction == 'bullish' else mt5.ORDER_TYPE_SELL
        
        # Get current tick for accurate pricing
        tick = mt5.symbol_info_tick(symbol)
        price = tick.ask if pattern.direction == 'bullish' else tick.bid
        
        sl = decision['sl']
        tp = decision['tp']
        lot = decision['lot']
        
        print(f"[RISK] {symbol} | Lot: {lot} | SL: {sl:.5f} | TP: {tp:.5f} | Reason: {decision['reason']}")

        request = {
            "action": action,
            "symbol": symbol,
            "volume": lot,
            "type": order_type,
            "price": price,
            "sl": sl,
            "tp": tp,
            "deviation": 20,
            "magic": MAGIC_NUMBER,
            "comment": f"Brain-Conf{decision['confidence']:.2f}",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            print(f"[FAIL] Order failed for {symbol}: {result.comment} ({result.retcode})")
            # Log failure?
        else:
            print(f"[EXEC] Order Sent! {symbol} {pattern.direction.upper()} @ {price}")
            # Log successful trade start
            self.brain.memory.log_trade(symbol, "BUY" if pattern.direction == 'bullish' else "SELL")

    def run(self):
        print(f"--- AI Live Trader Started (Symbols: {len(SYMBOLS)}) ---\n")
        print("Brain Activated: Scanning Market Structure & Context...\n")
        
        while True:
            for symbol in SYMBOLS:
                try:
                    # Fetch Context Data
                    h1 = self.fetch_data(symbol, mt5.TIMEFRAME_H1)
                    if h1 is None: continue
                    
                    h4 = self.fetch_data(symbol, mt5.TIMEFRAME_H4) # For Trend/Strength
                    d1 = self.fetch_data(symbol, mt5.TIMEFRAME_D1) # For Global Trend
                    
                    # Pass basic info
                    acc_info = mt5.account_info()
                    sym_info = mt5.symbol_info(symbol)
                    
                    if not sym_info or not acc_info: continue
                    
                    decision = self.brain.think(symbol, h1, h4, d1, acc_info, sym_info)
                    
                    if decision['decision'] == 'TRADE':
                        print(f"[SIGNAL] {symbol}: AI approved {decision['pattern'].name} ({decision['reason']})")
                        self.execute_trade_from_brain(symbol, decision)
                    elif decision['decision'] == 'REJECT':
                        pass
                        # print(f"[FILTER] {symbol}: {decision['reason']}") # Too verbose?
                    elif decision['decision'] == 'WAIT':
                        pass
                    
                except Exception as e:
                    print(f"[ERR] Error processing {symbol}: {e}")
                    import traceback
                    traceback.print_exc()
        
            print(".", end="", flush=True) # Heartbeat
            time.sleep(10) # Fast scan

    def shutdown(self):
        mt5.shutdown()

if __name__ == "__main__":
    bot = LiveTrader()
    if bot.connect():
        try:
            bot.run()
        except KeyboardInterrupt:
            print("\nStopping...")
        finally:
            bot.shutdown()
