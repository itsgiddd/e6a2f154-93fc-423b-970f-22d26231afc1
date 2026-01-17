import MetaTrader5 as mt5

class AdaptiveRiskManager:
    """
    Calculates position size dynamically based on:
    - Account Money Management Curve (Grow aggressively when small)
    - Trade Confidence (Confluence Score)
    - Market Volatility (SL Distance)
    """
    
    def calculate_lot_size(self, symbol: str, price: float, sl_price: float, confidence_score: int, account_info, symbol_info) -> float:
        balance = account_info.balance
        
        # 1. Base Risk Percentage (Conservative by default)
        risk_pct = 0.02 # Default 2%
        if balance < 1000: risk_pct = 0.04
        elif balance < 5000: risk_pct = 0.03
        
        # 2. Adjust for Confidence
        # Score 6/6 -> 100% of risk
        # Score 4/6 -> 50% of risk
        if confidence_score >= 6:
            risk_mult = 1.0
        elif confidence_score == 5:
            risk_mult = 0.8
        elif confidence_score == 4:
            risk_mult = 0.5
        else:
            return 0.0 # Should be filtered before here, but safety first
            
        final_risk_money = balance * risk_pct * risk_mult
        
        # 3. Calculate Lot based on SL Distance
        sl_points = abs(price - sl_price) / symbol_info.point
        if sl_points == 0: return 0.01
        
        # Value of 1 lot per point
        # trade_tick_value is profit for 1 tick (point) ?
        # Standard: Profit = (Close - Open) * Volume * ContractSize
        # Loss = SL_Dist * Volume * ContractSize
        # Risk = SL_Dist * Volume * ContractSize
        # Volume = Risk / (SL_Dist * ContractSize)
        
        # Approximation for Forex using tick_value (usually accurate for standard accounts)
        tick_value = symbol_info.trade_tick_value 
        if tick_value <= 0: tick_value = 1.0 # Fallback
        
        loss_per_lot = sl_points * tick_value
        if loss_per_lot <= 0: return 0.01
        
        lot = final_risk_money / loss_per_lot
        
        # 4. Normalize Lot Size
        step = symbol_info.volume_step
        min_vol = symbol_info.volume_min
        max_vol = symbol_info.volume_max
        
        lot = round(lot / step) * step
        if lot < min_vol: lot = min_vol # Or 0 if strictly adhering to risk? User is aggressive.
        if lot > max_vol: lot = max_vol
        
        # 5. Margin Check
        margin = mt5.order_calc_margin(mt5.TRADE_ACTION_DEAL, symbol, lot, price)
        if margin > account_info.margin_free * 0.95:
             lot = lot * (account_info.margin_free * 0.95 / margin)
             lot = round(lot / step) * step
             
        return lot
