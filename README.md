# Trading Brain Conflict Resolutions

This repository includes updates to address seven conflicts in the AI brain decision pipeline. The fixes focus on preventing runtime mismatches, invalid market inputs, and unsafe trade calculations.

## Resolved Conflicts (7)
1. **Empty input protection**: Rejects decisions when H1/H4/D1 data is empty or missing entirely.  
2. **Required column validation**: Blocks execution if OHLC columns are missing in any timeframe.  
3. **Time ordering mismatch**: Ensures market data is sorted by time before analysis to avoid stale context reads.  
4. **Symbol info gaps**: Stops processing when required symbol fields (point, lot sizing attributes, tick value) are missing.  
5. **Stop-loss fallback**: Adds a safety SL based on recent swing highs/lows when patterns omit SL details.  
6. **Target distance fallback**: Establishes a measured move target using recent range or 2x risk if the pattern provides none.  
7. **Minimum history guard**: Returns a wait decision when there is insufficient bar history to compute context and patterns reliably.  
