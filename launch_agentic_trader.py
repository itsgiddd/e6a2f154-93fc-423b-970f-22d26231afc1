#!/usr/bin/env python3
"""
Headless launcher for the agentic neural trading system.

Starts:
  1. MT5 connection
  2. Neural model load
  3. TradingEngine with all 9 trained symbols
  4. AgenticOrchestrator (background self-learning daemon)
  5. Runs until Ctrl+C
"""

import logging
import os
import signal
import sys
import time

sys.path.append(os.path.join(os.path.dirname(__file__), "app"))

from app.mt5_connector import MT5Connector
from app.model_manager import NeuralModelManager
from app.trading_engine import TradingEngine
from agentic_orchestrator import AgenticOrchestrator

# All 9 symbols the model was trained on.
ALL_SYMBOLS = [
    "EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD",
    "NZDUSD", "EURJPY", "GBPJPY", "BTCUSD",
]

MODEL_PATH = "neural_model.pth"


def setup_logging():
    os.makedirs("logs", exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(name)-28s  %(levelname)-7s  %(message)s",
        handlers=[
            logging.FileHandler("logs/agentic_trader.log"),
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger("launcher")


def main():
    logger = setup_logging()

    print("=" * 60)
    print("  AGENTIC NEURAL TRADING SYSTEM")
    print("  9 Symbols | Self-Learning | Auto-Retrain")
    print("=" * 60)

    # 1. MT5
    logger.info("Connecting to MT5...")
    mt5 = MT5Connector()
    if not mt5.connect():
        logger.error("MT5 connection failed. Is MT5 running and logged in?")
        return 1
    logger.info("MT5 connected")

    # 2. Model
    logger.info(f"Loading neural model from {MODEL_PATH}...")
    model_mgr = NeuralModelManager()
    if not model_mgr.load_model(MODEL_PATH):
        logger.error(f"Failed to load model from {MODEL_PATH}")
        return 1
    logger.info("Neural model loaded")

    # 3. Trading engine
    logger.info(f"Starting trading engine with {len(ALL_SYMBOLS)} symbols...")
    engine = TradingEngine(
        mt5_connector=mt5,
        model_manager=model_mgr,
        risk_per_trade=0.30,
        confidence_threshold=0.65,
        trading_pairs=ALL_SYMBOLS,
        max_concurrent_positions=8,
    )

    # 4. Agentic orchestrator
    logger.info("Starting agentic orchestrator...")
    orchestrator = AgenticOrchestrator(
        model_manager=model_mgr,
        trading_engine=engine,
        model_path=MODEL_PATH,
        symbols=ALL_SYMBOLS,
    )
    engine.orchestrator = orchestrator
    orchestrator.start()

    # 5. Go live
    engine.start()
    logger.info("Trading engine started — all systems live")

    print()
    print("  LIVE — Press Ctrl+C to stop")
    print("=" * 60)

    # Graceful shutdown
    stop_flag = False

    def _shutdown(sig, frame):
        nonlocal stop_flag
        if stop_flag:
            return
        stop_flag = True
        print("\nShutting down...")
        orchestrator.stop()
        engine.stop()
        logger.info("Shutdown complete")

    signal.signal(signal.SIGINT, _shutdown)

    try:
        while not stop_flag:
            time.sleep(1)
    except KeyboardInterrupt:
        _shutdown(None, None)

    return 0


if __name__ == "__main__":
    sys.exit(main())
