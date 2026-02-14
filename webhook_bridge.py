"""
TradingView -> MT5 Webhook Bridge (Email-based, Free Plan)

Monitors a Gmail inbox for TradingView alert emails containing JSON trade signals.
Parses the signal and executes trades on MT5 via the MetaTrader5 Python API.

Usage:
    python webhook_bridge.py

Requires:
    - webhook_config.json with email credentials and trading parameters
    - MT5 terminal running and logged in
    - TradingView alerts set to email notification with JSON alert() messages
"""

import imaplib
import email
import json
import re
import time
import logging
import signal
import sys
import os
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, Dict, Set

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(LOG_DIR, "webhook_bridge.log"), encoding="utf-8"),
    ],
)
log = logging.getLogger("webhook_bridge")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class EmailConfig:
    imap_server: str = "imap.gmail.com"
    imap_port: int = 993
    address: str = ""
    app_password: str = ""
    sender_filter: str = "tradingview.com"


@dataclass
class TradingConfig:
    risk_pct: float = 0.30
    default_lot: float = 0.40
    allowed_symbols: list = field(default_factory=lambda: [
        "EURUSD", "GBPUSD", "USDJPY", "AUDUSD",
        "USDCAD", "NZDUSD", "EURJPY", "GBPJPY",
    ])
    magic_number: int = 234567
    max_concurrent: int = 5


def load_config(path: str = None) -> tuple:
    """Load config from webhook_config.json."""
    if path is None:
        path = os.path.join(os.path.dirname(__file__), "webhook_config.json")

    if not os.path.exists(path):
        log.error(f"Config file not found: {path}")
        log.error("Create webhook_config.json with your email credentials. See template.")
        sys.exit(1)

    with open(path, "r") as f:
        cfg = json.load(f)

    email_cfg = EmailConfig(
        imap_server=cfg.get("email", {}).get("imap_server", "imap.gmail.com"),
        imap_port=cfg.get("email", {}).get("imap_port", 993),
        address=cfg.get("email", {}).get("address", ""),
        app_password=cfg.get("email", {}).get("app_password", ""),
    )

    trading_cfg = TradingConfig(
        risk_pct=cfg.get("trading", {}).get("risk_pct", 0.30),
        default_lot=cfg.get("trading", {}).get("default_lot", 0.40),
        allowed_symbols=cfg.get("trading", {}).get("allowed_symbols", [
            "EURUSD", "GBPUSD", "USDJPY", "AUDUSD",
            "USDCAD", "NZDUSD", "EURJPY", "GBPJPY",
        ]),
        magic_number=cfg.get("trading", {}).get("magic_number", 234567),
    )

    poll_interval = cfg.get("poll_interval_seconds", 10)

    return email_cfg, trading_cfg, poll_interval


# ---------------------------------------------------------------------------
# Signal Deduplication
# ---------------------------------------------------------------------------
class SignalDeduplicator:
    """Prevents executing the same signal twice (e.g., email retry)."""

    def __init__(self, ttl_seconds: int = 14400):
        self._cache: Dict[str, datetime] = {}
        self._ttl = timedelta(seconds=ttl_seconds)

    def _key(self, symbol: str, action: str, entry: float) -> str:
        return f"{symbol}:{action}:{round(entry, 4)}"

    def is_duplicate(self, symbol: str, action: str, entry: float) -> bool:
        self._cleanup()
        key = self._key(symbol, action, entry)
        return key in self._cache

    def record(self, symbol: str, action: str, entry: float):
        key = self._key(symbol, action, entry)
        self._cache[key] = datetime.now()

    def _cleanup(self):
        now = datetime.now()
        expired = [k for k, ts in self._cache.items() if now - ts > self._ttl]
        for k in expired:
            del self._cache[k]


# ---------------------------------------------------------------------------
# Webhook HTTP Server (receives POST from TradingView directly)
# ---------------------------------------------------------------------------
_webhook_bridge_ref = None  # set by WebhookBridge.start()


class _WebhookHandler(BaseHTTPRequestHandler):
    """Handles incoming POST /webhook with JSON signal from TradingView."""

    def do_POST(self):
        try:
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length).decode("utf-8", errors="replace")
            log.info(f"[WEBHOOK] Received POST: {body[:500]}")

            signal_data = json.loads(body)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"status":"ok"}')

            # Execute on bridge
            if _webhook_bridge_ref:
                _webhook_bridge_ref._execute_signal(signal_data)

        except json.JSONDecodeError as e:
            log.warning(f"[WEBHOOK] Bad JSON: {e}")
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b'{"error":"bad json"}')
        except Exception as e:
            log.error(f"[WEBHOOK] Error: {e}", exc_info=True)
            self.send_response(500)
            self.end_headers()
            self.wfile.write(b'{"error":"internal"}')

    def do_GET(self):
        """Health check endpoint."""
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.end_headers()
        self.wfile.write(b"ACi Webhook Bridge OK")

    def log_message(self, format, *args):
        """Suppress default HTTP logging — we use our own."""
        pass


# ---------------------------------------------------------------------------
# Webhook Bridge
# ---------------------------------------------------------------------------
class WebhookBridge:

    def __init__(self, email_cfg: EmailConfig, trading_cfg: TradingConfig,
                 poll_interval: int = 10, webhook_port: int = 5555):
        self.email_cfg = email_cfg
        self.trading_cfg = trading_cfg
        self.poll_interval = poll_interval
        self.webhook_port = webhook_port
        self._running = False
        self._mail = None
        self._dedup = SignalDeduplicator()
        self._mt5_initialized = False
        self._http_server = None

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def start(self):
        """Start the bridge -- poll for emails, execute trades."""
        global _webhook_bridge_ref
        _webhook_bridge_ref = self

        log.info("=" * 60)
        log.info("TradingView -> MT5 Webhook Bridge starting...")
        log.info(f"  Email: {self.email_cfg.address}")
        log.info(f"  Webhook: http://localhost:{self.webhook_port}/webhook")
        log.info(f"  Risk: {self.trading_cfg.risk_pct:.0%}")
        log.info(f"  Default lot: {self.trading_cfg.default_lot}")
        syms = ', '.join(self.trading_cfg.allowed_symbols) if self.trading_cfg.allowed_symbols else 'ANY'
        log.info(f"  Symbols: {syms}")
        log.info(f"  Poll interval: {self.poll_interval}s")
        log.info(f"  Magic: {self.trading_cfg.magic_number}")
        log.info("=" * 60)

        # Connect MT5
        if not self._init_mt5():
            log.error("Failed to connect MT5. Make sure MT5 is running.")
            return

        # Start webhook HTTP server in background
        try:
            self._http_server = HTTPServer(("0.0.0.0", self.webhook_port), _WebhookHandler)
            t = threading.Thread(target=self._http_server.serve_forever, daemon=True)
            t.start()
            log.info(f"Webhook server listening on port {self.webhook_port}")
        except OSError as e:
            log.warning(f"Could not start webhook server on port {self.webhook_port}: {e}")
            log.warning("Webhook mode disabled — email mode still active")

        self._running = True
        reconnect_delay = 5

        while self._running:
            try:
                # Connect IMAP
                if self._mail is None:
                    self._connect_imap()
                    reconnect_delay = 5  # Reset on success

                # Poll for new emails
                count = self._poll_new_emails()
                if count > 0:
                    log.info(f"Processed {count} signal(s)")

                # Wait
                for _ in range(self.poll_interval):
                    if not self._running:
                        break
                    time.sleep(1)

            except (imaplib.IMAP4.abort, imaplib.IMAP4.error, OSError, ConnectionError) as e:
                log.warning(f"IMAP connection error: {e}")
                self._mail = None
                log.info(f"Reconnecting in {reconnect_delay}s...")
                time.sleep(reconnect_delay)
                reconnect_delay = min(reconnect_delay * 2, 60)

            except Exception as e:
                log.error(f"Unexpected error: {e}", exc_info=True)
                time.sleep(5)

        self.stop()

    def stop(self):
        """Graceful shutdown."""
        self._running = False
        if self._http_server:
            try:
                self._http_server.shutdown()
            except Exception:
                pass
            self._http_server = None
        if self._mail:
            try:
                self._mail.logout()
            except Exception:
                pass
            self._mail = None
        if self._mt5_initialized:
            try:
                import MetaTrader5 as mt5
                mt5.shutdown()
            except Exception:
                pass
        log.info("Webhook bridge stopped.")

    # ------------------------------------------------------------------
    # IMAP
    # ------------------------------------------------------------------
    def _connect_imap(self):
        """Connect to IMAP server."""
        log.info(f"Connecting to {self.email_cfg.imap_server}...")
        self._mail = imaplib.IMAP4_SSL(
            self.email_cfg.imap_server,
            self.email_cfg.imap_port,
        )
        self._mail.login(self.email_cfg.address, self.email_cfg.app_password)
        self._mail.select("INBOX")
        log.info("IMAP connected.")

    def _poll_new_emails(self) -> int:
        """Check for new TradingView emails and process them."""
        if self._mail is None:
            return 0

        # Refresh mailbox
        self._mail.select("INBOX")

        # Search for unread emails with "Alert: ZeroPoint" in subject
        # This avoids matching Google "Security alert" emails
        search_criteria = '(UNSEEN SUBJECT "Alert: ZeroPoint")'
        status, msg_nums = self._mail.search(None, search_criteria)

        if status != "OK" or not msg_nums[0]:
            return 0

        count = 0
        for num in msg_nums[0].split():
            try:
                _, data = self._mail.fetch(num, "(RFC822)")
                if not data or not data[0]:
                    continue
                raw_email = data[0][1]
                msg = email.message_from_bytes(raw_email)

                signal = self._process_email(msg)
                if signal:
                    self._execute_signal(signal)
                    count += 1

            except Exception as e:
                log.error(f"Error processing email {num}: {e}", exc_info=True)

        return count

    def _process_email(self, msg) -> Optional[dict]:
        """Extract and parse signal from email."""
        subject = msg.get("Subject", "")
        log.info(f"Processing email: {subject}")

        # Extract body text
        body = ""
        if msg.is_multipart():
            for part in msg.walk():
                content_type = part.get_content_type()
                if content_type == "text/plain":
                    body = part.get_payload(decode=True).decode("utf-8", errors="replace")
                    break
                elif content_type == "text/html" and not body:
                    html = part.get_payload(decode=True).decode("utf-8", errors="replace")
                    # Strip HTML tags for JSON extraction
                    body = re.sub(r"<[^>]+>", " ", html)
        else:
            payload = msg.get_payload(decode=True)
            if payload:
                body = payload.decode("utf-8", errors="replace")
                if msg.get_content_type() == "text/html":
                    body = re.sub(r"<[^>]+>", " ", body)

        if not body:
            log.warning("Empty email body")
            return None

        return self._parse_tv_alert(body)

    def _parse_tv_alert(self, body: str) -> Optional[dict]:
        """Find and parse JSON signal from email body text."""
        # Look for JSON object containing "action":"BUY" or "action":"SELL"
        pattern = r'\{[^{}]*"action"\s*:\s*"(?:BUY|SELL)"[^{}]*\}'
        match = re.search(pattern, body)

        if not match:
            # Also try without quotes around action value
            pattern2 = r'\{[^{}]*action[^{}]*(?:BUY|SELL)[^{}]*\}'
            match = re.search(pattern2, body)

        if not match:
            log.debug(f"No JSON signal found in email body")
            return None

        try:
            signal = json.loads(match.group())
            log.info(f"Parsed signal: {json.dumps(signal, indent=2)}")
            return signal
        except json.JSONDecodeError as e:
            log.warning(f"JSON parse error: {e} | raw: {match.group()[:200]}")
            return None

    # ------------------------------------------------------------------
    # Signal validation
    # ------------------------------------------------------------------
    def _validate_signal(self, signal: dict) -> bool:
        """Validate signal has required fields and makes sense."""
        required = ["action", "symbol", "entry", "sl", "tp1"]
        for field_name in required:
            if field_name not in signal:
                log.warning(f"Missing field: {field_name}")
                return False

        action = signal["action"].upper()
        if action not in ("BUY", "SELL"):
            log.warning(f"Invalid action: {action}")
            return False

        # Normalize
        signal["action"] = action
        symbol = signal["symbol"].upper().replace(".", "").replace("#", "")
        signal["symbol"] = symbol

        # Check allowed symbols (empty list = accept any)
        if self.trading_cfg.allowed_symbols and symbol not in self.trading_cfg.allowed_symbols:
            log.info(f"Symbol {symbol} not in allowed list, skipping")
            return False

        entry = float(signal["entry"])
        sl = float(signal["sl"])
        tp1 = float(signal["tp1"])

        # SL sanity: must be on correct side
        if action == "BUY" and sl >= entry:
            log.warning(f"BUY but SL ({sl}) >= entry ({entry})")
            return False
        if action == "SELL" and sl <= entry:
            log.warning(f"SELL but SL ({sl}) <= entry ({entry})")
            return False

        # TP sanity: must be on correct side
        if action == "BUY" and tp1 <= entry:
            log.warning(f"BUY but TP1 ({tp1}) <= entry ({entry})")
            return False
        if action == "SELL" and tp1 >= entry:
            log.warning(f"SELL but TP1 ({tp1}) >= entry ({entry})")
            return False

        # R:R sanity
        sl_dist = abs(entry - sl)
        tp_dist = abs(tp1 - entry)
        if sl_dist > 0:
            rr = tp_dist / sl_dist
            if rr < 0.3:
                log.warning(f"R:R too low: {rr:.2f}")
                return False

        return True

    # ------------------------------------------------------------------
    # MT5 Integration
    # ------------------------------------------------------------------
    def _init_mt5(self) -> bool:
        """Initialize MT5 connection."""
        try:
            import MetaTrader5 as mt5
            if not mt5.initialize():
                log.error(f"MT5 initialize failed: {mt5.last_error()}")
                return False
            acct = mt5.account_info()
            if acct is None:
                log.error("Cannot read MT5 account info")
                mt5.shutdown()
                return False
            log.info(f"MT5 connected: Account {acct.login} | Balance: ${acct.balance:.2f}")
            self._mt5_initialized = True
            return True
        except ImportError:
            log.error("MetaTrader5 package not installed: pip install MetaTrader5")
            return False
        except Exception as e:
            log.error(f"MT5 init error: {e}")
            return False

    def _resolve_symbol(self, ticker: str) -> Optional[str]:
        """Map TradingView ticker to MT5 broker symbol."""
        import MetaTrader5 as mt5

        # Try exact match first
        candidates = [
            ticker,
            ticker + ".raw",
            ticker + "m",
            ticker + ".a",
            ticker + ".e",
            ticker[:6],  # Trim suffix if any
        ]

        for candidate in candidates:
            info = mt5.symbol_info(candidate)
            if info is not None:
                mt5.symbol_select(candidate, True)
                return candidate

        log.warning(f"Cannot resolve symbol: {ticker}")
        return None

    def _check_existing_position(self, symbol: str) -> bool:
        """Return True if we already have a position on this symbol."""
        import MetaTrader5 as mt5

        positions = mt5.positions_get()
        if not positions:
            return False

        norm = symbol.upper().replace(".", "").replace("#", "")
        for pos in positions:
            pos_norm = pos.symbol.upper().replace(".", "").replace("#", "")
            if pos_norm == norm:
                return True
        return False

    def _calc_lot_size(self, signal: dict, sym_info) -> float:
        """Calculate lot size based on risk percentage."""
        import MetaTrader5 as mt5

        try:
            acct = mt5.account_info()
            if acct is None:
                return self.trading_cfg.default_lot

            balance = acct.balance
            risk_amount = balance * self.trading_cfg.risk_pct

            point = sym_info.point
            tick_size = sym_info.trade_tick_size or point
            tick_value = sym_info.trade_tick_value
            if tick_value <= 0:
                tick_value = sym_info.trade_contract_size * tick_size

            sl_distance = abs(float(signal["entry"]) - float(signal["sl"]))
            sl_ticks = sl_distance / tick_size if tick_size > 0 else 0
            loss_per_lot = sl_ticks * tick_value

            if loss_per_lot <= 0:
                return self.trading_cfg.default_lot

            lot = risk_amount / loss_per_lot
            vol_step = sym_info.volume_step
            lot = round(lot / vol_step) * vol_step
            lot = max(sym_info.volume_min, min(lot, sym_info.volume_max))

            # Conservative cap by balance (same as original profitable system)
            cap_table = [
                (500, 0.10), (1000, 0.20), (3000, 0.50),
                (5000, 1.00), (10000, 2.00), (50000, 5.00),
                (float('inf'), 10.00),
            ]
            for threshold, max_lot in cap_table:
                if balance <= threshold:
                    lot = min(lot, max_lot)
                    break

            log.info(f"  Lot calc: balance=${balance:.0f} risk=${risk_amount:.0f} lot={lot:.2f}")
            return lot

        except Exception as e:
            log.error(f"Lot calc error: {e}, using default {self.trading_cfg.default_lot}")
            return self.trading_cfg.default_lot

    def _place_trade(self, signal: dict, lot: float, sym_info) -> bool:
        """Place trade on MT5 -- same pattern as original profitable app."""
        import MetaTrader5 as mt5

        try:
            action = signal["action"]
            if action == "BUY":
                order_type = mt5.ORDER_TYPE_BUY
                price = sym_info.ask
            else:
                order_type = mt5.ORDER_TYPE_SELL
                price = sym_info.bid

            digits = sym_info.digits
            sl = round(float(signal["sl"]), digits)
            tp = round(float(signal["tp1"]), digits)

            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": sym_info.name,
                "volume": lot,
                "type": order_type,
                "price": price,
                "sl": sl,
                "tp": tp,
                "deviation": 20,
                "magic": self.trading_cfg.magic_number,
                "comment": f"TV-ZP-{action}",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_FOK,
            }

            # Try fill modes (FOK -> IOC -> RETURN)
            for fill_mode in [mt5.ORDER_FILLING_FOK, mt5.ORDER_FILLING_IOC, mt5.ORDER_FILLING_RETURN]:
                request["type_filling"] = fill_mode
                result = mt5.order_send(request)
                if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                    rr = signal.get("rr", "?")
                    log.info(
                        f"  TRADE PLACED: {action} {sym_info.name} "
                        f"{lot:.2f}L @ {price:.5f} | SL={sl} TP={tp} | R:R={rr}"
                    )
                    return True

            rc = result.retcode if result else "?"
            log.error(f"  Trade FAILED: {sym_info.name} retcode={rc}")
            return False

        except Exception as e:
            log.error(f"  Trade error: {e}", exc_info=True)
            return False

    # ------------------------------------------------------------------
    # Execute signal pipeline
    # ------------------------------------------------------------------
    def _execute_signal(self, signal: dict):
        """Full pipeline: validate -> dedup -> resolve -> check position -> lot -> place."""
        import MetaTrader5 as mt5

        # Validate
        if not self._validate_signal(signal):
            return

        symbol = signal["symbol"]
        action = signal["action"]
        entry = float(signal["entry"])

        # Dedup
        if self._dedup.is_duplicate(symbol, action, entry):
            log.info(f"Duplicate signal for {symbol} {action}, skipping")
            return

        # Check existing position
        if self._check_existing_position(symbol):
            log.info(f"Already have position on {symbol}, skipping")
            return

        # Check max concurrent
        positions = mt5.positions_get()
        if positions and len(positions) >= self.trading_cfg.max_concurrent:
            log.info(f"Max concurrent positions ({self.trading_cfg.max_concurrent}) reached, skipping")
            return

        # Resolve symbol
        resolved = self._resolve_symbol(symbol)
        if resolved is None:
            return

        sym_info = mt5.symbol_info(resolved)
        if sym_info is None:
            log.error(f"Cannot get info for {resolved}")
            return

        # Staleness check — skip if price ran too far from signal entry
        current_price = sym_info.ask if action == "BUY" else sym_info.bid
        sl_distance = abs(entry - float(signal["sl"]))
        price_drift = abs(current_price - entry)
        max_drift = sl_distance * 0.50  # allow up to 50% of SL distance

        if sl_distance > 0 and price_drift > max_drift:
            drift_pct = (price_drift / sl_distance) * 100
            log.info(
                f"Signal too stale for {symbol}: price drifted {price_drift:.5f} "
                f"({drift_pct:.0f}% of SL distance) from entry {entry:.5f}, "
                f"current {current_price:.5f} — skipping"
            )
            return

        if price_drift > 0:
            log.info(f"  Price drift: {price_drift:.5f} from signal entry (within tolerance)")

        # Calculate lot
        lot = self._calc_lot_size(signal, sym_info)
        if lot <= 0:
            log.warning(f"Lot size 0 for {symbol}, skipping")
            return

        # Place trade
        log.info(f"EXECUTING: {action} {symbol} (resolved: {resolved})")
        success = self._place_trade(signal, lot, sym_info)

        if success:
            self._dedup.record(symbol, action, entry)
            log.info(f"Signal executed successfully: {action} {symbol}")
        else:
            log.error(f"Failed to execute: {action} {symbol}")


# ---------------------------------------------------------------------------
# Setup Wizard
# ---------------------------------------------------------------------------
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "webhook_config.json")


def _banner():
    print()
    print("=" * 60)
    print("  TradingView -> MT5 Webhook Bridge")
    print("  ZeroPoint PRO Email Alert -> Auto Trade")
    print("=" * 60)
    print()


def _needs_setup() -> bool:
    """Return True if config is missing or has placeholder values."""
    if not os.path.exists(CONFIG_PATH):
        return True
    try:
        with open(CONFIG_PATH, "r") as f:
            cfg = json.load(f)
        addr = cfg.get("email", {}).get("address", "")
        pw = cfg.get("email", {}).get("app_password", "")
        if not addr or not pw or "YOUR_" in addr or "YOUR_" in pw:
            return True
    except Exception:
        return True
    return False


def _setup_wizard():
    """Interactive first-time setup."""
    _banner()
    print("  FIRST TIME SETUP")
    print("  Let's get everything connected.\n")

    # ── Step 1: Gmail ──
    print("-" * 60)
    print("  STEP 1: Gmail Configuration")
    print("-" * 60)
    print()
    print("  TradingView will send alert emails to your Gmail.")
    print("  This bridge reads those emails and places trades on MT5.")
    print()
    print("  You need a Gmail account with:")
    print("    1. 2-Factor Authentication enabled")
    print("    2. An App Password generated")
    print("    3. IMAP enabled (Gmail Settings -> Forwarding and POP/IMAP)")
    print()
    print("  To create an App Password:")
    print("    Google Account -> Security -> 2-Step Verification -> App Passwords")
    print("    -> Select 'Mail' -> Generate -> Copy the 16-character password")
    print()

    email_addr = input("  Gmail address: ").strip()
    app_password = input("  App password (xxxx xxxx xxxx xxxx): ").strip()
    print()

    # Test IMAP connection
    print("  Testing Gmail connection...", end=" ", flush=True)
    try:
        import imaplib as _imap
        mail = _imap.IMAP4_SSL("imap.gmail.com", 993)
        mail.login(email_addr, app_password)
        mail.select("INBOX")
        mail.logout()
        print("SUCCESS [OK]")
    except Exception as e:
        print(f"FAILED [FAIL]")
        print(f"\n  Error: {e}")
        print("\n  Check that:")
        print("    - Email address is correct")
        print("    - App password is correct (not your regular password)")
        print("    - IMAP is enabled in Gmail settings")
        print("    - 2FA is enabled on the Google account")
        retry = input("\n  Try again? (y/n): ").strip().lower()
        if retry == "y":
            return _setup_wizard()
        sys.exit(1)

    # ── Step 2: MT5 ──
    print()
    print("-" * 60)
    print("  STEP 2: MetaTrader 5 Connection")
    print("-" * 60)
    print()
    print("  Testing MT5 connection...", end=" ", flush=True)
    try:
        import MetaTrader5 as mt5
        if not mt5.initialize():
            raise RuntimeError(f"MT5 initialize failed: {mt5.last_error()}")
        acct = mt5.account_info()
        if acct is None:
            raise RuntimeError("Cannot read account info")
        print("SUCCESS [OK]")
        print(f"  Account: {acct.login}")
        print(f"  Balance: ${acct.balance:.2f}")
        print(f"  Broker:  {acct.company}")
        mt5.shutdown()
    except ImportError:
        print("FAILED [FAIL]")
        print("\n  MetaTrader5 package not installed.")
        print("  Run: pip install MetaTrader5")
        sys.exit(1)
    except Exception as e:
        print(f"FAILED [FAIL]")
        print(f"\n  Error: {e}")
        print("  Make sure MT5 terminal is running and logged in.")
        retry = input("\n  Try again? (y/n): ").strip().lower()
        if retry == "y":
            return _setup_wizard()
        sys.exit(1)

    # ── Step 3: Trading Settings ──
    print()
    print("-" * 60)
    print("  STEP 3: Trading Settings")
    print("-" * 60)
    print()

    risk_input = input("  Risk per trade % (default 30): ").strip()
    risk_pct = float(risk_input) / 100 if risk_input else 0.30

    symbols_input = input("  Symbols (comma-separated, or press Enter for all 8): ").strip()
    if symbols_input:
        allowed_symbols = [s.strip().upper() for s in symbols_input.split(",")]
    else:
        allowed_symbols = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "NZDUSD", "EURJPY", "GBPJPY"]

    max_trades_input = input("  Max concurrent trades (default 5): ").strip()
    max_concurrent = int(max_trades_input) if max_trades_input else 5

    # ── Save Config ──
    cfg = {
        "email": {
            "imap_server": "imap.gmail.com",
            "imap_port": 993,
            "address": email_addr,
            "app_password": app_password,
        },
        "trading": {
            "risk_pct": risk_pct,
            "default_lot": 0.40,
            "allowed_symbols": allowed_symbols,
            "magic_number": 234567,
            "max_concurrent": max_concurrent,
        },
        "poll_interval_seconds": 10,
    }

    with open(CONFIG_PATH, "w") as f:
        json.dump(cfg, f, indent=4)

    print()
    print(f"  Config saved to webhook_config.json [OK]")

    # ── Step 4: TradingView Instructions ──
    print()
    print("-" * 60)
    print("  STEP 4: TradingView Setup (do this on TradingView)")
    print("-" * 60)
    print()
    print("  1. Open your ZeroPoint PRO indicator in Pine Script editor")
    print()
    print("  2. Add these 2 lines BEFORE the alertcondition() calls:")
    print()
    print("     if buySignal")
    print("         alert('{\"action\":\"BUY\",\"symbol\":\"' + syminfo.ticker + ...")
    print("     if sellSignal")
    print("         alert('{\"action\":\"SELL\",\"symbol\":\"' + syminfo.ticker + ...")
    print()
    print("     (Full lines are in zeropoint_multi_scanner.pine)")
    print()
    print("  3. Save the indicator")
    print()
    print("  4. Create Alert:")
    print("     - Condition: Your indicator -> 'Any alert() function call'")
    print(f"     - Notifications: check Email -> {email_addr}")
    print("     - Alert name: 'ZeroPoint USDCAD' (must contain 'ZeroPoint')")
    print("     - Expiration: Open-ended")
    print()

    input("  Press Enter when TradingView is set up (or to skip for now)...")
    print()
    print("=" * 60)
    print("  SETUP COMPLETE -- Starting bridge...")
    print("=" * 60)
    print()


def _preflight_check(email_cfg, trading_cfg):
    """Quick connection check on every launch (not first time)."""
    _banner()
    print("  Running preflight checks...\n")
    all_ok = True

    # Check Gmail
    print("  [1/3] Gmail IMAP...", end=" ", flush=True)
    try:
        mail = imaplib.IMAP4_SSL(email_cfg.imap_server, email_cfg.imap_port)
        mail.login(email_cfg.address, email_cfg.app_password)
        mail.logout()
        print(f"OK [OK]  ({email_cfg.address})")
    except Exception as e:
        print(f"FAILED [FAIL]  ({e})")
        all_ok = False

    # Check MT5
    print("  [2/3] MetaTrader 5...", end=" ", flush=True)
    try:
        import MetaTrader5 as mt5
        if not mt5.initialize():
            raise RuntimeError(mt5.last_error())
        acct = mt5.account_info()
        print(f"OK [OK]  (Account {acct.login}, ${acct.balance:.2f})")
        mt5.shutdown()
    except Exception as e:
        print(f"FAILED [FAIL]  ({e})")
        all_ok = False

    # Check symbols
    print(f"  [3/3] Symbols...", end=" ", flush=True)
    try:
        import MetaTrader5 as mt5
        mt5.initialize()
        missing = []
        for sym in trading_cfg.allowed_symbols:
            info = mt5.symbol_info(sym)
            if info is None:
                missing.append(sym)
            else:
                mt5.symbol_select(sym, True)
        mt5.shutdown()
        if missing:
            print(f"WARNING -- missing: {', '.join(missing)}")
        else:
            print(f"OK [OK]  ({len(trading_cfg.allowed_symbols)} symbols)")
    except Exception as e:
        print(f"FAILED [FAIL]  ({e})")
        all_ok = False

    print()
    if not all_ok:
        print("  Some checks failed. Fix the issues above or run setup again.")
        choice = input("  Continue anyway? (y/n): ").strip().lower()
        if choice != "y":
            sys.exit(1)

    print("  All checks passed -- starting bridge...\n")
    print("  Listening for TradingView alert emails...")
    print("  Press Ctrl+C to stop.\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    # First-time setup or regular launch
    if _needs_setup():
        _setup_wizard()

    email_cfg, trading_cfg, poll_interval = load_config()

    if not email_cfg.address or not email_cfg.app_password:
        log.error("Email address and app_password must be set. Run setup again.")
        sys.exit(1)

    # Preflight checks on every launch
    _preflight_check(email_cfg, trading_cfg)

    bridge = WebhookBridge(email_cfg, trading_cfg, poll_interval)

    # Handle Ctrl+C
    def _signal_handler(sig, frame):
        log.info("\nShutting down...")
        bridge.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    bridge.start()


if __name__ == "__main__":
    main()
