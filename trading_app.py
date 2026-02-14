#!/usr/bin/env python3
"""
Neural Forex Trader — PySide6 Production UI
"""

import sys, os, threading, logging, json, time as _time
from pathlib import Path
from datetime import datetime

# Ensure app modules importable
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QLabel, QPushButton, QCheckBox, QLineEdit,
    QGroupBox, QTableWidget, QTableWidgetItem, QTextEdit,
    QHeaderView, QFrame, QAbstractItemView,
)
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QColor

# Global crash handler — writes to crash.log so we can debug after a crash
import traceback as _tb
def _global_exception_hook(exc_type, exc_value, exc_tb):
    crash_msg = ''.join(_tb.format_exception(exc_type, exc_value, exc_tb))
    try:
        with open('crash.log', 'a') as f:
            f.write(f"\n{'='*60}\n{datetime.now()}\n{crash_msg}\n")
    except Exception:
        pass
    sys.__excepthook__(exc_type, exc_value, exc_tb)
sys.excepthook = _global_exception_hook

# Safe imports — show clear errors instead of silent crashes
_IMPORT_ERRORS = []

try:
    import MetaTrader5 as _mt5_check
except ImportError:
    _IMPORT_ERRORS.append("MetaTrader5 not installed. Run: pip install MetaTrader5")

try:
    from app.mt5_connector import MT5Connector
    from app.config_manager import ConfigManager
except ImportError as e:
    _IMPORT_ERRORS.append(f"Core module import failed: {e}")

try:
    from agentic_orchestrator import AgenticOrchestrator
except ImportError:
    AgenticOrchestrator = None  # Optional — trading works without it

# ---------------------------------------------------------------------------
# Dark theme stylesheet
# ---------------------------------------------------------------------------
CLAUDE_STYLE = """
QMainWindow, QWidget {
    background-color: #F4F3EE;
    color: #141413;
    font-family: Georgia, Cambria, 'Times New Roman', serif;
    font-size: 14px;
}
QGroupBox {
    border: 1px solid #D5D3C8;
    border-radius: 12px;
    margin-top: 10px;
    padding-top: 16px;
    font-weight: bold;
    color: #3D3929;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 14px;
    padding: 0 8px;
}
QPushButton {
    background-color: #EEECE2;
    border: 1px solid #D5D3C8;
    border-radius: 8px;
    padding: 8px 20px;
    color: #3D3929;
    font-weight: bold;
    min-height: 28px;
}
QPushButton:hover {
    background-color: #E4E2D8;
    border-color: #C15F3C;
}
QPushButton:pressed {
    background-color: #D5D3C8;
}
QPushButton:disabled {
    background-color: #F4F3EE;
    color: #B0AEA5;
    border-color: #E8E6DC;
}
QPushButton#startBtn {
    background-color: #C15F3C;
    border: none;
    color: #FFFFFF;
}
QPushButton#startBtn:hover {
    background-color: #B5523C;
}
QPushButton#startBtn:disabled {
    background-color: #D5D3C8;
    color: #B0AEA5;
}
QPushButton#stopBtn {
    background-color: #E8E6DC;
    border: 1px solid #D5D3C8;
    color: #3D3929;
}
QPushButton#stopBtn:hover {
    background-color: #D5D3C8;
}
QPushButton#emergencyBtn {
    background-color: #C44444;
    border: none;
    color: #FFFFFF;
}
QPushButton#emergencyBtn:hover {
    background-color: #A83838;
}
QLineEdit {
    background-color: #EEECE2;
    border: 1px solid #D5D3C8;
    border-radius: 8px;
    padding: 6px 10px;
    color: #141413;
}
QLineEdit:focus {
    border-color: #C15F3C;
}
QCheckBox {
    spacing: 6px;
    color: #141413;
}
QCheckBox::indicator {
    width: 16px;
    height: 16px;
    border: 1px solid #B0AEA5;
    border-radius: 4px;
    background-color: #FAF9F5;
}
QCheckBox::indicator:checked {
    background-color: #C15F3C;
    border-color: #B5523C;
}
QTableWidget {
    background-color: #FAF9F5;
    border: 1px solid #D5D3C8;
    border-radius: 8px;
    gridline-color: #E8E6DC;
    selection-background-color: #E8D5C8;
    color: #141413;
}
QTableWidget::item {
    padding: 4px;
}
QHeaderView::section {
    background-color: #EEECE2;
    color: #3D3929;
    border: none;
    border-bottom: 1px solid #D5D3C8;
    padding: 6px;
    font-weight: bold;
}
QTextEdit {
    background-color: #FAF9F5;
    border: 1px solid #D5D3C8;
    border-radius: 8px;
    color: #3D3929;
    font-family: 'JetBrains Mono', Consolas, 'Courier New', monospace;
    font-size: 12px;
}
QLabel#statusLabel {
    font-size: 12px;
    padding: 2px 8px;
}
QLabel#balanceLabel {
    font-size: 16px;
    font-weight: bold;
    color: #C15F3C;
}
QFrame#separator {
    background-color: #D5D3C8;
    max-height: 1px;
}
"""

DARK_STYLE = """
QMainWindow, QWidget {
    background-color: #1A1815;
    color: #E8E6E3;
    font-family: Georgia, Cambria, 'Times New Roman', serif;
    font-size: 14px;
}
QGroupBox {
    border: 1px solid #3A352B;
    border-radius: 12px;
    margin-top: 10px;
    padding-top: 16px;
    font-weight: bold;
    color: #B5AFA5;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 14px;
    padding: 0 8px;
}
QPushButton {
    background-color: #2A251D;
    border: 1px solid #3A352B;
    border-radius: 8px;
    padding: 8px 20px;
    color: #E8E6E3;
    font-weight: bold;
    min-height: 28px;
}
QPushButton:hover {
    background-color: #342E24;
    border-color: #C15F3C;
}
QPushButton:pressed {
    background-color: #3A352B;
}
QPushButton:disabled {
    background-color: #1A1815;
    color: #4A453B;
    border-color: #2A251D;
}
QPushButton#startBtn {
    background-color: #C15F3C;
    border: none;
    color: #FFFFFF;
}
QPushButton#startBtn:hover {
    background-color: #B5523C;
}
QPushButton#startBtn:disabled {
    background-color: #2A251D;
    color: #4A453B;
}
QPushButton#stopBtn {
    background-color: #2A251D;
    border: 1px solid #3A352B;
    color: #E8E6E3;
}
QPushButton#stopBtn:hover {
    background-color: #342E24;
}
QPushButton#emergencyBtn {
    background-color: #C44444;
    border: none;
    color: #FFFFFF;
}
QPushButton#emergencyBtn:hover {
    background-color: #A83838;
}
QLineEdit {
    background-color: #201D18;
    border: 1px solid #3A352B;
    border-radius: 8px;
    padding: 6px 10px;
    color: #E8E6E3;
}
QLineEdit:focus {
    border-color: #C15F3C;
}
QCheckBox {
    spacing: 6px;
    color: #E8E6E3;
}
QCheckBox::indicator {
    width: 16px;
    height: 16px;
    border: 1px solid #4A453B;
    border-radius: 4px;
    background-color: #201D18;
}
QCheckBox::indicator:checked {
    background-color: #C15F3C;
    border-color: #B5523C;
}
QTableWidget {
    background-color: #201D18;
    border: 1px solid #3A352B;
    border-radius: 8px;
    gridline-color: #2A251D;
    selection-background-color: #3A352B;
    color: #E8E6E3;
}
QTableWidget::item {
    padding: 4px;
}
QHeaderView::section {
    background-color: #2A251D;
    color: #B5AFA5;
    border: none;
    border-bottom: 1px solid #3A352B;
    padding: 6px;
    font-weight: bold;
}
QTextEdit {
    background-color: #161411;
    border: 1px solid #3A352B;
    border-radius: 8px;
    color: #B5AFA5;
    font-family: 'JetBrains Mono', Consolas, 'Courier New', monospace;
    font-size: 12px;
}
QLabel#statusLabel {
    font-size: 12px;
    padding: 2px 8px;
}
QLabel#balanceLabel {
    font-size: 16px;
    font-weight: bold;
    color: #C15F3C;
}
QFrame#separator {
    background-color: #3A352B;
    max-height: 1px;
}
"""

ALL_PAIRS = [
    'EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD',
    'NZDUSD', 'EURJPY', 'GBPJPY', 'BTCUSD',
]


class TradingApp(QMainWindow):
    _log_signal = Signal(str)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Neural Forex Trader")
        self.setMinimumSize(780, 620)
        self.resize(780, 660)

        self._setup_logging()

        # Core components
        self.config_manager = ConfigManager()
        self.mt5_connector = MT5Connector()
        self.orchestrator = None
        self.is_trading = False
        self._dark_mode = False

        self._build_ui()
        self._load_settings()

        # Thread-safe logging
        self._log_signal.connect(self._log_on_main_thread)

        # Live update timer
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._update_live)
        self.timer.start(2000)

        # Show import errors if any
        if _IMPORT_ERRORS:
            for err in _IMPORT_ERRORS:
                self._log(f"<span style='color:#C44444'>⚠ {err}</span>")
            self._log("<span style='color:#D4A040'>Fix the above errors then restart the app.</span>")
        else:
            self._log("Ready. Connect MT5 to begin.")

    # ------------------------------------------------------------------
    def _setup_logging(self):
        Path("logs").mkdir(exist_ok=True)
        self.logger = logging.getLogger("TradingApp")
        self.logger.setLevel(logging.INFO)
        fh = logging.FileHandler("logs/trading.log")
        fh.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
        self.logger.addHandler(fh)

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setSpacing(6)
        root.setContentsMargins(12, 10, 12, 10)

        # ---- Status bar ----
        status_row = QHBoxLayout()
        self.lbl_mt5 = QLabel("MT5: --")
        self.lbl_mt5.setObjectName("statusLabel")
        self.lbl_trading = QLabel("Trading: OFF")
        self.lbl_trading.setObjectName("statusLabel")
        self.lbl_balance = QLabel("$0.00")
        self.lbl_balance.setObjectName("balanceLabel")
        self.lbl_balance.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        for lbl in [self.lbl_mt5, self.lbl_trading]:
            status_row.addWidget(lbl)
        status_row.addStretch()
        status_row.addWidget(self.lbl_balance)
        root.addLayout(status_row)

        # Separator
        sep = QFrame()
        sep.setObjectName("separator")
        sep.setFrameShape(QFrame.HLine)
        root.addWidget(sep)

        # ---- Buttons ----
        btn_row = QHBoxLayout()
        self.btn_connect = QPushButton("Connect MT5")
        self.btn_connect.clicked.connect(self._on_connect)
        self.btn_start = QPushButton("Start")
        self.btn_start.setObjectName("startBtn")
        self.btn_start.setEnabled(False)
        self.btn_start.clicked.connect(self._on_start)
        self.btn_stop = QPushButton("Stop")
        self.btn_stop.setObjectName("stopBtn")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self._on_stop)
        self.btn_save = QPushButton("Save")
        self.btn_save.clicked.connect(self._on_save_settings)
        self.btn_theme = QPushButton("Dark Mode")
        self.btn_theme.clicked.connect(self._on_toggle_theme)
        self.btn_emergency = QPushButton("Emergency Stop")
        self.btn_emergency.setObjectName("emergencyBtn")
        self.btn_emergency.clicked.connect(self._on_emergency)

        for b in [self.btn_connect, self.btn_start, self.btn_stop, self.btn_save, self.btn_theme]:
            btn_row.addWidget(b)
        btn_row.addStretch()
        btn_row.addWidget(self.btn_emergency)
        root.addLayout(btn_row)

        # ---- Settings (two group boxes side by side) ----
        settings_row = QHBoxLayout()

        # Left: Mode
        mode_box = QGroupBox("Mode")
        mode_grid = QGridLayout(mode_box)
        mode_grid.setSpacing(6)

        self.chk_fixed_lots = QCheckBox("Fixed Lots")
        self.chk_fixed_lots.setChecked(True)
        self.chk_fixed_lots.stateChanged.connect(self._on_lot_mode_changed)
        mode_grid.addWidget(self.chk_fixed_lots, 0, 0)
        self.inp_lot = QLineEdit("0.40")
        self.inp_lot.setFixedWidth(70)
        mode_grid.addWidget(self.inp_lot, 0, 1)

        mode_grid.addWidget(QLabel("Risk %:"), 1, 0)
        self.inp_risk = QLineEdit("8")
        self.inp_risk.setFixedWidth(70)
        mode_grid.addWidget(self.inp_risk, 1, 1)

        mode_grid.addWidget(QLabel("Min Conf %:"), 2, 0)
        self.inp_conf = QLineEdit("65")
        self.inp_conf.setFixedWidth(70)
        mode_grid.addWidget(self.inp_conf, 2, 1)

        settings_row.addWidget(mode_box)

        # Right: Trade Monitor
        monitor_box = QGroupBox("Trade Monitor")
        mon_grid = QGridLayout(monitor_box)
        mon_grid.setSpacing(6)

        mon_grid.addWidget(QLabel("Max Loss ($):"), 0, 0)
        self.inp_maxloss = QLineEdit("80")
        self.inp_maxloss.setFixedWidth(70)
        mon_grid.addWidget(self.inp_maxloss, 0, 1)

        mon_grid.addWidget(QLabel("BE Pips:"), 1, 0)
        self.inp_be = QLineEdit("15")
        self.inp_be.setFixedWidth(70)
        mon_grid.addWidget(self.inp_be, 1, 1)

        mon_grid.addWidget(QLabel("Stall (min):"), 2, 0)
        self.inp_stall = QLineEdit("30")
        self.inp_stall.setFixedWidth(70)
        mon_grid.addWidget(self.inp_stall, 2, 1)

        mon_grid.addWidget(QLabel("Deadline (min):"), 3, 0)
        self.inp_deadline = QLineEdit("60")
        self.inp_deadline.setFixedWidth(70)
        mon_grid.addWidget(self.inp_deadline, 3, 1)

        mon_grid.addWidget(QLabel("Profit Target ($):"), 4, 0)
        self.inp_profit_target = QLineEdit("")
        self.inp_profit_target.setFixedWidth(70)
        self.inp_profit_target.setPlaceholderText("e.g. 120")
        self.inp_profit_target.setStyleSheet(
            "background: #EEECE2; border: 1px solid #4A8C5D; color: #4A8C5D; padding: 4px 8px; border-radius: 8px;")
        mon_grid.addWidget(self.inp_profit_target, 4, 1)

        settings_row.addWidget(monitor_box)
        root.addLayout(settings_row)

        # ---- Pairs row ----
        pairs_box = QGroupBox("Pairs")
        pairs_row = QHBoxLayout(pairs_box)
        pairs_row.setSpacing(10)
        self.pair_checks = {}
        for pair in ALL_PAIRS:
            chk = QCheckBox(pair)
            chk.setChecked(pair != 'USDJPY')
            self.pair_checks[pair] = chk
            pairs_row.addWidget(chk)
        pairs_row.addStretch()
        root.addWidget(pairs_box)

        # ---- Positions table ----
        pos_box = QGroupBox("Open Positions")
        pos_layout = QVBoxLayout(pos_box)
        pos_layout.setContentsMargins(4, 14, 4, 4)

        # Total P/L + Profit Target display row
        pnl_row = QHBoxLayout()
        self.lbl_total_pnl = QLabel("Total P/L: $0.00")
        self.lbl_total_pnl.setStyleSheet(
            "font-size: 15px; font-weight: bold; color: #141413; padding: 2px 4px;")
        pnl_row.addWidget(self.lbl_total_pnl)
        self.lbl_target_status = QLabel("")
        self.lbl_target_status.setStyleSheet(
            "font-size: 12px; color: #B0AEA5; padding: 2px 4px;")
        pnl_row.addWidget(self.lbl_target_status)
        pnl_row.addStretch()
        # Session timer — how long trading has been running
        self.lbl_session_timer = QLabel("Session: 0s")
        self.lbl_session_timer.setStyleSheet(
            "font-size: 13px; color: #8B7355; padding: 2px 8px;")
        pnl_row.addWidget(self.lbl_session_timer)
        self._app_start_time = _time.time()
        self._session_start_time = None

        self.btn_close_all = QPushButton("Close All")
        self.btn_close_all.setFixedWidth(80)
        self.btn_close_all.setStyleSheet("background-color: #C44444; border: none; color: #FFFFFF; border-radius: 8px;")
        self.btn_close_all.clicked.connect(self._on_close_all)
        pnl_row.addWidget(self.btn_close_all)
        pos_layout.addLayout(pnl_row)

        self.pos_table = QTableWidget(0, 11)
        self.pos_table.setHorizontalHeaderLabels(
            ['Symbol', 'Dir', 'Lots', 'Entry', 'Current', 'P/L', 'Timer', 'SL', 'TP', 'New TP', ''])
        header = self.pos_table.horizontalHeader()
        for col in range(9):  # Symbol..TP stretch
            header.setSectionResizeMode(col, QHeaderView.Stretch)
        header.setSectionResizeMode(9, QHeaderView.Fixed)   # New TP edit
        self.pos_table.setColumnWidth(9, 90)
        header.setSectionResizeMode(10, QHeaderView.Fixed)   # Set button
        self.pos_table.setColumnWidth(10, 70)
        self.pos_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.pos_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.pos_table.verticalHeader().setVisible(False)
        self.pos_table.setMaximumHeight(180)

        # Store position data for actions
        self._pos_data = []

        pos_layout.addWidget(self.pos_table)
        root.addWidget(pos_box)

        # ---- Log ----
        log_box = QGroupBox("Log")
        log_layout = QVBoxLayout(log_box)
        log_layout.setContentsMargins(4, 14, 4, 4)
        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumHeight(140)
        log_layout.addWidget(self.log_view)
        root.addWidget(log_box)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _log(self, msg):
        if threading.current_thread() is not threading.main_thread():
            try:
                self._log_signal.emit(msg)
            except RuntimeError:
                pass
            return
        self._log_on_main_thread(msg)

    def _log_on_main_thread(self, msg):
        ts = datetime.now().strftime('%H:%M:%S')
        self.log_view.append(f"<span style='color:#B0AEA5'>[{ts}]</span> {msg}")
        bar = self.log_view.verticalScrollBar()
        bar.setValue(bar.maximum())

    def _set_status(self, label, text, ok):
        color = "#4A8C5D" if ok else "#B0AEA5"
        label.setText(text)
        label.setStyleSheet(f"color: {color};")

    def _float(self, widget, default):
        try:
            return float(widget.text())
        except ValueError:
            return default

    @staticmethod
    def _fmt_duration(open_timestamp):
        """Format seconds since open_timestamp as  Xh Ym  or  Ym Zs."""
        elapsed = int(_time.time() - open_timestamp)
        if elapsed < 0:
            elapsed = 0
        h, rem = divmod(elapsed, 3600)
        m, s = divmod(rem, 60)
        if h > 0:
            return f"{h}h {m:02d}m"
        elif m > 0:
            return f"{m}m {s:02d}s"
        else:
            return f"{s}s"

    # ------------------------------------------------------------------
    # Live update
    # ------------------------------------------------------------------
    def _update_live(self):
        mt5_ok = self.mt5_connector.is_connected()

        self._set_status(self.lbl_mt5, f"MT5: {'ON' if mt5_ok else 'OFF'}", mt5_ok)
        self._set_status(self.lbl_trading,
                         f"Trading: {'LIVE' if self.is_trading else 'OFF'}",
                         self.is_trading)

        if mt5_ok:
            try:
                info = self.mt5_connector.get_account_info()
                if info:
                    bal = float(info.get('balance', 0))
                    eq = float(info.get('equity', 0))
                    self.lbl_balance.setText(f"${bal:,.2f}  (eq: ${eq:,.2f})")
            except Exception:
                pass

        if mt5_ok and not self.is_trading:
            self.btn_start.setEnabled(True)

        # Update session timer — show trading session time if live, otherwise app uptime
        if self._session_start_time is not None:
            self.lbl_session_timer.setText(f"Trading: {self._fmt_duration(self._session_start_time)}")
        else:
            self.lbl_session_timer.setText(f"Uptime: {self._fmt_duration(self._app_start_time)}")

        self._refresh_positions()

    def _refresh_positions(self):
        if not self.mt5_connector.is_connected():
            self.pos_table.setRowCount(0)
            self._pos_data = []
            self.lbl_total_pnl.setText("Total P/L: $0.00")
            self.lbl_total_pnl.setStyleSheet(
                "font-size: 15px; font-weight: bold; color: #141413; padding: 2px 4px;")
            self.lbl_target_status.setText("")
            return
        try:
            import MetaTrader5 as mt5_lib
            positions = mt5_lib.positions_get()
            if not positions:
                self.pos_table.setRowCount(0)
                self._pos_data = []
                self.lbl_total_pnl.setText("Total P/L: $0.00")
                self.lbl_total_pnl.setStyleSheet(
                    "font-size: 15px; font-weight: bold; color: #141413; padding: 2px 4px;")
                self.lbl_target_status.setText("")
                return

            # Only rebuild table if position count changed (avoids flickering edits)
            if len(positions) != len(self._pos_data) or \
               [p.ticket for p in positions] != [d['ticket'] for d in self._pos_data]:
                self._rebuild_positions_table(positions)
            else:
                # Just update price / P/L columns (don't touch TP edit fields)
                self._update_positions_values(positions)

            # --- Total P/L + Profit Target check ---
            total_pnl = sum(p.profit for p in positions)
            pnl_color = "#4A8C5D" if total_pnl >= 0 else "#C44444"
            self.lbl_total_pnl.setText(f"Total P/L: ${total_pnl:+.2f}")
            self.lbl_total_pnl.setStyleSheet(
                f"font-size: 15px; font-weight: bold; color: {pnl_color}; padding: 2px 4px;")

            # Check profit target
            target_text = self.inp_profit_target.text().strip()
            if target_text:
                try:
                    target = float(target_text)
                    pct = (total_pnl / target * 100) if target > 0 else 0
                    self.lbl_target_status.setText(
                        f"Target: ${target:.0f}  ({pct:.0f}%)")
                    if total_pnl >= target:
                        self.lbl_target_status.setStyleSheet(
                            "font-size: 12px; color: #4A8C5D; font-weight: bold; padding: 2px 4px;")
                        self._log(
                            f"<span style='color:#4A8C5D'>PROFIT TARGET HIT! "
                            f"${total_pnl:+.2f} >= ${target:.0f} -- Closing all positions</span>")
                        self.inp_profit_target.setText("")  # Clear to prevent re-trigger
                        self._on_close_all()
                    else:
                        self.lbl_target_status.setStyleSheet(
                            "font-size: 12px; color: #B0AEA5; padding: 2px 4px;")
                except ValueError:
                    self.lbl_target_status.setText("(invalid target)")
                    self.lbl_target_status.setStyleSheet(
                        "font-size: 12px; color: #C44444; padding: 2px 4px;")
            else:
                self.lbl_target_status.setText("")

        except Exception:
            pass

    def _rebuild_positions_table(self, positions):
        """Full rebuild of positions table with edit fields and buttons."""
        self.pos_table.setRowCount(len(positions))
        self._pos_data = []

        for i, p in enumerate(positions):
            direction = "BUY" if p.type == 0 else "SELL"
            pnl = p.profit
            color = QColor("#4A8C5D") if pnl >= 0 else QColor("#C44444")
            timer_str = self._fmt_duration(p.time)

            pos_info = {
                'ticket': p.ticket, 'symbol': p.symbol,
                'direction': direction, 'volume': p.volume,
                'sl': p.sl, 'tp': p.tp,
            }
            self._pos_data.append(pos_info)

            vals = [
                p.symbol, direction, f"{p.volume:.2f}",
                f"{p.price_open:.5f}", f"{p.price_current:.5f}",
                f"${pnl:+.2f}", timer_str, f"{p.sl:.5f}", f"{p.tp:.5f}",
            ]
            for j, v in enumerate(vals):
                item = QTableWidgetItem(v)
                item.setTextAlignment(Qt.AlignCenter)
                if j == 5:  # P/L color
                    item.setForeground(color)
                if j == 6:  # Timer color
                    item.setForeground(QColor("#8B7355"))
                self.pos_table.setItem(i, j, item)

            # New TP edit field (col 9)
            tp_edit = QLineEdit(f"{p.tp:.5f}")
            tp_edit.setAlignment(Qt.AlignCenter)
            tp_edit.setStyleSheet("background: #EEECE2; border: 1px solid #D5D3C8; color: #C15F3C; padding: 2px; border-radius: 4px;")
            self.pos_table.setCellWidget(i, 9, tp_edit)

            # Set TP button (col 10)
            btn = QPushButton("Set")
            btn.setStyleSheet("background: #C15F3C; color: #FFFFFF; padding: 2px 8px; font-size: 11px; border-radius: 6px; border: none;")
            row_idx = i
            btn.clicked.connect(lambda checked, r=row_idx: self._on_set_single_tp(r))
            self.pos_table.setCellWidget(i, 10, btn)

    def _update_positions_values(self, positions):
        """Update only price/P/L/timer columns without rebuilding edit fields."""
        for i, p in enumerate(positions):
            pnl = p.profit
            color = QColor("#4A8C5D") if pnl >= 0 else QColor("#C44444")

            # Update Current price (col 4)
            item = QTableWidgetItem(f"{p.price_current:.5f}")
            item.setTextAlignment(Qt.AlignCenter)
            self.pos_table.setItem(i, 4, item)

            # Update P/L (col 5)
            item = QTableWidgetItem(f"${pnl:+.2f}")
            item.setTextAlignment(Qt.AlignCenter)
            item.setForeground(color)
            self.pos_table.setItem(i, 5, item)

            # Update Timer (col 6)
            item = QTableWidgetItem(self._fmt_duration(p.time))
            item.setTextAlignment(Qt.AlignCenter)
            item.setForeground(QColor("#8B7355"))
            self.pos_table.setItem(i, 6, item)

            # Update SL (col 7) — may have been trailed
            item = QTableWidgetItem(f"{p.sl:.5f}")
            item.setTextAlignment(Qt.AlignCenter)
            self.pos_table.setItem(i, 7, item)

            # Update current TP display (col 8)
            item = QTableWidgetItem(f"{p.tp:.5f}")
            item.setTextAlignment(Qt.AlignCenter)
            self.pos_table.setItem(i, 8, item)

            # Update stored data
            if i < len(self._pos_data):
                self._pos_data[i]['sl'] = p.sl
                self._pos_data[i]['tp'] = p.tp

    def _on_set_single_tp(self, row):
        """Set TP for a single position from the edit field in that row."""
        if row >= len(self._pos_data):
            return
        tp_widget = self.pos_table.cellWidget(row, 9)
        if tp_widget is None:
            return
        try:
            new_tp = float(tp_widget.text())
        except ValueError:
            self._log("Invalid TP value")
            return

        pos = self._pos_data[row]
        self._modify_tp(pos['ticket'], pos['symbol'], pos['sl'], new_tp)

    def _on_close_all(self):
        """Close all open positions."""
        if not self._pos_data:
            return
        for pos in self._pos_data:
            self._close_position_mt5(pos['ticket'], pos['symbol'], pos['direction'], pos['volume'])

    def _modify_tp(self, ticket, symbol, sl, new_tp):
        """Send SLTP modify to MT5."""
        try:
            import MetaTrader5 as mt5_lib
            request = {
                "action": mt5_lib.TRADE_ACTION_SLTP,
                "symbol": symbol,
                "position": ticket,
                "sl": sl,
                "tp": new_tp,
            }
            result = mt5_lib.order_send(request)
            if result and result.retcode == mt5_lib.TRADE_RETCODE_DONE:
                self._log(f"TP updated: {symbol} #{ticket} -> {new_tp:.5f}")
            else:
                rc = result.retcode if result else "?"
                self._log(f"TP update failed {symbol}: retcode={rc}")
        except Exception as e:
            self._log(f"TP error: {e}")

    def _close_position_mt5(self, ticket, symbol, direction, volume):
        """Close a position on MT5."""
        try:
            import MetaTrader5 as mt5_lib
            sym_info = mt5_lib.symbol_info(symbol)
            if not sym_info:
                return
            if direction == "BUY":
                close_type = mt5_lib.ORDER_TYPE_SELL
                price = sym_info.bid
            else:
                close_type = mt5_lib.ORDER_TYPE_BUY
                price = sym_info.ask

            request = {
                "action": mt5_lib.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": volume,
                "type": close_type,
                "position": ticket,
                "price": price,
                "deviation": 20,
                "magic": 123456,
                "comment": "manual-close",
                "type_time": mt5_lib.ORDER_TIME_GTC,
                "type_filling": mt5_lib.ORDER_FILLING_FOK,
            }
            result = mt5_lib.order_send(request)
            if result and result.retcode == mt5_lib.TRADE_RETCODE_DONE:
                self._log(f"Closed: {symbol} {direction} #{ticket}")
            else:
                # Try other fill modes
                for fill in [mt5_lib.ORDER_FILLING_IOC, mt5_lib.ORDER_FILLING_RETURN]:
                    request["type_filling"] = fill
                    result = mt5_lib.order_send(request)
                    if result and result.retcode == mt5_lib.TRADE_RETCODE_DONE:
                        self._log(f"Closed: {symbol} {direction} #{ticket}")
                        return
                self._log(f"Close failed {symbol}: {result}")
        except Exception as e:
            self._log(f"Close error: {e}")

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------
    def _on_connect(self):
        def _work():
            try:
                self.btn_connect.setEnabled(False)
                self.btn_connect.setText("Connecting...")

                # Check if MetaTrader5 package is even installed
                try:
                    import MetaTrader5 as mt5_lib
                except ImportError:
                    self._log("<span style='color:#C44444'>MetaTrader5 package not installed!</span>")
                    self._log("Run: <b>pip install MetaTrader5</b>")
                    return

                ok = self.mt5_connector.connect()
                if ok:
                    self._log("<span style='color:#4A8C5D'>MT5 connected</span>")
                    info = self.mt5_connector.get_account_info()
                    if info:
                        self._log(f"Account: {info.get('login','?')} | "
                                  f"Server: {info.get('server','?')} | "
                                  f"Balance: ${info.get('balance','?')}")
                else:
                    # Get the actual error from MT5
                    try:
                        err = mt5_lib.last_error()
                        err_code, err_msg = err if err else (0, "unknown")
                    except Exception:
                        err_code, err_msg = 0, "unknown"

                    self._log(f"<span style='color:#C44444'>MT5 connection failed</span>")
                    self._log(f"Error: [{err_code}] {err_msg}")

                    # Give helpful advice based on error code
                    if err_code == -10003 or "IPC" in str(err_msg):
                        self._log("<span style='color:#D4A040'>→ MT5 terminal is not running. Open MetaTrader 5 first.</span>")
                    elif err_code == -10004:
                        self._log("<span style='color:#D4A040'>→ MT5 not found. Install MetaTrader 5 from your broker.</span>")
                    elif err_code == -10005:
                        self._log("<span style='color:#D4A040'>→ Wrong MT5 version. Update MetaTrader 5.</span>")
                    elif "timeout" in str(err_msg).lower():
                        self._log("<span style='color:#D4A040'>→ Connection timed out. Check your internet.</span>")
                    else:
                        self._log("<span style='color:#D4A040'>→ Make sure MT5 is open and logged into an account.</span>")

            except Exception as e:
                self._log(f"<span style='color:#C44444'>MT5 error: {e}</span>")
            finally:
                self.btn_connect.setEnabled(True)
                self.btn_connect.setText("Connect MT5")
        threading.Thread(target=_work, daemon=True).start()

    def _on_lot_mode_changed(self, state):
        """Toggle between fixed lots and adaptive (risk-based) lot sizing."""
        is_fixed = self.chk_fixed_lots.isChecked()
        self.inp_lot.setEnabled(is_fixed)
        if is_fixed:
            self._log("Lot mode: Fixed")
        else:
            self._log("Lot mode: Adaptive (risk-based from Risk %)")

    def _on_save_settings(self):
        """Save all UI settings to settings.json."""
        settings = {
            'fixed_lots': self.chk_fixed_lots.isChecked(),
            'lot_size': self.inp_lot.text(),
            'risk_pct': self.inp_risk.text(),
            'min_conf': self.inp_conf.text(),
            'max_loss': self.inp_maxloss.text(),
            'be_pips': self.inp_be.text(),
            'stall_min': self.inp_stall.text(),
            'deadline_min': self.inp_deadline.text(),
            'profit_target': self.inp_profit_target.text(),
            'dark_mode': self._dark_mode,
            'pairs': {p: c.isChecked() for p, c in self.pair_checks.items()},
        }
        try:
            settings_path = os.path.join(os.path.dirname(__file__), 'settings.json')
            with open(settings_path, 'w') as f:
                json.dump(settings, f, indent=2)
            self._log("<span style='color:#4A8C5D'>Settings saved</span>")
        except Exception as e:
            self._log(f"<span style='color:#C44444'>Save failed: {e}</span>")

    def _load_settings(self):
        """Load settings from settings.json on startup."""
        settings_path = os.path.join(os.path.dirname(__file__), 'settings.json')
        if not os.path.exists(settings_path):
            return
        try:
            with open(settings_path, 'r') as f:
                s = json.load(f)
            self.chk_fixed_lots.setChecked(s.get('fixed_lots', True))
            self.inp_lot.setText(s.get('lot_size', '0.40'))
            self.inp_risk.setText(s.get('risk_pct', '30'))
            self.inp_conf.setText(s.get('min_conf', '65'))
            self.inp_maxloss.setText(s.get('max_loss', '80'))
            self.inp_be.setText(s.get('be_pips', '15'))
            self.inp_stall.setText(s.get('stall_min', '30'))
            self.inp_deadline.setText(s.get('deadline_min', '60'))
            self.inp_profit_target.setText(s.get('profit_target', ''))
            pairs = s.get('pairs', {})
            for p, c in self.pair_checks.items():
                if p in pairs:
                    c.setChecked(pairs[p])
            if s.get('dark_mode', False):
                self._dark_mode = True
                self.btn_theme.setText("Light Mode")
                QApplication.instance().setStyleSheet(DARK_STYLE)
            self._on_lot_mode_changed(None)
        except Exception:
            pass

    def _on_toggle_theme(self):
        """Toggle between light (Claude) and dark themes."""
        self._dark_mode = not self._dark_mode
        if self._dark_mode:
            QApplication.instance().setStyleSheet(DARK_STYLE)
            self.btn_theme.setText("Light Mode")
        else:
            QApplication.instance().setStyleSheet(CLAUDE_STYLE)
            self.btn_theme.setText("Dark Mode")

    def _on_start(self):
        """Start the ZP trade scanner — uses actual zp_trade_now.py generate_signal() logic."""
        try:
            if not self.mt5_connector.is_connected():
                self._log("Connect MT5 first")
                return

            self._zp_running = True
            self.is_trading = True
            self._session_start_time = _time.time()
            self.btn_start.setEnabled(False)
            self.btn_stop.setEnabled(True)

            use_fixed = self.chk_fixed_lots.isChecked()
            lot = self._float(self.inp_lot, 0.40)
            risk_pct = self._float(self.inp_risk, 30) / 100.0
            min_conf = self._float(self.inp_conf, 65) / 100.0
            mode_str = f"Fixed {lot}" if use_fixed else f"Adaptive {risk_pct:.0%} risk"
            self._log(f"ZP Scanner started | {mode_str} | Min conf: {min_conf:.0%} | Scanning every 60s")

            def _zp_loop():
                import MetaTrader5 as mt5_lib
                import pandas as pd
                from app.zeropoint_signal import ZeroPointEngine

                zp_engine = ZeroPointEngine()

                # V4 Adaptive risk state
                _consec_wins = 0
                _consec_losses = 0
                _last_trade_count = 0

                while getattr(self, '_zp_running', False):
                    try:
                        selected = [p for p, c in self.pair_checks.items() if c.isChecked()]
                        use_fixed = self.chk_fixed_lots.isChecked()
                        fixed_lot = self._float(self.inp_lot, 0.40)
                        base_risk_pct = self._float(self.inp_risk, 30) / 100.0
                        min_conf = self._float(self.inp_conf, 65) / 100.0

                        # V4 Adaptive risk: scale risk based on recent performance
                        # Base: 30% (aggressive for 97.5% WR system — double every ~18 days)
                        # After 3+ consec wins: bump risk 25% (30% -> 37.5%)
                        # After a loss: reduce risk 37.5% for next 2 trades (30% -> 18.75%)
                        # After account grows past $50K: cap risk at 20%
                        risk_pct = base_risk_pct
                        try:
                            acct_info = mt5_lib.account_info()
                            if acct_info and acct_info.balance > 50000:
                                risk_pct = min(risk_pct, 0.20)  # Cap at 20% above $50K

                            # Track consecutive W/L from deal history
                            import datetime as _dt
                            from_date = _dt.datetime.now() - _dt.timedelta(days=30)
                            deals = mt5_lib.history_deals_get(from_date, _dt.datetime.now())
                            if deals:
                                # Count recent consecutive results (last 10 closed trades)
                                recent_pnls = [d.profit for d in deals if d.profit != 0 and d.magic == 123456][-10:]
                                _consec_wins = 0
                                _consec_losses = 0
                                for p in reversed(recent_pnls):
                                    if p > 0:
                                        _consec_wins += 1
                                        if _consec_losses > 0:
                                            break
                                    else:
                                        _consec_losses += 1
                                        if _consec_wins > 0:
                                            break

                                if _consec_wins >= 3:
                                    risk_pct = min(risk_pct * 1.25, 0.40)  # Max 40%
                                elif _consec_losses >= 1:
                                    risk_pct = risk_pct * 0.625  # 30% -> 18.75%
                        except Exception:
                            pass  # Keep base risk on error

                        # Track which symbols already have open positions
                        open_positions = mt5_lib.positions_get()
                        open_symbols = set()
                        if open_positions:
                            for pos in open_positions:
                                open_symbols.add(pos.symbol.upper().replace(".", "").replace("#", ""))

                        signals = []

                        for symbol in selected:
                            norm = symbol.upper().replace(".", "").replace("#", "")
                            if norm in open_symbols or symbol in open_symbols:
                                continue

                            # Resolve symbol on broker
                            sym_info = mt5_lib.symbol_info(symbol)
                            sym_resolved = symbol
                            if sym_info is None:
                                for alt in [symbol, symbol + ".raw", symbol[:3]]:
                                    sym_info = mt5_lib.symbol_info(alt)
                                    if sym_info is not None:
                                        sym_resolved = alt
                                        break
                            if sym_info is None:
                                continue

                            mt5_lib.symbol_select(sym_resolved, True)

                            rates_h4 = mt5_lib.copy_rates_from_pos(sym_resolved, mt5_lib.TIMEFRAME_H4, 0, 200)
                            rates_h1 = mt5_lib.copy_rates_from_pos(sym_resolved, mt5_lib.TIMEFRAME_H1, 0, 200)

                            df_h4 = None
                            if rates_h4 is not None and len(rates_h4) >= 20:
                                df_h4 = pd.DataFrame(rates_h4)
                                df_h4["time"] = pd.to_datetime(df_h4["time"], unit="s")
                            df_h1 = None
                            if rates_h1 is not None and len(rates_h1) >= 20:
                                df_h1 = pd.DataFrame(rates_h1)
                                df_h1["time"] = pd.to_datetime(df_h1["time"], unit="s")
                            if df_h4 is None:
                                continue

                            # Use generate_signal() — actual profitable logic
                            # Accepts ANY active ZP position with age penalty on confidence
                            sig = zp_engine.generate_signal(sym_resolved, df_h4, df_h1)
                            if sig is None:
                                continue

                            # Apply confidence filter
                            if sig.confidence < min_conf:
                                continue

                            self._log(
                                f"  {symbol}: ZP {sig.direction} "
                                f"R:R={sig.risk_reward:.2f} conf={sig.confidence:.0%} "
                                f"tier={sig.tier}"
                            )
                            signals.append((sig, sym_resolved))

                        # V4: Correlation filter — reduce lot if correlated pairs signal same direction
                        # USD pairs: EURUSD, GBPUSD, AUDUSD, NZDUSD, USDCAD, USDJPY
                        # JPY pairs: EURJPY, GBPJPY, USDJPY
                        USD_PAIRS = {"EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "USDCAD", "USDJPY"}
                        JPY_PAIRS = {"EURJPY", "GBPJPY", "USDJPY"}

                        def _get_correlation_group(sym_name):
                            norm = sym_name.upper().replace(".", "").replace("#", "").replace(".RAW", "")
                            groups = []
                            if norm in USD_PAIRS:
                                groups.append("USD")
                            if norm in JPY_PAIRS:
                                groups.append("JPY")
                            return groups

                        # Count signals per correlation group + direction
                        corr_counts = {}  # e.g. {"USD_BUY": 3, "JPY_SELL": 1}
                        for sig, sym_r in signals:
                            norm = sym_r.upper().replace(".", "").replace("#", "").replace(".RAW", "")
                            for grp in _get_correlation_group(norm):
                                key = f"{grp}_{sig.direction}"
                                corr_counts[key] = corr_counts.get(key, 0) + 1

                        # Place signals with risk scoring + correlation adjustment
                        if signals:
                            self._log(f"Placing {len(signals)} trade(s)...")
                            for sig, sym_resolved in signals:
                                lot = self._calc_lot_size(sig, sym_resolved, use_fixed, fixed_lot, risk_pct)

                                # ── V4 Loser Risk Scoring ──
                                # Score 0-5 warning flags. Each flag reduces lot size.
                                # Based on DNA profiler: losers have weaker momentum, buy at
                                # 20-bar extremes, enter during low ATR, BUY side, and Fridays.
                                risk_flags = 0
                                risk_reasons = []

                                # Flag 1: BUY side (3.3% loss rate vs 1.5% SELL)
                                if sig.direction == "BUY":
                                    risk_flags += 1
                                    risk_reasons.append("BUY-side")

                                # Flag 2: Friday (4.1% loss rate vs 2.0% avg)
                                import datetime as _dt_check
                                if _dt_check.datetime.now().weekday() == 4:
                                    risk_flags += 1
                                    risk_reasons.append("Friday")

                                # Flag 3: Price at 20-bar extreme (buying high / selling low)
                                # Losers avg 0.78, winners avg 0.52 on 20-bar range
                                try:
                                    norm_sym = sym_resolved.upper().replace(".", "").replace("#", "").replace(".RAW", "")
                                    rates_check = mt5_lib.copy_rates_from_pos(sym_resolved, mt5_lib.TIMEFRAME_H4, 0, 25)
                                    if rates_check is not None and len(rates_check) >= 20:
                                        highs_20 = [float(r[2]) for r in rates_check[-21:-1]]  # last 20 bars (exclude current)
                                        lows_20 = [float(r[3]) for r in rates_check[-21:-1]]
                                        h20 = max(highs_20)
                                        l20 = min(lows_20)
                                        rng = h20 - l20
                                        if rng > 0:
                                            price_pos = (sig.entry_price - l20) / rng
                                            # BUY at top of range or SELL at bottom = exhaustion
                                            if (sig.direction == "BUY" and price_pos > 0.85) or \
                                               (sig.direction == "SELL" and price_pos < 0.15):
                                                risk_flags += 1
                                                risk_reasons.append(f"20bar-extreme({price_pos:.2f})")
                                except Exception:
                                    pass

                                # Flag 4: ATR not expanding (losers: atr_vs_median=1.01, winners: 1.12)
                                try:
                                    if rates_check is not None and len(rates_check) >= 20:
                                        # Quick ATR from last bar
                                        cur_atr = getattr(sig, '_atr', None)
                                        if cur_atr is None:
                                            # Approximate from SL distance
                                            cur_atr = abs(sig.entry_price - sig.stop_loss) / 3.0
                                        # ATR 10 bars ago (rough)
                                        closes = [float(r[4]) for r in rates_check[-12:]]
                                        highs = [float(r[2]) for r in rates_check[-12:]]
                                        lows = [float(r[3]) for r in rates_check[-12:]]
                                        trs = []
                                        for k in range(1, len(closes)):
                                            tr = max(highs[k]-lows[k], abs(highs[k]-closes[k-1]), abs(lows[k]-closes[k-1]))
                                            trs.append(tr)
                                        if len(trs) >= 5:
                                            recent_atr = sum(trs[-5:]) / 5
                                            older_atr = sum(trs[:5]) / 5
                                            if older_atr > 0 and recent_atr / older_atr < 0.95:
                                                risk_flags += 1
                                                risk_reasons.append(f"ATR-contracting({recent_atr/older_atr:.2f})")
                                except Exception:
                                    pass

                                # Flag 5: Low 3-bar momentum (losers: 1.52 ATR, winners: 1.78 ATR)
                                try:
                                    if rates_check is not None and len(rates_check) >= 5:
                                        close_now = float(rates_check[-1][4])
                                        close_3ago = float(rates_check[-4][4])
                                        est_atr = abs(sig.entry_price - sig.stop_loss) / 3.0
                                        if est_atr > 0:
                                            mom = abs(close_now - close_3ago) / est_atr
                                            if mom < 1.2:  # well below loser average of 1.52
                                                risk_flags += 1
                                                risk_reasons.append(f"low-momentum({mom:.2f})")
                                except Exception:
                                    pass

                                # Apply risk score: 0 flags = full lot, 3+ flags = 50% lot, 4+ = skip
                                if risk_flags >= 4:
                                    self._log(f"    {sym_resolved}: SKIP ({risk_flags} risk flags: {', '.join(risk_reasons)})")
                                    continue
                                elif risk_flags >= 3:
                                    lot = round(lot * 0.50, 2)
                                    self._log(f"    {sym_resolved}: x0.50 lot ({risk_flags} flags: {', '.join(risk_reasons)})")
                                elif risk_flags >= 2:
                                    lot = round(lot * 0.75, 2)
                                    self._log(f"    {sym_resolved}: x0.75 lot ({risk_flags} flags: {', '.join(risk_reasons)})")
                                elif risk_flags >= 1:
                                    self._log(f"    {sym_resolved}: 1 flag ({', '.join(risk_reasons)}) — no adjustment")

                                # Correlation filter — reduce lot if correlated pairs signal same direction
                                norm = sym_resolved.upper().replace(".", "").replace("#", "").replace(".RAW", "")
                                corr_reduction = 1.0
                                for grp in _get_correlation_group(norm):
                                    key = f"{grp}_{sig.direction}"
                                    if corr_counts.get(key, 0) >= 3:
                                        corr_reduction = 0.7
                                        self._log(f"    {sym_resolved}: corr filter x0.7 ({corr_counts[key]} {grp} {sig.direction})")
                                        break
                                lot = round(lot * corr_reduction, 2)

                                if lot > 0:
                                    self._zp_place_trade(sig, sym_resolved, lot)
                        else:
                            self._log("Scan: no signals above confidence threshold")

                    except Exception as e:
                        self._log(f"Scan error: {e}")

                    # Wait 60s between scans
                    for _ in range(60):
                        if not getattr(self, '_zp_running', False):
                            break
                        _time.sleep(1)

            threading.Thread(target=_zp_loop, daemon=True).start()

        except Exception as e:
            self._log(f"Start error: {e}")

    # Tier-based lot multipliers: higher PF symbols get bigger positions
    TIER_LOT_MULTIPLIER = {
        "S": 1.5,   # USDCAD (PF 5.45) — 50% bigger
        "A": 1.2,   # GBPJPY, USDJPY — 20% bigger
        "B": 1.0,   # AUDUSD, EURJPY, NZDUSD — standard
        "C": 0.6,   # GBPUSD (PF 1.0) — 40% smaller
    }

    def _calc_lot_size(self, sig, symbol, use_fixed, fixed_lot, risk_pct):
        """Calculate lot size — risk-based with V4 tier scaling.

        Tier S (PF 5.45) gets 1.5x lot, Tier C (PF 1.0) gets 0.6x.
        """
        if use_fixed:
            # Even fixed lots get tier-scaled
            tier = getattr(sig, 'tier', 'B')
            tier_mult = self.TIER_LOT_MULTIPLIER.get(tier, 1.0)
            return round(fixed_lot * tier_mult, 2)
        try:
            import MetaTrader5 as mt5_lib
            sym_info = mt5_lib.symbol_info(symbol)
            if sym_info is None:
                return fixed_lot

            acct = mt5_lib.account_info()
            if acct is None:
                return fixed_lot

            balance = acct.balance
            risk_amount = balance * risk_pct

            point = sym_info.point
            tick_size = sym_info.trade_tick_size or point
            tick_value = sym_info.trade_tick_value
            if tick_value <= 0:
                tick_value = sym_info.trade_contract_size * tick_size

            sl_distance = abs(sig.entry_price - sig.stop_loss)
            sl_ticks = sl_distance / tick_size if tick_size > 0 else 0
            loss_per_lot = sl_ticks * tick_value

            if loss_per_lot <= 0:
                return fixed_lot

            lot = risk_amount / loss_per_lot

            # V4 Tier scaling — higher PF symbols get bigger lots
            tier = getattr(sig, 'tier', 'B')
            tier_mult = self.TIER_LOT_MULTIPLIER.get(tier, 1.0)
            lot *= tier_mult

            vol_step = sym_info.volume_step
            lot = round(lot / vol_step) * vol_step
            lot = max(sym_info.volume_min, min(lot, sym_info.volume_max))

            self._log(
                f"    {symbol} adaptive lot={lot:.2f} "
                f"(risk ${risk_amount:.0f}, loss/lot ${loss_per_lot:.0f}, "
                f"tier={tier} x{tier_mult})"
            )
            return lot
        except Exception as e:
            self._log(f"    Lot calc error: {e}, using fixed={fixed_lot}")
            return fixed_lot

    def _zp_place_trade(self, sig, sym_resolved, lot):
        """Place a trade exactly like zp_trade_now.py."""
        try:
            import MetaTrader5 as mt5_lib

            sym_info = mt5_lib.symbol_info(sym_resolved)
            if sym_info is None:
                self._log(f"Cannot get info for {sym_resolved}")
                return

            if sig.direction == "BUY":
                order_type = mt5_lib.ORDER_TYPE_BUY
                price = sym_info.ask
            else:
                order_type = mt5_lib.ORDER_TYPE_SELL
                price = sym_info.bid

            digits = sym_info.digits
            sl = round(sig.stop_loss, digits)
            # V4: Set broker TP to TP3 (5.0x ATR) — V4 manages partials at TP1/TP2 internally
            # If we set TP to TP1, broker auto-closes before our partial system runs
            tp = round(sig.tp3, digits)

            request = {
                "action": mt5_lib.TRADE_ACTION_DEAL,
                "symbol": sym_resolved,
                "volume": lot,
                "type": order_type,
                "price": price,
                "sl": sl,
                "tp": tp,
                "deviation": 20,
                "magic": 123456,
                "comment": f"ZP-{sig.tier}",
                "type_time": mt5_lib.ORDER_TIME_GTC,
                "type_filling": mt5_lib.ORDER_FILLING_FOK,
            }

            result = mt5_lib.order_send(request)
            if result and result.retcode == mt5_lib.TRADE_RETCODE_DONE:
                self._log(
                    f"<span style='color:#4A8C5D'>TRADE: {sig.direction} {sym_resolved} "
                    f"@ {price:.5f} | SL={sl} TP={tp} | "
                    f"R:R={sig.risk_reward:.2f} Tier={sig.tier}</span>"
                )
            else:
                # Try other fill modes
                for fill in [mt5_lib.ORDER_FILLING_IOC, mt5_lib.ORDER_FILLING_RETURN]:
                    request["type_filling"] = fill
                    result = mt5_lib.order_send(request)
                    if result and result.retcode == mt5_lib.TRADE_RETCODE_DONE:
                        self._log(
                            f"<span style='color:#4A8C5D'>TRADE: {sig.direction} {sym_resolved} "
                            f"@ {price:.5f} | SL={sl} TP={tp}</span>"
                        )
                        return
                rc = result.retcode if result else "?"
                self._log(f"<span style='color:#C44444'>Trade failed {sym_resolved}: {rc}</span>")

        except Exception as e:
            self._log(f"Trade error: {e}")

    def _on_stop(self):
        try:
            self._zp_running = False
            self.is_trading = False
            self._session_start_time = None
            self.lbl_session_timer.setText(f"Uptime: {self._fmt_duration(self._app_start_time)}")
            self.btn_start.setEnabled(True)
            self.btn_stop.setEnabled(False)
            self._log("Trading stopped")
        except Exception as e:
            self._log(f"Stop error: {e}")

    def _on_emergency(self):
        try:
            self._zp_running = False
            self.is_trading = False
            self._session_start_time = None
            self.lbl_session_timer.setText(f"Uptime: {self._fmt_duration(self._app_start_time)}")
            self.btn_start.setEnabled(True)
            self.btn_stop.setEnabled(False)
            self._log("<span style='color:#C44444'>EMERGENCY STOP</span>")
        except Exception as e:
            self._log(f"Emergency error: {e}")


def main():
    app = QApplication(sys.argv)
    app.setStyleSheet(CLAUDE_STYLE)
    window = TradingApp()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
