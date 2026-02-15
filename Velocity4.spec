# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for Velocity 4.
Builds a standalone Windows application from scripts/webhook_app.py.

Usage:
    pyinstaller Velocity4.spec --noconfirm
"""

import os

a = Analysis(
    ['scripts/webhook_app.py'],
    pathex=['.'],
    binaries=[],
    datas=[
        # Web UI (sits next to webhook_app.py at runtime)
        ('scripts/v4_ui.html', 'scripts'),
        # App package (zeropoint_signal, etc.)
        ('app', 'app'),
        # App icon (PNG for Qt window icon at runtime)
        ('assets/velocity4.png', 'assets'),
    ],
    hiddenimports=[
        # PySide6 Web Engine
        'PySide6.QtCore',
        'PySide6.QtWidgets',
        'PySide6.QtGui',
        'PySide6.QtWebEngineWidgets',
        'PySide6.QtWebEngineCore',
        'PySide6.QtWebChannel',
        # MetaTrader5
        'MetaTrader5',
        # Data / Numeric
        'numpy',
        'pandas',
        # Standard lib commonly missed
        'json',
        'logging',
        'threading',
        'math',
        'pathlib',
        'concurrent.futures',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Exclude heavy packages not used by the web UI app
        'torch',
        'tensorflow',
        'keras',
        'scipy',
        'sklearn',
        'scikit-learn',
        'matplotlib',
        'PIL',
        'Pillow',
        'cv2',
        'opencv',
        'pytest',
        'IPython',
        'jupyter',
        'notebook',
        'tkinter',
        'torchaudio',
        'torchvision',
        'tensorboard',
        'lightweight_charts',
    ],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='Velocity4',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,  # No console window — PySide6 GUI app
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='assets/velocity4.ico',
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='Velocity4',
)
