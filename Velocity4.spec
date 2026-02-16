# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for Velocity 4.
Builds a standalone Windows application from scripts/webhook_app.py.

Usage:
    pyinstaller Velocity4.spec --noconfirm
"""

import os
from PyInstaller.utils.hooks import collect_all, collect_submodules

# Collect ALL numpy files (binaries, data, submodules) to avoid missing DLL errors
numpy_datas, numpy_binaries, numpy_hiddenimports = collect_all('numpy')
pandas_datas, pandas_binaries, pandas_hiddenimports = collect_all('pandas')
mt5_datas, mt5_binaries, mt5_hiddenimports = collect_all('MetaTrader5')

a = Analysis(
    ['scripts/webhook_app.py'],
    pathex=['.'],
    binaries=numpy_binaries + pandas_binaries + mt5_binaries,
    datas=[
        # Web UI (sits next to webhook_app.py at runtime)
        ('scripts/v4_ui.html', 'scripts'),
        # App package (zeropoint_signal, etc.)
        ('app', 'app'),
        # App icon (PNG for Qt window icon at runtime)
        ('assets/velocity4.png', 'assets'),
    ] + numpy_datas + pandas_datas + mt5_datas,
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
        # Data / Numeric — collected via collect_all above
        'numpy',
        'numpy._core',
        'numpy._core._methods',
        'numpy._core._dtype_ctypes',
        'numpy._core.multiarray',
        'numpy._core.umath',
        'numpy._core._multiarray_umath',
        'numpy.core',
        'numpy.core._methods',
        'numpy.core._dtype_ctypes',
        'numpy.core.multiarray',
        'numpy.core.umath',
        'numpy.core._multiarray_umath',
        'numpy.random',
        'numpy.random._generator',
        'numpy.random._bounded_integers',
        'numpy.random._common',
        'numpy.random._mt19937',
        'numpy.random._pcg64',
        'numpy.random._philox',
        'numpy.random._sfc64',
        'numpy.random.bit_generator',
        'numpy.random.mtrand',
        'numpy.linalg',
        'numpy.linalg._umath_linalg',
        'numpy.fft',
        'pandas',
        'pandas._libs',
        'pandas._libs.tslibs',
        # Standard lib commonly missed
        'json',
        'logging',
        'threading',
        'math',
        'pathlib',
        'concurrent.futures',
    ] + numpy_hiddenimports + pandas_hiddenimports + mt5_hiddenimports,
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
