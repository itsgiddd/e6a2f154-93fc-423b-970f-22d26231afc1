@echo off
echo ============================================
echo  NeuralForexTrader - Build Windows .exe
echo ============================================
echo.

:: Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found. Install Python 3.10+ first.
    pause
    exit /b 1
)

:: Install dependencies
echo [1/3] Installing dependencies...
pip install -r requirements.txt
pip install pyinstaller

:: Build
echo.
echo [2/3] Building NeuralForexTrader.exe...
pyinstaller NeuralForexTrader.spec --noconfirm

:: Check result
if exist "dist\NeuralForexTrader\NeuralForexTrader.exe" (
    echo.
    echo ============================================
    echo  BUILD SUCCESSFUL
    echo  Output: dist\NeuralForexTrader\
    echo  Run:    dist\NeuralForexTrader\NeuralForexTrader.exe
    echo ============================================
) else (
    echo.
    echo ERROR: Build failed. Check output above.
)

pause
