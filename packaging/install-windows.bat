@echo off
setlocal EnableDelayedExpansion

:: MARS Installer for Windows
:: Creates isolated virtual environment and installs MARS

echo.
echo ===============================================================
echo            MARS - macOS Artifact Recovery Suite
echo                       Installer
echo ===============================================================
echo.

:: Get script directory
set "SCRIPT_DIR=%~dp0"
set "INSTALL_DIR=%SCRIPT_DIR:~0,-1%"
set "VENV_DIR=%INSTALL_DIR%\.venv"

:: Check Python version - try 'python' first, then 'py' launcher
echo Checking Python...
set "PYTHON_CMD="
where python >nul 2>&1
if not errorlevel 1 (
    set "PYTHON_CMD=python"
) else (
    where py >nul 2>&1
    if not errorlevel 1 (
        set "PYTHON_CMD=py"
    )
)

if not defined PYTHON_CMD (
    echo ERROR: Python is not installed or not in PATH.
    echo Please install Python 3.13 or higher from https://python.org
    echo Make sure to check "Add Python to PATH" during installation,
    echo or use the Python Launcher ^(py^) which is installed by default.
    pause
    exit /b 1
)

for /f "tokens=*" %%i in ('%PYTHON_CMD% -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"') do set PYTHON_VERSION=%%i
for /f "tokens=*" %%i in ('%PYTHON_CMD% -c "import sys; print(sys.version_info.major)"') do set PYTHON_MAJOR=%%i
for /f "tokens=*" %%i in ('%PYTHON_CMD% -c "import sys; print(sys.version_info.minor)"') do set PYTHON_MINOR=%%i

if %PYTHON_MAJOR% LSS 3 (
    echo ERROR: Python 3.13 or higher is required. Found: %PYTHON_VERSION%
    pause
    exit /b 1
)
if %PYTHON_MAJOR% EQU 3 if %PYTHON_MINOR% LSS 13 (
    echo ERROR: Python 3.13 or higher is required. Found: %PYTHON_VERSION%
    pause
    exit /b 1
)
echo Found Python %PYTHON_VERSION% (using '%PYTHON_CMD%')

:: Find wheel file
echo.
echo Looking for MARS wheel...
set "WHEEL_FILE="
for %%f in ("%SCRIPT_DIR%mars-*.whl") do set "WHEEL_FILE=%%f"
if not defined WHEEL_FILE (
    for %%f in ("%SCRIPT_DIR%dist\mars-*.whl") do set "WHEEL_FILE=%%f"
)

if not defined WHEEL_FILE (
    echo ERROR: No MARS wheel file found.
    echo Expected location: %SCRIPT_DIR%dist\mars-*.whl
    pause
    exit /b 1
)
echo Found: %WHEEL_FILE%

:: Create virtual environment
echo.
echo Creating virtual environment...
if exist "%VENV_DIR%" (
    set /p "REINSTALL=Virtual environment already exists. Reinstall? [y/N] "
    if /i "!REINSTALL!"=="y" (
        rmdir /s /q "%VENV_DIR%"
    ) else (
        echo Keeping existing installation.
        goto :install_deps
    )
)

%PYTHON_CMD% -m venv "%VENV_DIR%"
echo Virtual environment created

:install_deps
:: Activate and install
echo.
call "%VENV_DIR%\Scripts\activate.bat"
pip install --upgrade pip wheel >nul 2>&1

:: Check for pre-built wheels
set "WHEEL_DIR=%SCRIPT_DIR%wheels\windows-x64"

echo Installing MARS...
if exist "%WHEEL_DIR%\*.whl" (
    echo Using pre-built wheels
    pip install --find-links "%WHEEL_DIR%" "%WHEEL_FILE%"
) else (
    echo Pre-built wheels not found, building from source...
    echo This may take 10+ minutes for dfvfs dependencies.
    pip install "%WHEEL_FILE%"
)
echo MARS installed successfully

:: Create junction to bundled tools for easy user access (LGPL compliance)
echo.
echo Creating tools junction...
set "MARS_TOOLS_DIR=%INSTALL_DIR%\tools"
set "SITE_PACKAGES=%VENV_DIR%\Lib\site-packages"
set "RESOURCES_DIR=%SITE_PACKAGES%\resources\windows"

if exist "%RESOURCES_DIR%" (
    if exist "%MARS_TOOLS_DIR%" rmdir "%MARS_TOOLS_DIR%" 2>nul
    mklink /J "%MARS_TOOLS_DIR%" "%RESOURCES_DIR%" >nul 2>&1
    if exist "%MARS_TOOLS_DIR%" (
        echo Tools accessible at: %MARS_TOOLS_DIR%
        echo   Bundled binaries and libraries are linked here.
        echo   See THIRD-PARTY-NOTICES.md for licenses and source code.
    ) else (
        echo Note: Could not create tools junction. Run as administrator if needed.
    )
) else (
    echo Note: Resources directory not found in site-packages
)

:: Optional dependencies
echo.
echo ---------------------------------------------------------------
echo                    Optional Dependencies
echo ---------------------------------------------------------------
echo MARS can export timeline visualizations as PDF/PNG images.
echo This requires Kaleido.
echo.
set /p "INSTALL_KALEIDO=Install Kaleido for PDF/PNG export? [y/N] "
if /i "%INSTALL_KALEIDO%"=="y" (
    echo Installing Kaleido...
    pip install kaleido
    echo Kaleido installed
) else (
    echo Skipping Kaleido. You can install it later with:
    echo   %VENV_DIR%\Scripts\pip install kaleido
)

:: Create launcher batch file
echo.
echo Creating launcher script...
(
echo @echo off
echo call "%VENV_DIR%\Scripts\activate.bat"
echo mars %%*
) > "%INSTALL_DIR%\mars.bat"
echo Launcher created: %INSTALL_DIR%\mars.bat

:: Create PowerShell launcher
(
echo $env:VIRTUAL_ENV = "%VENV_DIR%"
echo $env:PATH = "%VENV_DIR%\Scripts;" + $env:PATH
echo ^& mars @args
) > "%INSTALL_DIR%\mars.ps1"

:: Installation complete
echo.
echo ===============================================================
echo                  Installation Complete!
echo ===============================================================
echo.
echo To run MARS:
echo   %INSTALL_DIR%\mars.bat
echo.
echo Or add to your PATH:
echo   1. Open System Properties ^> Environment Variables
echo   2. Add "%INSTALL_DIR%" to your PATH
echo.
echo Then run with just:
echo   "mars" or ".\mars"
echo.

:: Offer to add to PATH
set /p "ADD_PATH=Add MARS to your user PATH? [y/N] "
if /i "%ADD_PATH%"=="y" (
    setx PATH "%PATH%;%INSTALL_DIR%" >nul 2>&1
    echo Added to PATH. Restart your terminal for changes to take effect.
)

echo.
echo Done!
pause
