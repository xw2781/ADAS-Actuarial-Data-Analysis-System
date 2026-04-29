@echo off
setlocal EnableExtensions
set "SCRIPT_DIR=%~dp0"
for %%I in ("%SCRIPT_DIR%..") do set "APP_ROOT=%%~fI"
cd /d "%APP_ROOT%"

echo ========================================
echo Building ArcRho Standalone Application
echo ========================================
echo.

REM Setup portable node in PATH
set "NODE_HOME=%APP_ROOT%\node-portable"
set "PATH=%NODE_HOME%;%PATH%"
set "APP_BUILDER_EXE=node_modules\app-builder-bin\win\x64\app-builder.exe"
set "APP_VERSION="

echo Step 0: Validating release note fragments...
echo ----------------------------------------
python build\release_notes.py check
if errorlevel 1 (
    echo ERROR: Release note fragment validation failed.
    echo.
    pause
    exit /b 1
)
echo Release note fragments validated.
echo.

echo Step 1: Updating application version...
echo ----------------------------------------
if "%~1"=="" (
    for /f "usebackq delims=" %%I in (`python build\version_manager.py`) do set "APP_VERSION=%%I"
) else (
    for /f "usebackq delims=" %%I in (`python build\version_manager.py "%~1"`) do set "APP_VERSION=%%I"
)
if errorlevel 1 (
    echo ERROR: Failed to update application version metadata.
    echo.
    pause
    exit /b 1
)
if not defined APP_VERSION (
    echo ERROR: Version updater did not return a version.
    echo.
    pause
    exit /b 1
)
echo Building version %APP_VERSION%
echo.

echo Step 2: Building Python app server with PyInstaller...
echo ----------------------------------------
call :run_pyinstaller
if errorlevel 1 (
    echo.
    pause
    exit /b 1
)
echo Python app server built successfully!
echo.

echo Step 3: Building Electron app with electron-builder...
echo ----------------------------------------
if not exist "python_dist\arcrho_server\arcrho_server.exe" (
    echo ERROR: Missing app-server bundle: python_dist\arcrho_server\arcrho_server.exe
    echo HINT: PyInstaller step did not produce the server executable.
    echo       Do not continue, otherwise installer may build fast but fail at launch.
    echo.
    pause
    exit /b 1
)
call :prepare_app_builder
call :run_electron
if errorlevel 1 (
    echo.
    pause
    exit /b 1
)

if not exist "dist\ArcRho-Setup-*.exe" (
    echo ERROR: Installer was not generated in dist\.
    echo.
    pause
    exit /b 1
)

echo.
echo Step 4: Generating release notes...
echo ----------------------------------------
for /f "usebackq delims=" %%I in (`python build\release_notes.py release "%APP_VERSION%"`) do set "RELEASE_NOTE_PATH=%%I"
if errorlevel 1 (
    echo ERROR: Failed to generate release notes for version %APP_VERSION%.
    echo.
    pause
    exit /b 1
)
echo Release notes generated: %RELEASE_NOTE_PATH%
echo.

echo Step 5: Cleaning Python build artifacts...
echo ----------------------------------------
if exist "python_dist" (
    rmdir /s /q "python_dist"
)
if exist "python_build" (
    rmdir /s /q "python_build"
)

if exist "dist\win-unpacked" (
    rmdir /s /q "dist\win-unpacked"
)
del /q "dist\*Portable*.exe" 2>nul
del /q "dist\*-portable*.exe" 2>nul
del /q "dist\*.zip" 2>nul

echo.
echo ========================================
echo Build completed successfully!
echo ========================================
echo.
echo Output location: dist\
echo.
echo - ArcRho-Setup-%APP_VERSION%.exe  (Installer)
echo - %RELEASE_NOTE_PATH%  (Release Notes)
echo.
pause
endlocal
exit /b 0

:run_pyinstaller
pyinstaller build\server.spec --distpath python_dist --workpath python_build --noconfirm
if not errorlevel 1 exit /b 0

echo.
echo WARNING: PyInstaller failed on first attempt.
echo Retrying once with a clean Python build workspace...
if exist "python_dist" (
    rmdir /s /q "python_dist"
)
if exist "python_build" (
    rmdir /s /q "python_build"
)
pyinstaller build\server.spec --distpath python_dist --workpath python_build --noconfirm --clean
if not errorlevel 1 exit /b 0

echo ERROR: PyInstaller build failed after retry.
echo HINT: Re-run manually to capture full traceback:
echo       pyinstaller build\server.spec --distpath python_dist --workpath python_build --noconfirm --clean
exit /b 1

:prepare_app_builder
if not exist "%APP_BUILDER_EXE%" exit /b 0
REM Best-effort unblock in case Windows marks this binary as downloaded.
powershell -NoProfile -ExecutionPolicy Bypass -Command "Get-Item -Path '%APP_BUILDER_EXE%' -ErrorAction SilentlyContinue | Unblock-File -ErrorAction SilentlyContinue" >nul 2>nul
exit /b 0

:run_electron
call "%NODE_HOME%\node.exe" node_modules\electron-builder\cli.js --win
if not errorlevel 1 exit /b 0

echo.
echo WARNING: Electron build failed on first attempt.
echo Retrying once after re-preparing app-builder...
call :prepare_app_builder
timeout /t 2 /nobreak >nul
call "%NODE_HOME%\node.exe" node_modules\electron-builder\cli.js --win
if not errorlevel 1 exit /b 0

echo ERROR: Electron build failed after retry.
echo HINT: If error shows "spawn EPERM" for app-builder.exe, run:
echo       powershell -NoProfile -Command "Get-Item '%APP_BUILDER_EXE%' ^| Unblock-File"
echo       Then retry build\build_app.bat.
exit /b 1
