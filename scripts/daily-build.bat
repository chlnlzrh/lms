@echo off
REM Daily Module Description Builder
REM This script should be run daily to regenerate all module descriptions

echo ======================================================================
echo LMS Platform - Daily Module Description Build
echo Started at %date% %time%
echo ======================================================================

REM Change to the project directory
cd /d "C:\ai\training\lms-platform"

REM Run the module description builder for all tracks
"C:\Users\bimal\AppData\Local\Programs\Python\Python311\python.exe" scripts/build_module_descriptions.py all

REM Generate landing page data
echo.
echo Generating landing page data...
"C:\Users\bimal\AppData\Local\Programs\Python\Python311\python.exe" scripts/build_landing_pages.py

REM Check if the build was successful
if %ERRORLEVEL% EQU 0 (
    echo.
    echo ======================================================================
    echo SUCCESS: Daily module description build completed successfully!
    echo Completed at %date% %time%
    echo ======================================================================
    
    REM Optional: Log success to a file
    echo %date% %time% - Daily build SUCCESS >> scripts/build-log.txt
) else (
    echo.
    echo ======================================================================
    echo ERROR: Daily module description build failed!
    echo Failed at %date% %time%
    echo ======================================================================
    
    REM Optional: Log failure to a file
    echo %date% %time% - Daily build FAILED >> scripts/build-log.txt
    
    REM Exit with error code
    exit /b 1
)

pause