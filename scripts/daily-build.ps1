# LMS Platform - Daily Module Description Build (PowerShell version)
# This script regenerates all module descriptions with enhanced lesson metadata

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "LMS Platform - Daily Module Description Build" -ForegroundColor White
Write-Host "Started at $(Get-Date)" -ForegroundColor Gray
Write-Host "======================================================================" -ForegroundColor Cyan

# Set working directory
Set-Location "C:\ai\training\lms-platform"

# Define Python executable path
$pythonPath = "C:\Users\bimal\AppData\Local\Programs\Python\Python311\python.exe"

# Verify Python is available
if (Test-Path $pythonPath) {
    Write-Host "✓ Python found at: $pythonPath" -ForegroundColor Green
} else {
    Write-Host "✗ Python not found at: $pythonPath" -ForegroundColor Red
    exit 1
}

# Run the module description builder for all tracks
Write-Host "`nRunning module description builder for all tracks..." -ForegroundColor Yellow

try {
    & $pythonPath "scripts/build_module_descriptions.py" "all"
    $exitCode = $LASTEXITCODE
    
    if ($exitCode -eq 0) {
        Write-Host "`n======================================================================" -ForegroundColor Green
        Write-Host "SUCCESS: Daily module description build completed successfully!" -ForegroundColor Green
        Write-Host "Completed at $(Get-Date)" -ForegroundColor Gray
        Write-Host "======================================================================" -ForegroundColor Green
        
        # Log success
        "$(Get-Date) - Daily build SUCCESS" | Add-Content -Path "scripts/build-log.txt"
        
        # Show summary of generated files
        Write-Host "`nGenerated module.json files:" -ForegroundColor Cyan
        Get-ChildItem -Path "src/data/*/modules-descriptions/module.json" | ForEach-Object {
            $size = [math]::Round($_.Length / 1KB, 1)
            Write-Host "  $($_.FullName.Replace((Get-Location).Path + '\', '')) ($size KB)" -ForegroundColor Gray
        }
        
        exit 0
    } else {
        throw "Build process returned exit code: $exitCode"
    }
} catch {
    Write-Host "`n======================================================================" -ForegroundColor Red
    Write-Host "ERROR: Daily module description build failed!" -ForegroundColor Red
    Write-Host "Error: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host "Failed at $(Get-Date)" -ForegroundColor Gray
    Write-Host "======================================================================" -ForegroundColor Red
    
    # Log failure
    "$(Get-Date) - Daily build FAILED: $($_.Exception.Message)" | Add-Content -Path "scripts/build-log.txt"
    
    exit 1
}