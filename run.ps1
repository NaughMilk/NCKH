# SDY Pipeline PowerShell Runner
# Run with: .\run.ps1

Write-Host "🚀 SDY Pipeline - Starting..." -ForegroundColor Green
Write-Host ""

# Check if Python is available
try {
    $pythonVersion = python --version 2>&1
    Write-Host "✅ Python found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ Python not found! Please install Python first." -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

# Run the main pipeline
Write-Host "🚀 Launching SDY Pipeline..." -ForegroundColor Yellow
python main.py $args

# Check exit code
if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "❌ Pipeline failed with error code $LASTEXITCODE" -ForegroundColor Red
    Read-Host "Press Enter to exit"
}
