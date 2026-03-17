# Deployment script for Python application on Windows
# Safe deployment for Windows

Write-Host "Step 0: Deactivating virtual environment if active..."
# Deactivate venv (works if using PowerShell)
deactivate 2>$null

Start-Sleep -Seconds 1

Write-Host "Step 1: Removing existing virtual environment..."
$venvPath = ".\.venv"

if (Test-Path $venvPath) {
    # Take ownership and remove ReadOnly attributes to avoid access errors
    Write-Host "Attempting to remove $venvPath..."
    Get-ChildItem $venvPath -Recurse -Force | ForEach-Object {
        $_.Attributes = 'Normal'
    }
    Remove-Item $venvPath -Recurse -Force -ErrorAction SilentlyContinue
    Write-Host "Existing virtual environment removed."
} else {
    Write-Host "No existing virtual environment found."
}

Write-Host "Step 2: Creating new virtual environment..."
python -m venv .venv
if (-Not (Test-Path ".\.venv\Scripts\Activate.ps1")) {
    Write-Error "Failed to create virtual environment. Check Python installation."
    exit 1
}

Write-Host "Step 3: Activating new virtual environment..."
# Activate venv for PowerShell
. .\.venv\Scripts\Activate.ps1

Write-Host "Step 4: Upgrading pip..."
python -m pip install --upgrade pip

Write-Host "Step 5: Installing required packages..."
python -m pip install -r requirements.txt

Write-Host "✅ Deployment complete! Virtual environment is ready."