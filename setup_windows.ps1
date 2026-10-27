$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    py -3.11 -m venv .venv
}
& ".venv\Scripts\python.exe" -m pip install --upgrade pip
& ".venv\Scripts\python.exe" -m pip install -r requirements.txt

$repoPath = Join-Path $PSScriptRoot "third_party\stylegan2-ada-pytorch"
if (-not (Test-Path (Join-Path $repoPath "legacy.py"))) {
    git clone https://github.com/NVlabs/stylegan2-ada-pytorch.git $repoPath
}

Write-Host ""
Write-Host "Setup complete. Download the pretrained checkpoint with:"
Write-Host "  .\.venv\Scripts\python.exe download_model.py --model afhqcat"
Write-Host "Then start the app with:"
Write-Host "  .\.venv\Scripts\python.exe -m streamlit run app.py"
