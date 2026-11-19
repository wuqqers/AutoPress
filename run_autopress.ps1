$ErrorActionPreference = 'Stop'
$python = Join-Path $PSScriptRoot '.venv-tk\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) {
    throw 'Proje sanal ortamı bulunamadı. README.md içindeki kurulum adımlarını uygulayın.'
}
Push-Location $PSScriptRoot
try {
    & $python -m autopress.app
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
