& (Join-Path $PSScriptRoot 'crfid-python.ps1') -m pytest @args
exit $LASTEXITCODE
