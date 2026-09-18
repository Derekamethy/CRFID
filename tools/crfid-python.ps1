$python = $env:CRFID_PYTHON
if ([string]::IsNullOrWhiteSpace($python)) {
    $localPython = Join-Path $PSScriptRoot '..\.venv\Scripts\python.exe'
    if (Test-Path -LiteralPath $localPython -PathType Leaf) {
        $python = (Resolve-Path -LiteralPath $localPython).Path
    } else {
        $command = Get-Command python -ErrorAction SilentlyContinue
        if ($null -ne $command) {
            $python = $command.Source
        }
    }
}
if ([string]::IsNullOrWhiteSpace($python) -or -not (Test-Path -LiteralPath $python -PathType Leaf)) {
    Write-Error 'Set CRFID_PYTHON to a Python executable or create .venv.'
    exit 127
}
& $python @args
exit $LASTEXITCODE
