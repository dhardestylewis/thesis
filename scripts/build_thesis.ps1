# Build thesis PDF from Draft_v1. Uses alternate jobname if the default PDF is locked (e.g. open in a viewer).
# Usage (repo root): pwsh -File scripts/build_thesis.ps1
$ErrorActionPreference = "Stop"
$draft = (Resolve-Path (Join-Path $PSScriptRoot "..\Thesis_Draft\Draft_v1")).Path
Push-Location $draft
$defaultPdf = Join-Path $draft "Austin_NIMBY_Thesis_Draft.pdf"
$job = "Austin_NIMBY_Thesis_Draft"
if (Test-Path $defaultPdf) {
    try {
        [System.IO.File]::OpenWrite($defaultPdf).Close()
    } catch {
        $job = "Austin_NIMBY_Thesis_Draft_build"
        Write-Host "Default PDF appears locked; building as ${job}.pdf"
    }
}

& latexmk -pdf -interaction=nonstopmode -jobname=$job Austin_NIMBY_Thesis_Draft.tex
$code = $LASTEXITCODE
Pop-Location
exit $code
