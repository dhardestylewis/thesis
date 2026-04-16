# Build thesis PDF from Draft_v1. Uses alternate jobname if the default PDF is locked (e.g. open in a viewer).
# Usage (repo root): powershell -File scripts/pipeline/build_thesis.ps1
$ErrorActionPreference = "Stop"
$draft = (Resolve-Path (Join-Path $PSScriptRoot "..\..\Thesis_Draft\Draft_v1")).Path
Push-Location $draft
$defaultPdf = Join-Path $draft "Lewis_2026_NIMBYism_Austin_Thesis.pdf"
$job = "Lewis_2026_NIMBYism_Austin_Thesis"
if (Test-Path $defaultPdf) {
    try {
        [System.IO.File]::OpenWrite($defaultPdf).Close()
    } catch {
        $job = "Lewis_2026_NIMBYism_Austin_Thesis_build"
        Write-Host "Default PDF appears locked; building as ${job}.pdf"
    }
}

& latexmk -pdf -interaction=nonstopmode -jobname=$job Lewis_2026_NIMBYism_Austin_Thesis.tex
$code = $LASTEXITCODE
Pop-Location
exit $code
