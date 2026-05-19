# Build thesis PDF from Draft_v1 with full 3-pass compile for cross-references.
# Usage (repo root): powershell -File Scripts/pipeline/compile_thesis.ps1
$ErrorActionPreference = "Stop"
$draft = (Resolve-Path (Join-Path $PSScriptRoot "..\..\Thesis_Draft\Draft_v1")).Path
Push-Location $draft

$jobName = "Lewis_2026_NIMBYism_Austin_Thesis"
$texFile = "$jobName.tex"
$pdfFile = "$jobName.pdf"

# Use _build suffix if the canonical PDF is locked (e.g. open in a viewer)
if (Test-Path $pdfFile) {
    try { [System.IO.File]::OpenWrite($pdfFile).Close() }
    catch {
        $jobName = "${jobName}_build"
        Write-Host "PDF locked; building as ${jobName}.pdf" -ForegroundColor Yellow
    }
}

# NOTE: pass -jobname and value as SEPARATE args (not -jobname=value) to avoid
# PowerShell passing the literal string '$jobName' to pdflatex.
Write-Host "=== Pass 1: pdflatex ===" -ForegroundColor Cyan
pdflatex -interaction=nonstopmode -jobname $jobName $texFile
if ($LASTEXITCODE -ne 0) { Write-Error "Pass 1 failed"; Pop-Location; exit 1 }

Write-Host "=== bibtex ===" -ForegroundColor Cyan
bibtex $jobName
if ($LASTEXITCODE -gt 1) { Write-Error "bibtex failed"; Pop-Location; exit 1 }

Write-Host "=== Pass 2: pdflatex ===" -ForegroundColor Cyan
pdflatex -interaction=nonstopmode -jobname $jobName $texFile
if ($LASTEXITCODE -ne 0) { Write-Error "Pass 2 failed"; Pop-Location; exit 1 }

Write-Host "=== Pass 3: pdflatex ===" -ForegroundColor Cyan
pdflatex -interaction=nonstopmode -jobname $jobName $texFile
if ($LASTEXITCODE -ne 0) { Write-Error "Pass 3 failed"; Pop-Location; exit 1 }

$outPdf = "${jobName}.pdf"
$sizeKB = [math]::Round((Get-Item $outPdf).length / 1KB, 0)
$pages  = (Select-String "Output written on" $outPdf -SimpleMatch -ErrorAction SilentlyContinue)

Write-Host ""
Write-Host "=== COMPILE COMPLETE ===" -ForegroundColor Green
Write-Host "Output: $outPdf  (${sizeKB} KB)" -ForegroundColor Green

# Summarise log quality
$log       = Get-Content "${jobName}.log" -Raw
$undefined = ([regex]::Matches($log, "undefined")).Count
$rerun     = ([regex]::Matches($log, "Rerun")).Count
Write-Host "Log: undefined=$undefined  rerun=$rerun" -ForegroundColor Yellow

Pop-Location
