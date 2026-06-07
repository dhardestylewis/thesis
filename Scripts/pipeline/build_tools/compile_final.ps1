# Build final thesis PDF with full 3-pass compile.
$ErrorActionPreference = "Stop"
$final_dir = (Resolve-Path (Join-Path $PSScriptRoot "..\..\Thesis_Draft\GSAPP_Final_Submission")).Path
Push-Location $final_dir

$jobName = "Lewis_Daniel_GSAPPUP2026_Thesis"
$texFile = "$jobName.tex"
$pdfFile = "$jobName.pdf"

Write-Host "=== Pass 1: pdflatex ===" -ForegroundColor Cyan
pdflatex -interaction=nonstopmode -jobname $jobName $texFile
if ($LASTEXITCODE -ne 0) { Write-Error "Pass 1 failed"; Pop-Location; exit 1 }

Write-Host "=== bibtex ===" -ForegroundColor Cyan
bibtex $jobName
if ($LASTEXITCODE -gt 1) { Write-Host "bibtex warning/failure (expected if no cites yet, check log)" -ForegroundColor Yellow }

Write-Host "=== Pass 2: pdflatex ===" -ForegroundColor Cyan
pdflatex -interaction=nonstopmode -jobname $jobName $texFile
if ($LASTEXITCODE -ne 0) { Write-Error "Pass 2 failed"; Pop-Location; exit 1 }

Write-Host "=== Pass 3: pdflatex ===" -ForegroundColor Cyan
pdflatex -interaction=nonstopmode -jobname $jobName $texFile
if ($LASTEXITCODE -ne 0) { Write-Error "Pass 3 failed"; Pop-Location; exit 1 }

$outPdf = "${jobName}.pdf"
$sizeKB = [math]::Round((Get-Item $outPdf).length / 1KB, 0)

Write-Host ""
Write-Host "=== COMPILE COMPLETE ===" -ForegroundColor Green
Write-Host "Output: $outPdf (${sizeKB} KB)" -ForegroundColor Green

Pop-Location
