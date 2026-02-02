# Ollama Model Cleanup Script
# Run from PowerShell on Windows

Write-Host "=== Ollama Model Cleanup ===" -ForegroundColor Cyan
Write-Host ""

# List current models
Write-Host "Current models:" -ForegroundColor Yellow
ollama list

Write-Host ""
Write-Host "Models to KEEP:" -ForegroundColor Green
Write-Host "  - qwen2.5:14b (PRIMARY - used by RAG-Hybrid)"
Write-Host ""

Write-Host "Models to REMOVE (if installed):" -ForegroundColor Red
Write-Host "  - llama3 (legacy)"
Write-Host "  - meta-llama3-8b (legacy)"
Write-Host "  - tinyllama-1b (legacy)"
Write-Host "  - qwen3 (testing, not needed)"
Write-Host ""

$confirm = Read-Host "Remove unused models? (y/N)"

if ($confirm -eq "y" -or $confirm -eq "Y") {
    # Remove legacy models
    $modelsToRemove = @("llama3", "meta-llama3-8b", "tinyllama-1b", "qwen3")

    foreach ($model in $modelsToRemove) {
        Write-Host "Removing $model..." -ForegroundColor Yellow
        ollama rm $model 2>$null
        if ($LASTEXITCODE -eq 0) {
            Write-Host "  Removed $model" -ForegroundColor Green
        } else {
            Write-Host "  $model not installed, skipping" -ForegroundColor Gray
        }
    }

    Write-Host ""
    Write-Host "Cleanup complete!" -ForegroundColor Green
} else {
    Write-Host "Cancelled." -ForegroundColor Gray
}

Write-Host ""
Write-Host "Remaining models:" -ForegroundColor Yellow
ollama list
