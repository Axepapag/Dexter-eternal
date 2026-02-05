#!/usr/bin/env powershell
# Dexter TRM Status Checker
# Run this anytime to see training progress: .\check_status.ps1

param(
    [switch]$Watch,
    [int]$Interval = 10
)

function Get-Status {
    Clear-Host
    $width = 70
    
    # Header
    Write-Host ("=" * $width) -ForegroundColor Cyan
    Write-Host "    DEXTER TRM TRAINING STATUS MONITOR" -ForegroundColor Cyan
    Write-Host ("=" * $width) -ForegroundColor Cyan
    Write-Host "    $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')`n" -ForegroundColor Gray
    
    $baseDir = "D:\Dexter-Eternal"
    
    # 1. Check Running Processes
    Write-Host "[1] TRAINING PROCESSES" -ForegroundColor Yellow
    Write-Host ("-" * $width) -ForegroundColor DarkGray
    
    $pythonProcs = Get-Process python -ErrorAction SilentlyContinue
    if ($pythonProcs) {
        Write-Host "    ✓ Python is RUNNING!" -ForegroundColor Green
        foreach ($proc in $pythonProcs) {
            $cpu = [math]::Round($proc.CPU, 2)
            $mem = [math]::Round($proc.WorkingSet64 / 1MB, 2)
            Write-Host "    PID $($proc.Id): CPU=${cpu}s, RAM=${mem} MB" -ForegroundColor Green
        }
    } else {
        Write-Host "    ✗ No Python processes found" -ForegroundColor Red
        Write-Host "    Training may be complete or not started" -ForegroundColor Yellow
    }
    Write-Host ""
    
    # 2. Check Dataset Files
    Write-Host "[2] DATASET STATUS" -ForegroundColor Yellow
    Write-Host ("-" * $width) -ForegroundColor DarkGray
    
    $datasets = @(
        @{Name="Memory"; Path="$baseDir\dexter_TRMs\datasets\offline\memory\memory_training_data.npz"; MinSize=100KB},
        @{Name="Tool"; Path="$baseDir\dexter_TRMs\datasets\offline\tool\tool_training_data.npz"; MinSize=1MB},
        @{Name="Reasoning"; Path="$baseDir\dexter_TRMs\datasets\offline\reasoning\reasoning_training_data.npz"; MinSize=500KB}
    )
    
    $datasetReady = $true
    foreach ($ds in $datasets) {
        if (Test-Path $ds.Path) {
            $size = (Get-Item $ds.Path).Length
            $sizeMB = [math]::Round($size / 1MB, 2)
            $sizeKB = [math]::Round($size / 1KB, 2)
            
            if ($size -ge $ds.MinSize) {
                Write-Host "    ✓ $($ds.Name): $sizeMB MB ($sizeKB KB)" -ForegroundColor Green
            } else {
                Write-Host "    ⚠ $($ds.Name): $sizeMB MB (small)" -ForegroundColor Yellow
            }
            
            # Count examples in NPZ
            try {
                $npz = [System.IO.File]::OpenRead($ds.Path)
                $npz.Close()
            } catch {}
        } else {
            Write-Host "    ✗ $($ds.Name): Not created" -ForegroundColor Red
            $datasetReady = $false
        }
    }
    Write-Host ""
    
    # 3. Check Vocabulary
    Write-Host "[3] VOCABULARY STATUS" -ForegroundColor Yellow
    Write-Host ("-" * $width) -ForegroundColor DarkGray
    
    $vocabFiles = @(
        "$baseDir\dexter_TRMs\datasets\offline\unified_vocab.json",
        "$baseDir\dexter_TRMs\datasets\offline\memory\memory_vocab.json"
    )
    
    foreach ($vf in $vocabFiles) {
        if (Test-Path $vf) {
            try {
                $json = Get-Content $vf -Raw | ConvertFrom-Json
                if ($json.token2id) {
                    $size = ($json.token2id | Get-Member -MemberType NoteProperty).Count
                    Write-Host "    ✓ $(Split-Path $vf -Leaf): $size tokens" -ForegroundColor Green
                } elseif ($json.PSObject.Properties['size']) {
                    Write-Host "    ✓ $(Split-Path $vf -Leaf): $($json.size) tokens" -ForegroundColor Green
                } else {
                    Write-Host "    ⚠ $(Split-Path $vf -Leaf): Unknown format" -ForegroundColor Yellow
                }
            } catch {
                Write-Host "    ⚠ $(Split-Path $vf -Leaf): Could not parse" -ForegroundColor Yellow
            }
        }
    }
    Write-Host ""
    
    # 4. Check Trained Models
    Write-Host "[4] TRAINED MODELS" -ForegroundColor Yellow
    Write-Host ("-" * $width) -ForegroundColor DarkGray
    
    $modelDirs = @(
        "$baseDir\dexter_TRMs\models"
    )
    
    $foundModels = $false
    foreach ($mdir in $modelDirs) {
        if (Test-Path $mdir) {
            $models = Get-ChildItem $mdir -Filter "*.pt" -ErrorAction SilentlyContinue
            if ($models) {
                $foundModels = $true
                Write-Host "    Directory: $(Split-Path $mdir -Leaf)" -ForegroundColor Cyan
                foreach ($m in $models | Sort-Object Length -Descending) {
                    $sizeMB = [math]::Round($m.Length / 1MB, 2)
                    $age = [math]::Round(((Get-Date) - $m.LastWriteTime).TotalMinutes, 1)
                    Write-Host "      ✓ $($m.Name)" -ForegroundColor Green -NoNewline
                    Write-Host " ($sizeMB MB, ${age}m ago)" -ForegroundColor Gray
                }
            }
        }
    }
    
    if (-not $foundModels) {
        Write-Host "    ✗ No trained models found yet" -ForegroundColor Red
    }
    Write-Host ""
    
    # 5. Check Training Logs
    Write-Host "[5] TRAINING LOG" -ForegroundColor Yellow
    Write-Host ("-" * $width) -ForegroundColor DarkGray
    
    $logFiles = @(
        "$baseDir\training_log.txt",
        "$baseDir\dexter_TRMs\logs\master.log"
    )
    
    $foundLog = $false
    foreach ($lf in $logFiles) {
        if (Test-Path $lf) {
            $foundLog = $true
            $lines = Get-Content $lf -Tail 5 -ErrorAction SilentlyContinue
            if ($lines) {
                Write-Host "    Last 5 lines from $(Split-Path $lf -Leaf):" -ForegroundColor Cyan
                foreach ($line in $lines) {
                    # Color code based on content
                    if ($line -match "error|fail|traceback") {
                        Write-Host "      $line" -ForegroundColor Red
                    } elseif ($line -match "epoch.*loss.*acc") {
                        Write-Host "      $line" -ForegroundColor Green
                    } elseif ($line -match "✓|success|complete") {
                        Write-Host "      $line" -ForegroundColor Green
                    } else {
                        Write-Host "      $line" -ForegroundColor Gray
                    }
                }
            }
            break
        }
    }
    
    if (-not $foundLog) {
        Write-Host "    ✗ No training log found" -ForegroundColor Red
    }
    Write-Host ""
    
    # 6. System Resources
    Write-Host "[6] SYSTEM RESOURCES" -ForegroundColor Yellow
    Write-Host ("-" * $width) -ForegroundColor DarkGray
    
    try {
        $cpu = (Get-Counter '\Processor(_Total)\% Processor Time' -SampleInterval 1 -MaxSamples 1).CounterSamples.CookedValue
        $cpu = [math]::Round($cpu, 1)
        
        $mem = Get-CimInstance Win32_OperatingSystem
        $memUsed = [math]::Round(($mem.TotalVisibleMemorySize - $mem.FreePhysicalMemory) / 1MB, 2)
        $memTotal = [math]::Round($mem.TotalVisibleMemorySize / 1MB, 2)
        $memPercent = [math]::Round(($memUsed / $memTotal) * 100, 1)
        
        Write-Host "    CPU Usage: $cpu%" -NoNewline
        if ($cpu -gt 80) { Write-Host " (HIGH!)" -ForegroundColor Red }
        elseif ($cpu -gt 50) { Write-Host " (Moderate)" -ForegroundColor Yellow }
        else { Write-Host " (Normal)" -ForegroundColor Green }
        
        Write-Host "    RAM Usage: $memUsed / $memTotal GB ($memPercent%)" -NoNewline
        if ($memPercent -gt 90) { Write-Host " (CRITICAL!)" -ForegroundColor Red }
        elseif ($memPercent -gt 75) { Write-Host " (High)" -ForegroundColor Yellow }
        else { Write-Host " (OK)" -ForegroundColor Green }
    } catch {
        Write-Host "    Could not read system resources" -ForegroundColor Yellow
    }
    Write-Host ""
    
    # Summary
    Write-Host ("=" * $width) -ForegroundColor Cyan
    Write-Host "SUMMARY" -ForegroundColor Cyan
    Write-Host ("=" * $width) -ForegroundColor Cyan
    
    if ($pythonProcs -and $foundModels) {
        Write-Host "    ✓ Training IN PROGRESS with models being saved!" -ForegroundColor Green
    } elseif ($pythonProcs) {
        Write-Host "    ✓ Training RUNNING (datasets being created/models training)" -ForegroundColor Green
    } elseif ($foundModels) {
        Write-Host "    ✓ Training COMPLETE - Models ready!" -ForegroundColor Green
        Write-Host "    Next: Test with 'python trained_trm_wrappers.py'" -ForegroundColor Yellow
    } else {
        Write-Host "    ✗ Training NOT STARTED or NO OUTPUT yet" -ForegroundColor Red
        Write-Host "    Start with: python build_memory_trm_dataset.py" -ForegroundColor Yellow
        Write-Host "    Then: python train_all_trms.py" -ForegroundColor Yellow
    }
    
    Write-Host ("=" * $width) -ForegroundColor Cyan
    Write-Host ""
}

# Main execution
if ($Watch) {
    Write-Host "Starting continuous monitoring (Ctrl+C to stop)..." -ForegroundColor Cyan
    while ($true) {
        Get-Status
        Write-Host "Updating in $Interval seconds... (Ctrl+C to exit)" -ForegroundColor DarkGray
        Start-Sleep -Seconds $Interval
    }
} else {
    Get-Status
}
