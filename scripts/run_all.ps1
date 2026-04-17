# run_all.ps1
# Runs Part 1 (GStreamer) and Part 2 (Face Detection) for both videos
# Usage: .\scripts\run_all.ps1
# Run from project root: C:\dev\face_detection_project\

$ROOT    = "C:\dev\face_detection_project"
$EXE     = "$ROOT\build\Release\face_detection.exe"
$FRONTAL = "$ROOT\data\haarcascade_frontalface_default.xml"
$PROFILE = "$ROOT\data\haarcascade_profileface.xml"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Face Detection Pipeline" -ForegroundColor Cyan
Write-Host "  Root: $ROOT" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# Guard: check exe exists 
if (-not (Test-Path $EXE)) {
    Write-Host "[ERROR] Executable not found: $EXE" -ForegroundColor Red
    Write-Host "        Build the project in Visual Studio first" -ForegroundColor Red
    exit 1
}

# Guard: check XML files exist 
if (-not (Test-Path $FRONTAL)) {
    Write-Host "[ERROR] Frontal XML not found: $FRONTAL" -ForegroundColor Red
    exit 1
}
if (-not (Test-Path $PROFILE)) {
    Write-Host "[ERROR] Profile XML not found: $PROFILE" -ForegroundColor Red
    exit 1
}

Write-Host "[OK] Executable found" -ForegroundColor Green
Write-Host "[OK] XML files found`n" -ForegroundColor Green

# PART 1 — GStreamer (optional, skip if frames exist)
function Run-GStreamer {
    param($video, $testdir)

    $input  = "$ROOT\videos\$video.mp4"
    $output = "$ROOT\$testdir\frames"

    Write-Host "----------------------------------------" -ForegroundColor Yellow
    Write-Host "[Part 1] $video -> $testdir\frames\" -ForegroundColor Yellow
    Write-Host "----------------------------------------" -ForegroundColor Yellow

    if (-not (Test-Path $input)) {
        Write-Host "[SKIP] Video not found: $input" -ForegroundColor Magenta
        return
    }

    $existing = (Get-ChildItem "$output\*.jpg" -ErrorAction SilentlyContinue).Count
    if ($existing -gt 0) {
        Write-Host "[SKIP] Frames already exist ($existing frames in $output)" -ForegroundColor Magenta
        Write-Host "       Delete $output\ to re-extract" -ForegroundColor Magenta
        return
    }

    if (-not (Test-Path $output)) { New-Item -ItemType Directory -Path $output | Out-Null }

    Write-Host "[RUN ] gst-launch-1.0 on $input" -ForegroundColor Cyan

    $gstArgs = "-v filesrc location=`"$input`" ! decodebin ! videoconvert ! videoscale ! video/x-raw,width=640,height=640 ! jpegenc quality=85 ! multifilesink location=`"$output\frame_%05d.jpg`""

    $proc = Start-Process -FilePath "gst-launch-1.0" `
                          -ArgumentList $gstArgs `
                          -Wait -PassThru -NoNewWindow

    if ($proc.ExitCode -eq 0) {
        $count = (Get-ChildItem "$output\*.jpg").Count
        Write-Host "[OK ] Done! $count frames saved to $output\" -ForegroundColor Green
    } else {
        Write-Host "[ERR] GStreamer failed (exit: $($proc.ExitCode))" -ForegroundColor Red
    }
}

# PART 2 — Face Detection
function Run-FaceDetection {
    param($testdir)

    $framesDir = "$ROOT\$testdir\frames"
    $outputDir = "$ROOT\$testdir\output"

    Write-Host "----------------------------------------" -ForegroundColor Yellow
    Write-Host "[Part 2] Face detection on $testdir" -ForegroundColor Yellow
    Write-Host "----------------------------------------" -ForegroundColor Yellow

    $frameCount = (Get-ChildItem "$framesDir\*.jpg" -ErrorAction SilentlyContinue).Count
    if ($frameCount -eq 0) {
        Write-Host "[SKIP] No frames found in $framesDir" -ForegroundColor Magenta
        Write-Host "       Run Part 1 first" -ForegroundColor Magenta
        return
    }

    Write-Host "[RUN ] Processing $frameCount frames from $framesDir" -ForegroundColor Cyan

    & $EXE $framesDir $outputDir $FRONTAL $PROFILE

    if ($LASTEXITCODE -eq 0) {
        $faceFolders = (Get-ChildItem "$outputDir" -Directory -ErrorAction SilentlyContinue).Count
        $faceFiles   = (Get-ChildItem "$outputDir\*\*.jpg" -ErrorAction SilentlyContinue).Count
        Write-Host "[OK ] $faceFolders frame folders, $faceFiles face crops in $outputDir\" -ForegroundColor Green
    } else {
        Write-Host "[ERR] Face detection failed" -ForegroundColor Red
    }
}

# RUN BOTH VIDEOS

Write-Host "`n===== VIDEO 1 =====" -ForegroundColor Cyan
Run-GStreamer     "video1" "test1"
Run-FaceDetection "test1"

Write-Host "`n===== VIDEO 2 =====" -ForegroundColor Cyan
Run-GStreamer     "video2" "test2"
Run-FaceDetection "test2"

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "  ALL DONE!" -ForegroundColor Green
Write-Host "  test1\output\ -> video1 face crops" -ForegroundColor Green
Write-Host "  test2\output\ -> video2 face crops" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan