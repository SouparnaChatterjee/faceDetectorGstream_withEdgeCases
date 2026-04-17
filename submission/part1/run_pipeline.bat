@echo off
REM Part 1: GStreamer Pipeline
REM Usage: run_pipeline.bat [video1|video2] [test1|test2]

SET VIDEO=%1
SET TESTDIR=%2
IF "%VIDEO%"==""   SET VIDEO=video1
IF "%TESTDIR%"=="" SET TESTDIR=test1

SET ROOT=C:\dev\face_detection_project
SET INPUT=%ROOT%\videos\%VIDEO%.mp4
SET OUTPUT=%ROOT%\%TESTDIR%\frames

IF NOT EXIST "%INPUT%" (
    echo [ERROR] Video not found: %INPUT%
    pause & exit /b 1
)

WHERE gst-launch-1.0 >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo [ERROR] gst-launch-1.0 not found in PATH
    pause & exit /b 1
)

IF NOT EXIST "%OUTPUT%" mkdir "%OUTPUT%"

REM Convert backslashes to forward slashes for GStreamer
SET INPUT_FWD=%INPUT:\=/%
SET OUTPUT_FWD=%OUTPUT:\=/%

echo ================================================
echo   GStreamer Frame Extractor
echo   Input  : %INPUT%
echo   Output : %OUTPUT%\frame_%%05d.jpg
echo ================================================

gst-launch-1.0 -v filesrc location="%INPUT_FWD%" ! decodebin ! videoconvert ! videoscale ! "video/x-raw,width=640,height=640" ! jpegenc quality=85 ! multifilesink location="%OUTPUT_FWD%/frame_%%05d.jpg"

IF %ERRORLEVEL% EQU 0 (
    echo [OK] Done! Frames saved to %OUTPUT%\
) ELSE (
    echo [ERROR] Pipeline failed
    pause & exit /b 1
)
pause
