@echo off
REM Part 1: GStreamer Pipeline (Windows Batch Version)
SET INPUT_FILE=input.mp4
SET OUTPUT_DIR=frames

IF NOT EXIST %OUTPUT_DIR% mkdir %OUTPUT_DIR%

echo Starting GStreamer pipeline...
echo Input: %INPUT_FILE%
echo Output: %OUTPUT_DIR%\frame_%%05d.jpg

gst-launch-1.0 -v ^
    filesrc location=%INPUT_FILE% ^
    ! decodebin ^
    ! videoconvert ^
    ! videoscale ^
    ! "video/x-raw,width=640,height=640" ^
    ! jpegenc quality=85 ^
    ! multifilesink location="%OUTPUT_DIR%/frame_%%05d.jpg"

echo Done! Check frames folder.
pause
