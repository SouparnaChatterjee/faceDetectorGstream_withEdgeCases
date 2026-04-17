#!/bin/bash
# Part 1: GStreamer Pipeline
# Reads MP4 video, scales to 640x640, outputs JPEG frames

INPUT_FILE="${1:-input.mp4}"   # ← accept argument, fallback to input.mp4
OUTPUT_DIR="${2:-frames}"       # ← accept argument, fallback to frames

mkdir -p "$OUTPUT_DIR"

# ── Guards ──────────────────────────────────────────────
if [ ! -f "$INPUT_FILE" ]; then
    echo "[ERROR] File not found: $INPUT_FILE"
    exit 1
fi

if ! command -v gst-launch-1.0 &>/dev/null; then
    echo "[ERROR] gst-launch-1.0 not installed"
    exit 1
fi

echo "Starting GStreamer pipeline..."
echo "Input : $INPUT_FILE"
echo "Output: $OUTPUT_DIR/frame_%05d.jpg"

gst-launch-1.0 -v \
    filesrc location="$INPUT_FILE" ! \
    decodebin ! \
    videoconvert ! \
    videoscale ! \
    "video/x-raw,width=640,height=640" ! \  # ← quote the caps string
    jpegenc quality=85 ! \
    multifilesink location="$OUTPUT_DIR/frame_%05d.jpg"

# ── Result ──────────────────────────────────────────────
if [ $? -eq 0 ]; then
    COUNT=$(ls "$OUTPUT_DIR"/*.jpg 2>/dev/null | wc -l)
    echo "Done! $COUNT frames saved in $OUTPUT_DIR/"
    ls -la "$OUTPUT_DIR/" | head -20
else
    echo "[ERROR] Pipeline failed"
    exit 1
fi