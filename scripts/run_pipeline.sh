#!/bin/bash
# Part 1: GStreamer Pipeline
# Reads MP4 video, scales to 640x640, outputs JPEG frames

INPUT_FILE="input.mp4"
OUTPUT_DIR="frames"

mkdir -p "$OUTPUT_DIR"

echo "Starting GStreamer pipeline..."
echo "Input: $INPUT_FILE"
echo "Output: $OUTPUT_DIR/frame_%05d.jpg"

gst-launch-1.0 -v \
    filesrc location="$INPUT_FILE" ! \
    decodebin ! \
    videoconvert ! \
    videoscale ! \
    video/x-raw,width=640,height=640 ! \
    jpegenc quality=85 ! \
    multifilesink location="$OUTPUT_DIR/frame_%05d.jpg"

echo "Pipeline complete! Frames saved in $OUTPUT_DIR/"
ls -la "$OUTPUT_DIR/" | head -20
