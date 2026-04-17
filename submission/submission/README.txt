================================================
  C++ Internship Assignment Submission
================================================

PART 1 - GStreamer Pipeline
---------------------------
Reads MP4 video, scales to 640x640, writes JPEG frames.

Linux:
  chmod +x part1/run_pipeline.sh
  ./part1/run_pipeline.sh input.mp4 frames/

Windows:
  part1\run_pipeline.bat input.mp4 frames

Pipeline:
  filesrc -> decodebin -> videoconvert -> videoscale
  -> video/x-raw,width=640,height=640
  -> jpegenc quality=85
  -> multifilesink

Sample output: part1/sample_frames/ (first 5 frames)

------------------------------------------------

PART 2 - Face Detection C++ Application
----------------------------------------
Reads JPEG frames, detects faces using Haar cascades,
crops and saves in frame-wise folder structure.

Source : part2/src/face_detection_advanced.cpp
Build  : See part2/BUILD_INSTRUCTIONS.txt

Enhancements beyond basic requirement:
  1. Blur filtering  - skip motion-blurred frames
  2. CLAHE           - better contrast than equalizeHist
  3. Profile cascade - catches side-facing heads
  4. Rotation scan   - handles tilted heads (+-20 degrees)
  5. IoU-based NMS   - removes duplicate detections
  6. Hard face cap   - prevents false-positive explosion

Output structure:
  output/
    frame_1/
      face_1.jpg
      face_2.jpg
    frame_2/
      face_1.jpg

Results:
  video1: 270 frames, 226 with faces, 502 face crops
  video2: 345 frames, 100 with faces, 139 face crops

================================================

