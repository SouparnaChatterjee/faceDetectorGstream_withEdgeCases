C++ Internship Assignment
Submitted by: Souparna
================================================


PART 1 - GStreamer Pipeline
===========================

I used gst-launch-1.0 to build a pipeline that reads an MP4 file,
scales each frame down to 640x640 resolution and saves them as
individual JPEG files.

To run on Linux:
    chmod +x part1/run_pipeline.sh
    ./part1/run_pipeline.sh

To run on Windows:
    From command prompt: part1\run_pipeline.bat video1 test1

The pipeline elements used:
    filesrc       reads the input mp4 file
    decodebin     automatically detects and decodes the video
    videoconvert  converts colorspace for downstream elements
    videoscale    resizes frames to 640x640
    jpegenc       encodes each frame as JPEG at quality 85
    multifilesink writes individual frame_00000.jpg files

Sample frames showing Part 1 output are in part1/sample_frames/


PART 2 - Face Detection Application
=====================================

I wrote a C++ application that reads the JPEG frames output from
Part 1 and detects faces in each frame using OpenCV Haar cascade
classifiers. Detected faces are cropped and saved in a folder
structure organized by frame number.

Source code  : part2/src/face_detection_advanced.cpp
Build steps  : part2/BUILD_INSTRUCTIONS.txt

How to run after building:

    face_detection.exe <frames_folder> <output_folder> <frontal_xml> <profile_xml>

    Example:
    face_detection.exe test1\frames test1\output data\haarcascade_frontalface_default.xml data\haarcascade_profileface.xml

Output folder structure:
    output/
        frame_1/
            face_1.jpg
            face_2.jpg
        frame_2/
            face_1.jpg

Beyond the basic requirement I also added the following:

1. Blur detection
   Frames that are too blurry due to motion or scene transitions
   are automatically skipped. This avoids false detections on
   frames where the face is not visible clearly anyway.

2. Profile face detection
   Added a second Haar cascade for side profile faces so people
   looking left or right are also detected, not just frontal faces.

3. Rotation scanning
   The detector runs at 0, -20 and +20 degree rotations to catch
   slightly tilted heads that the frontal cascade would miss.

4. CLAHE preprocessing
   Used CLAHE instead of standard equalizeHist for better contrast
   enhancement, especially in dark or unevenly lit frames.

5. Duplicate removal using NMS
   When multiple detections overlap the same face, only the best
   one is kept using Intersection over Union based suppression.

6. Face count cap
   Hard limit of 3 faces per frame to prevent false positives
   from flooding the output folder.

Results on the two test videos:

    video1  270 frames processed   226 had faces   502 crops saved
    video2  345 frames processed   100 had faces   139 crops saved

    Note: video2 had a long blurry transition sequence in the
    middle which was correctly skipped by the blur filter.
    That is why the detection rate is lower for video2.

