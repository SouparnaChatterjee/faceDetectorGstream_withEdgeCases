#include <opencv2/opencv.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <filesystem>
#include <string>
#include <vector>

namespace fs = std::filesystem;

int main(int argc, char* argv[]) {
    // ── Config ────────────────────────────────────────────────────────
    std::string framesDir   = "frames";
    std::string outputDir   = "output";
    std::string cascadePath = "data/haarcascade_frontalface_default.xml";

    // Allow command-line override
    if (argc > 1) framesDir   = argv[1];
    if (argc > 2) outputDir   = argv[2];
    if (argc > 3) cascadePath = argv[3];

    // ── Load Haar Cascade ─────────────────────────────────────────────
    cv::CascadeClassifier faceCascade;
    if (!faceCascade.load(cascadePath)) {
        std::cerr << "❌ Error: Could not load cascade: " << cascadePath << std::endl;
        return -1;
    }
    std::cout << "✅ Loaded Haar Cascade: " << cascadePath << std::endl;

    // ── Create Output Directory ───────────────────────────────────────
    fs::create_directories(outputDir);

    // ── Process Each Frame ────────────────────────────────────────────
    int frameCount   = 0;
    int faceCount    = 0;
    int detectedFrames = 0;

    std::vector<fs::path> frameFiles;
    for (const auto& entry : fs::directory_iterator(framesDir)) {
        if (entry.path().extension() == ".jpg" ||
            entry.path().extension() == ".jpeg" ||
            entry.path().extension() == ".png") {
            frameFiles.push_back(entry.path());
        }
    }

    // Sort frames in order
    std::sort(frameFiles.begin(), frameFiles.end());
    std::cout << "📂 Found " << frameFiles.size() << " frames in: " << framesDir << std::endl;

    for (const auto& framePath : frameFiles) {
        cv::Mat frame = cv::imread(framePath.string());
        if (frame.empty()) {
            std::cerr << "⚠️  Could not read: " << framePath << std::endl;
            continue;
        }

        frameCount++;

        // Convert to grayscale for detection
        cv::Mat gray;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        cv::equalizeHist(gray, gray);

        // Detect faces
        std::vector<cv::Rect> faces;
        faceCascade.detectMultiScale(
            gray, faces,
            1.1,   // scaleFactor
            4,     // minNeighbors
            0,     // flags
            cv::Size(30, 30)  // minSize
        );

        if (!faces.empty()) {
            detectedFrames++;
            std::cout << "🔍 Frame " << frameCount
                      << " (" << framePath.filename().string() << "): "
                      << faces.size() << " face(s) detected" << std::endl;
        }

        // ── Crop & Save Each Detected Face ────────────────────────────
        for (size_t i = 0; i < faces.size(); i++) {
            cv::Rect faceRect = faces[i];

            // Add padding (20%) around face
            int padX = static_cast<int>(faceRect.width  * 0.2);
            int padY = static_cast<int>(faceRect.height * 0.2);

            // Clamp to image bounds
            int x = std::max(0, faceRect.x - padX);
            int y = std::max(0, faceRect.y - padY);
            int w = std::min(frame.cols - x, faceRect.width  + 2 * padX);
            int h = std::min(frame.rows - y, faceRect.height + 2 * padY);

            cv::Rect paddedRect(x, y, w, h);
            cv::Mat  faceCrop = frame(paddedRect);

            // Build output filename: face_frame0001_0.jpg
            char outName[256];
            snprintf(outName, sizeof(outName),
                "%s/face_frame%05d_%zu.jpg",
                outputDir.c_str(), frameCount, i);

            cv::imwrite(outName, faceCrop);
            faceCount++;
        }
    }

    // ── Summary ───────────────────────────────────────────────────────
    std::cout << "\n========== SUMMARY ==========" << std::endl;
    std::cout << "Total frames processed : " << frameCount       << std::endl;
    std::cout << "Frames with faces      : " << detectedFrames   << std::endl;
    std::cout << "Total face crops saved : " << faceCount        << std::endl;
    std::cout << "Output directory       : " << outputDir        << std::endl;
    std::cout << "==============================" << std::endl;

    return 0;
}
