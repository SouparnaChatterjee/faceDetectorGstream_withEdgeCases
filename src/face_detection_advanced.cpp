// face_detection_advanced.cpp
// Enhanced face detection with:
//   1. Blur filtering     — skip low-quality/motion-blurred frames
//   2. CLAHE              — better contrast than equalizeHist
//   3. Profile cascade    — catches side-facing heads
//   4. Rotation scanning  — handles tilted heads (±20°)
//   5. IoU-based NMS      — removes duplicate detections
//   6. Max-face hard cap  — prevents false-positive explosion
//
// Compile:
//   g++ -std=c++17 face_detection_advanced.cpp -o face_detection_adv \
//       $(pkg-config --cflags --libs opencv4)

#include <opencv2/opencv.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <filesystem>
#include <iomanip>

namespace fs = std::filesystem;


// Config — all tunable parameters in one place
struct Config {
    // Paths
    std::string framesDir   = "frames";
    std::string outputDir   = "output";
    std::string frontalPath = "data/haarcascade_frontalface_default.xml";
    std::string profilePath = "data/haarcascade_profileface.xml";

    // Detection parameters
    double scaleFactor      = 1.1;   // pyramid scale step
    int    minNeighbors     = 6;     // higher = fewer false positives
    int    minFaceSize      = 40;    // ignore detections smaller than this
    int    maxFaceSize      = 500;   // ignore detections larger than this
    int    maxFacesPerFrame = 3;     // hard cap adjust to video

    // Quality
    double blurThreshold    = 80.0;  // Laplacian variance-> below = skip frame
    double nmsIoUThreshold  = 0.25;  // overlap threshold for NMS
    int    paddingPercent   = 10;    // % padding around face crop

    // Rotation angles to scan — 0 always first (upright)
    std::vector<int> rotationAngles = {0, -20, 20};
};

// Utilities

// Collect and sort all image files in a directory
std::vector<fs::path> getImageFiles(const std::string& dir) {
    std::vector<fs::path> files;
    if (!fs::exists(dir)) return files;
    for (const auto& e : fs::directory_iterator(dir)) {
        std::string ext = e.path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        if (ext == ".jpg" || ext == ".jpeg" || ext == ".png")
            files.push_back(e.path());
    }
    std::sort(files.begin(), files.end());
    return files;
}

// Laplacian variance — measures image sharpness
// High value = sharp.  Low value = blurry.
double computeBlurScore(const cv::Mat& gray) {
    cv::Mat lap;
    cv::Laplacian(gray, lap, CV_64F);
    cv::Scalar mean, stddev;
    cv::meanStdDev(lap, mean, stddev);
    return stddev[0] * stddev[0];
}

// CLAHE: Contrast Limited Adaptive Histogram Equalization
// Avoids noise amplification in uniform regions unlike equalizeHist
cv::Mat applyCLAHE(const cv::Mat& gray) {
    auto clahe = cv::createCLAHE(2.0, cv::Size(8, 8));
    cv::Mat out;
    clahe->apply(gray, out);
    return out;
}

// Intersection over Union between two rectangles
double computeIoU(const cv::Rect& a, const cv::Rect& b) {
    cv::Rect inter = a & b;
    if (inter.area() == 0) return 0.0;
    return (double)inter.area() / (a.area() + b.area() - inter.area());
}

// Non-Maximum Suppression
// Removes overlapping boxes, keeping the largest (highest area = most confident)
std::vector<cv::Rect> applyNMS(std::vector<cv::Rect> rects, double iouThresh) {
    std::sort(rects.begin(), rects.end(),
        [](const cv::Rect& a, const cv::Rect& b){ return a.area() > b.area(); });

    std::vector<bool> suppressed(rects.size(), false);
    std::vector<cv::Rect> result;
    for (size_t i = 0; i < rects.size(); i++) {
        if (suppressed[i]) continue;
        result.push_back(rects[i]);
        for (size_t j = i + 1; j < rects.size(); j++)
            if (!suppressed[j] && computeIoU(rects[i], rects[j]) > iouThresh)
                suppressed[j] = true;
    }
    return result;
}

// Rotate image by angle (fills border with gray = 128)
cv::Mat rotateImage(const cv::Mat& src, double angle) {
    if (angle == 0.0) return src.clone();
    cv::Point2f center(src.cols / 2.0f, src.rows / 2.0f);
    cv::Mat M = cv::getRotationMatrix2D(center, angle, 1.0);
    cv::Mat dst;
    cv::warpAffine(src, dst, M, src.size(),
                   cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(128));
    return dst;
}

// Map bounding box from rotated-image coordinates back to original
cv::Rect unrotateRect(const cv::Rect& r, double angle, cv::Size sz) {
    if (angle == 0.0) return r;
    cv::Point2f center(sz.width / 2.0f, sz.height / 2.0f);
    cv::Mat M = cv::getRotationMatrix2D(center, -angle, 1.0);
    std::vector<cv::Point2f> corners = {
        {(float)r.x,           (float)r.y           },
        {(float)(r.x+r.width), (float)r.y            },
        {(float)r.x,           (float)(r.y+r.height) },
        {(float)(r.x+r.width), (float)(r.y+r.height) }
    };
    std::vector<cv::Point2f> mapped;
    cv::transform(corners, mapped, M);
    float x0 = std::min({mapped[0].x,mapped[1].x,mapped[2].x,mapped[3].x});
    float y0 = std::min({mapped[0].y,mapped[1].y,mapped[2].y,mapped[3].y});
    float x1 = std::max({mapped[0].x,mapped[1].x,mapped[2].x,mapped[3].x});
    float y1 = std::max({mapped[0].y,mapped[1].y,mapped[2].y,mapped[3].y});
    return {(int)x0, (int)y0, (int)(x1-x0), (int)(y1-y0)};
}

// Add padding around face rect, clamped to image bounds
cv::Rect safePadded(const cv::Rect& face, int padPct, int imgW, int imgH) {
    int px = face.width  * padPct / 100;
    int py = face.height * padPct / 100;
    int x  = std::max(0, face.x - px);
    int y  = std::max(0, face.y - py);
    int w  = std::min(imgW - x, face.width  + 2 * px);
    int h  = std::min(imgH - y, face.height + 2 * py);
    return {x, y, w, h};
}

// Keep only the N largest detections
std::vector<cv::Rect> keepTopN(std::vector<cv::Rect> rects, int n) {
    if ((int)rects.size() <= n) return rects;
    std::sort(rects.begin(), rects.end(),
        [](const cv::Rect& a, const cv::Rect& b){ return a.area() > b.area(); });
    rects.resize(n);
    return rects;
}

// Core Detection
// Runs frontal + profile (left + right mirror) at each
// rotation angle, then deduplicates with NMS
std::vector<cv::Rect> detectFaces(
    const cv::Mat& gray,
    cv::CascadeClassifier& frontal,
    cv::CascadeClassifier& profile,
    const Config& cfg)
{
    std::vector<cv::Rect> all;
    const cv::Size minSz(cfg.minFaceSize, cfg.minFaceSize);
    const cv::Size maxSz(cfg.maxFaceSize, cfg.maxFaceSize);

    for (int angle : cfg.rotationAngles) {

        // Rotate and enhance
        cv::Mat rot      = rotateImage(gray, angle);
        cv::Mat enhanced = applyCLAHE(rot);

        // Frontal detection 
        std::vector<cv::Rect> frontalHits;
        frontal.detectMultiScale(enhanced, frontalHits,
            cfg.scaleFactor, cfg.minNeighbors, 0, minSz, maxSz);

        // Left profile 
        std::vector<cv::Rect> leftHits;
        profile.detectMultiScale(enhanced, leftHits,
            cfg.scaleFactor, cfg.minNeighbors, 0, minSz, maxSz);

        // Right profile (horizontal flip trick) ──
        // Profile cascade only detects one side;
        // flip image to catch the other side, then mirror coords back
        cv::Mat flipped;
        cv::flip(enhanced, flipped, 1);
        std::vector<cv::Rect> rightHits;
        profile.detectMultiScale(flipped, rightHits,
            cfg.scaleFactor, cfg.minNeighbors, 0, minSz, maxSz);
        for (auto& r : rightHits)
            r.x = gray.cols - r.x - r.width;  // mirror x back

        // ── Map all hits back to original image space
        for (auto& r : frontalHits) all.push_back(unrotateRect(r, angle, gray.size()));
        for (auto& r : leftHits)    all.push_back(unrotateRect(r, angle, gray.size()));
        for (auto& r : rightHits)   all.push_back(unrotateRect(r, angle, gray.size()));
    }

    // Remove duplicate/overlapping boxes
    all = applyNMS(all, cfg.nmsIoUThreshold);

    // Remove boxes outside image or too small
    cv::Rect bounds(0, 0, gray.cols, gray.rows);
    std::vector<cv::Rect> valid;
    for (const auto& r : all) {
        cv::Rect c = r & bounds;
        if (c.width  > cfg.minFaceSize &&
            c.height > cfg.minFaceSize)
            valid.push_back(c);
    }

    return keepTopN(valid, cfg.maxFacesPerFrame);
}

// main
int main(int argc, char* argv[]) {
    Config cfg;
    if (argc > 1) cfg.framesDir   = argv[1];
    if (argc > 2) cfg.outputDir   = argv[2];
    if (argc > 3) cfg.frontalPath = argv[3];
    if (argc > 4) cfg.profilePath = argv[4];

    std::cout << "========================================\n"
              << "  Face Crop Extractor (Advanced)\n"
              << "  Frames    : " << cfg.framesDir        << "\n"
              << "  Output    : " << cfg.outputDir        << "\n"
              << "  MaxFaces  : " << cfg.maxFacesPerFrame << "\n"
              << "  BlurGate  : " << cfg.blurThreshold    << "\n"
              << "========================================\n";

    // Load cascades
    cv::CascadeClassifier frontalCascade, profileCascade;
    if (!frontalCascade.load(cfg.frontalPath)) {
        std::cerr << "[ERROR] Frontal cascade not found: " << cfg.frontalPath << "\n";
        return -1;
    }
    if (!profileCascade.load(cfg.profilePath)) {
        std::cerr << "[ERROR] Profile cascade not found: " << cfg.profilePath << "\n";
        return -1;
    }
    std::cout << "[OK] Cascades loaded\n";

    // Gather frames
    auto frames = getImageFiles(cfg.framesDir);
    if (frames.empty()) {
        std::cerr << "[ERROR] No images in: " << cfg.framesDir << "\n";
        return -1;
    }
    std::cout << "[OK] Found " << frames.size() << " frames\n\n";
    fs::create_directories(cfg.outputDir);

    // Stats 
    int totalFrames=0, skippedCorrupt=0, skippedBlurry=0;
    int framesWithFaces=0, totalFaces=0;

    // Frame loop 
    for (const auto& framePath : frames) {
        totalFrames++;

        cv::Mat img = cv::imread(framePath.string());
        if (img.empty()) {
            std::cerr << "[WARN] Corrupt: " << framePath.filename() << "\n";
            skippedCorrupt++;
            continue;
        }

        cv::Mat gray;
        cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);

        // Skip blurry frames — not worth detecting in
        double blur = computeBlurScore(gray);
        if (blur < cfg.blurThreshold) {
            std::cout << "[Frame " << std::setw(5) << totalFrames << "] "
                      << framePath.filename().string()
                      << " SKIP blurry=" << std::fixed
                      << std::setprecision(1) << blur << "\n";
            skippedBlurry++;
            continue;
        }

        // Detect
        auto faces = detectFaces(gray, frontalCascade, profileCascade, cfg);

        std::cout << "[Frame " << std::setw(5) << totalFrames << "] "
                  << std::setw(25) << std::left
                  << framePath.filename().string()
                  << std::right
                  << " faces=" << faces.size()
                  << " blur="  << std::fixed << std::setprecision(0) << blur;

        if (faces.empty()) { std::cout << "\n"; continue; }

        // Create frame folder: output/frame_1/
        framesWithFaces++;
        std::string frameDir = cfg.outputDir + "/frame_"
                             + std::to_string(totalFrames);
        fs::create_directories(frameDir);

        // Crop & save
        int saved = 0;
        for (size_t i = 0; i < faces.size(); i++) {
            cv::Rect padded = safePadded(faces[i], cfg.paddingPercent,
                                          img.cols, img.rows);
            if (padded.width <= 0 || padded.height <= 0) continue;

            std::string outPath = frameDir + "/face_"
                                + std::to_string(i + 1) + ".jpg";
            try {
                if (cv::imwrite(outPath, img(padded),
                    {cv::IMWRITE_JPEG_QUALITY, 95}))
                { saved++; totalFaces++; }
            } catch (const cv::Exception& e) {
                std::cerr << "\n[ERROR] " << e.what();
            }
        }
        std::cout << " -> " << saved << " saved to " << frameDir << "\n";
    }

    // Summary 
    std::cout << "\n========================================\n"
              << "  SUMMARY\n"
              << "  Total frames      : " << totalFrames     << "\n"
              << "  Skipped (corrupt) : " << skippedCorrupt  << "\n"
              << "  Skipped (blurry)  : " << skippedBlurry   << "\n"
              << "  Frames with faces : " << framesWithFaces << "\n"
              << "  Total faces saved : " << totalFaces       << "\n"
              << "  Output            : " << cfg.outputDir    << "\n"
              << "========================================\n";
    return 0;
}