#include <opencv2/opencv.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <filesystem>
#include <iomanip>
#include <sstream>

namespace fs = std::filesystem;

struct Config {
    std::string framesDir   = "frames";
    std::string outputDir   = "output";
    std::string frontalPath = "data/haarcascade_frontalface_default.xml";
    std::string profilePath = "data/haarcascade_profileface.xml";

    // Tighter detection — reduces false positives
    double scaleFactor    = 1.1;
    int    minNeighbors   = 6;    // raised from 4 → fewer false positives
    int    minFaceSize    = 40;   // raised from 20 → ignore tiny false detections
    int    maxFaceSize    = 500;
    int    maxFacesPerFrame = 3;  // hard cap matching your video

    // Quality filters
    double blurThreshold  = 80.0;
    double nmsIOUThreshold = 0.25; // tighter NMS

    int paddingPercent    = 10;

    // Only try 3 angles instead of 5 — reduces false rotation detections
    std::vector<int> rotationAngles = {0, -20, 20};
};

std::vector<fs::path> getJpegFiles(const std::string& dir) {
    std::vector<fs::path> files;
    if (!fs::exists(dir)) return files;
    for (const auto& entry : fs::directory_iterator(dir)) {
        std::string ext = entry.path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        if (ext == ".jpg" || ext == ".jpeg")
            files.push_back(entry.path());
    }
    std::sort(files.begin(), files.end());
    return files;
}

double computeBlurScore(const cv::Mat& gray) {
    cv::Mat lap;
    cv::Laplacian(gray, lap, CV_64F);
    cv::Scalar mean, stddev;
    cv::meanStdDev(lap, mean, stddev);
    return stddev[0] * stddev[0];
}

cv::Mat applyCLAHE(const cv::Mat& gray) {
    auto clahe = cv::createCLAHE(2.0, cv::Size(8, 8));
    cv::Mat enhanced;
    clahe->apply(gray, enhanced);
    return enhanced;
}

double computeIoU(const cv::Rect& a, const cv::Rect& b) {
    cv::Rect intersection = a & b;
    if (intersection.area() == 0) return 0.0;
    double unionArea = a.area() + b.area() - intersection.area();
    return (double)intersection.area() / unionArea;
}

std::vector<cv::Rect> applyNMS(std::vector<cv::Rect> rects, double iouThresh) {
    std::sort(rects.begin(), rects.end(), [](const cv::Rect& a, const cv::Rect& b) {
        return a.area() > b.area();
    });
    std::vector<bool> suppressed(rects.size(), false);
    std::vector<cv::Rect> result;
    for (size_t i = 0; i < rects.size(); i++) {
        if (suppressed[i]) continue;
        result.push_back(rects[i]);
        for (size_t j = i + 1; j < rects.size(); j++) {
            if (!suppressed[j] && computeIoU(rects[i], rects[j]) > iouThresh)
                suppressed[j] = true;
        }
    }
    return result;
}

cv::Mat rotateImage(const cv::Mat& src, double angle) {
    if (angle == 0) return src;
    cv::Point2f center(src.cols / 2.0f, src.rows / 2.0f);
    cv::Mat rot = cv::getRotationMatrix2D(center, angle, 1.0);
    cv::Mat dst;
    cv::warpAffine(src, dst, rot, src.size(), cv::INTER_LINEAR,
                   cv::BORDER_CONSTANT, cv::Scalar(128));
    return dst;
}

cv::Rect rotateRect(const cv::Rect& r, double angle, cv::Size imgSize) {
    if (angle == 0) return r;
    cv::Point2f center(imgSize.width / 2.0f, imgSize.height / 2.0f);
    cv::Mat rot = cv::getRotationMatrix2D(center, -angle, 1.0);
    std::vector<cv::Point2f> corners = {
        {(float)r.x,            (float)r.y},
        {(float)(r.x+r.width),  (float)r.y},
        {(float)r.x,            (float)(r.y+r.height)},
        {(float)(r.x+r.width),  (float)(r.y+r.height)}
    };
    std::vector<cv::Point2f> transformed;
    cv::transform(corners, transformed, rot);
    float minX = std::min({transformed[0].x,transformed[1].x,transformed[2].x,transformed[3].x});
    float minY = std::min({transformed[0].y,transformed[1].y,transformed[2].y,transformed[3].y});
    float maxX = std::max({transformed[0].x,transformed[1].x,transformed[2].x,transformed[3].x});
    float maxY = std::max({transformed[0].y,transformed[1].y,transformed[2].y,transformed[3].y});
    return cv::Rect((int)minX,(int)minY,(int)(maxX-minX),(int)(maxY-minY));
}

cv::Rect safePaddedRect(const cv::Rect& face, int padPct, int imgW, int imgH) {
    int px = (int)(face.width  * padPct / 100.0);
    int py = (int)(face.height * padPct / 100.0);
    int x  = std::max(0, face.x - px);
    int y  = std::max(0, face.y - py);
    int w  = std::min(imgW - x, face.width  + 2 * px);
    int h  = std::min(imgH - y, face.height + 2 * py);
    return cv::Rect(x, y, w, h);
}

// Confidence score = area * neighbors proxy (larger face = more confident)
std::vector<cv::Rect> keepTopN(std::vector<cv::Rect> rects, int n) {
    if ((int)rects.size() <= n) return rects;
    // Sort by area descending — larger detections = more confident
    std::sort(rects.begin(), rects.end(), [](const cv::Rect& a, const cv::Rect& b) {
        return a.area() > b.area();
    });
    rects.resize(n);
    return rects;
}

std::vector<cv::Rect> detectAllFaces(
    const cv::Mat& gray,
    cv::CascadeClassifier& frontal,
    cv::CascadeClassifier& profile,
    const Config& cfg)
{
    std::vector<cv::Rect> allFaces;

    for (int angle : cfg.rotationAngles) {
        cv::Mat rotGray  = rotateImage(gray, angle);
        cv::Mat enhanced = applyCLAHE(rotGray);

        // Frontal
        std::vector<cv::Rect> frontalFaces;
        frontal.detectMultiScale(enhanced, frontalFaces,
            cfg.scaleFactor, cfg.minNeighbors, 0,
            cv::Size(cfg.minFaceSize, cfg.minFaceSize),
            cv::Size(cfg.maxFaceSize, cfg.maxFaceSize));

        // Left profile
        std::vector<cv::Rect> leftProfile;
        profile.detectMultiScale(enhanced, leftProfile,
            cfg.scaleFactor, cfg.minNeighbors, 0,
            cv::Size(cfg.minFaceSize, cfg.minFaceSize),
            cv::Size(cfg.maxFaceSize, cfg.maxFaceSize));

        // Right profile (flip)
        cv::Mat flipped;
        cv::flip(enhanced, flipped, 1);
        std::vector<cv::Rect> rightProfile;
        profile.detectMultiScale(flipped, rightProfile,
            cfg.scaleFactor, cfg.minNeighbors, 0,
            cv::Size(cfg.minFaceSize, cfg.minFaceSize),
            cv::Size(cfg.maxFaceSize, cfg.maxFaceSize));

        for (auto& r : rightProfile)
            r.x = gray.cols - r.x - r.width;

        for (auto& r : frontalFaces) allFaces.push_back(rotateRect(r, angle, gray.size()));
        for (auto& r : leftProfile)  allFaces.push_back(rotateRect(r, angle, gray.size()));
        for (auto& r : rightProfile) allFaces.push_back(rotateRect(r, angle, gray.size()));
    }

    // NMS to remove overlaps
    allFaces = applyNMS(allFaces, cfg.nmsIOUThreshold);

    // Remove out-of-bounds
    std::vector<cv::Rect> valid;
    cv::Rect imgBounds(0, 0, gray.cols, gray.rows);
    for (const auto& r : allFaces) {
        cv::Rect clamped = r & imgBounds;
        if (clamped.width  > cfg.minFaceSize &&
            clamped.height > cfg.minFaceSize)
            valid.push_back(clamped);
    }

    // Hard cap: keep only top N by face area
    valid = keepTopN(valid, cfg.maxFacesPerFrame);

    return valid;
}

int main(int argc, char* argv[]) {
    Config cfg;
    if (argc > 1) cfg.framesDir   = argv[1];
    if (argc > 2) cfg.outputDir   = argv[2];
    if (argc > 3) cfg.frontalPath = argv[3];
    if (argc > 4) cfg.profilePath = argv[4];

    std::cout << "========================================\n";
    std::cout << "   Production Face Detection App\n";
    std::cout << "   Max faces per frame: " << cfg.maxFacesPerFrame << "\n";
    std::cout << "========================================\n";

    cv::CascadeClassifier frontalCascade, profileCascade;
    if (!frontalCascade.load(cfg.frontalPath)) {
        std::cerr << "[ERROR] Frontal cascade not found\n"; return -1;
    }
    if (!profileCascade.load(cfg.profilePath)) {
        std::cerr << "[ERROR] Profile cascade not found\n"; return -1;
    }
    std::cout << "[OK] Cascades loaded\n";

    auto frameFiles = getJpegFiles(cfg.framesDir);
    if (frameFiles.empty()) {
        std::cerr << "[ERROR] No frames in: " << cfg.framesDir << "\n"; return -1;
    }
    std::cout << "[OK] Found " << frameFiles.size() << " frames\n";
    fs::create_directories(cfg.outputDir);

    int totalFrames=0, skippedBlurry=0, skippedCorrupt=0;
    int framesWithFaces=0, totalFaces=0;

    for (const auto& framePath : frameFiles) {
        totalFrames++;

        cv::Mat img = cv::imread(framePath.string());
        if (img.empty()) {
            std::cerr << "[WARN] Corrupt frame: " << framePath.filename() << "\n";
            skippedCorrupt++;
            continue;
        }

        cv::Mat gray;
        cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);

        double blurScore = computeBlurScore(gray);
        if (blurScore < cfg.blurThreshold) {
            std::cout << "[Frame " << std::setw(5) << totalFrames << "] "
                      << framePath.filename().string()
                      << " -> SKIPPED (blurry: " << std::fixed
                      << std::setprecision(1) << blurScore << ")\n";
            skippedBlurry++;
            continue;
        }

        std::vector<cv::Rect> faces = detectAllFaces(gray, frontalCascade,
                                                      profileCascade, cfg);

        std::cout << "[Frame " << std::setw(5) << totalFrames << "] "
                  << framePath.filename().string()
                  << " -> " << faces.size() << " face(s)"
                  << " [blur:" << std::fixed << std::setprecision(0)
                  << blurScore << "]";

        if (faces.empty()) { std::cout << "\n"; continue; }

        framesWithFaces++;
        std::string frameFolder = cfg.outputDir + "/frame_" +
                                  std::to_string(totalFrames);
        fs::create_directories(frameFolder);

        int faceNum = 1;
        for (const auto& faceRect : faces) {
            cv::Rect padded = safePaddedRect(faceRect, cfg.paddingPercent,
                                             img.cols, img.rows);
            if (padded.width <= 0 || padded.height <= 0) continue;

            cv::Mat faceROI = img(padded);
            std::string outPath = frameFolder + "/face_" +
                                  std::to_string(faceNum) + ".jpg";
            try {
                if (cv::imwrite(outPath, faceROI, {cv::IMWRITE_JPEG_QUALITY, 95}))
                { faceNum++; totalFaces++; }
                else
                    std::cerr << "[WARN] Failed to write: " << outPath << "\n";
            } catch (const cv::Exception& e) {
                std::cerr << "[ERROR] imwrite: " << e.what() << "\n";
            }
        }
        std::cout << " -> Saved " << (faceNum-1)
                  << " crops to " << frameFolder << "\n";
    }

    std::cout << "\n========================================\n";
    std::cout << "           PROCESSING COMPLETE\n";
    std::cout << "========================================\n";
    std::cout << "Total frames        : " << totalFrames    << "\n";
    std::cout << "Skipped (corrupt)   : " << skippedCorrupt << "\n";
    std::cout << "Skipped (blurry)    : " << skippedBlurry  << "\n";
    std::cout << "Frames with faces   : " << framesWithFaces << "\n";
    std::cout << "Total faces saved   : " << totalFaces      << "\n";
    std::cout << "Output directory    : " << cfg.outputDir   << "\n";
    std::cout << "========================================\n";
    return 0;
}
