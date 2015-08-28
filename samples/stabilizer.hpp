#pragma once

#include <vector>

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"

class Stabilizer
{
 public:
    ~Stabilizer() {}

    bool init( const cv::Mat& frame);
    bool track( const cv::Mat& frame);
    bool forward_backward_track(const cv::Mat& frame);
    void generateFinalShift();
    void resizeVideo(cv::VideoCapture cap);
    void saveStabedVideo(const std::string& in_file,
        const std::string& out_file = "video_stab.avi") const;
    void onlineProsessing(cv::VideoCapture cap);
    void fastOfflineProsessing(cv::VideoCapture cap);
    void smooth(int pos);
    void onlineSmooth(int num, float alfa);
    cv::Mat smoothedImage(cv::Mat frame, float dx, float dy);
    void calcMaxShifts();
    void responce();

    cv::Mat prevFrame;
    std::vector<float> xshift, yshift, xsmoothed, ysmoothed;
    std::vector<cv::Point2f> previousFeatures;
    int maxX, maxY;
    int maxUp, maxLeft,maxRight, maxDown;
    int Radius;
    bool flagUpdateFeatures;
    cv::VideoWriter writeOutputVideo;
};
