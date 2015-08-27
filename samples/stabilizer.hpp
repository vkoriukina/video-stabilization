#pragma once

#include <opencv2/core/core.hpp>
#include "opencv2/highgui/highgui.hpp"
#include <vector>

class Stabilizer
{
 public:
    ~Stabilizer() {}

    bool init( const cv::Mat& frame, std::string type);
    bool track( const cv::Mat& frame);
    void generateFinalShift();
    void resizeVideo(cv::VideoCapture cap);
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