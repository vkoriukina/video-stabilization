#pragma once

#include <opencv2/core/core.hpp>
#include "opencv2/highgui/highgui.hpp"
#include <vector>

class Stabilizer
{
 public:
    ~Stabilizer() {}

    bool init( const cv::Mat& frame);
    bool track( const cv::Mat& frame);
    void generateFinalShift();
    void drawPlots();
    void resizeVideo(cv::VideoCapture cap);
    void caclMaxShifts();
    void responce();

    cv::Mat prevFrame;
    std::vector<float> xshift, yshift, xsmoothed, ysmoothed;
    std::vector<cv::Point2f> previousFeatures;
    int maxX, maxY;
    int maxUp, maxLeft,maxRight, maxDown;
};
