#pragma once

#include <opencv2/core/core.hpp>
#include <vector>

class Stabilizer
{
 public:
    ~Stabilizer() {}

    bool init( const cv::Mat& frame);
    bool track( const cv::Mat& frame);
    void generateFinalShift();
    void drawPlots();
    void resizeVideo(const cv::Mat& frame, int number, cv::Mat& outputFrame);
    void caclMaxShifts();

private:
    cv::Mat prevFrame;
    std::vector<float> xshift, yshift, xsmoothed, ysmoothed;
    std::vector<cv::Point2f> previousFeatures;
    int maxX, maxY;
};
