#pragma once

#include <opencv2/core/core.hpp>
#include <vector>

class Stabilizer
{
 public:
    ~Stabilizer() {}

    bool init( const cv::Mat& frame);
    bool track( const cv::Mat& frame);
    bool generateFinalShift();

private:
    cv::Mat prevFrame;
    std::vector<float> xshift, yshift;
};