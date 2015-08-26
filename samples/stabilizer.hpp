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
    void caclMaxShifts();
    void responce();

    cv::Mat prevFrame;
    std::vector<float> xshift, yshift, xsmoothed, ysmoothed;
    std::vector<cv::Point2f> previousFeatures;
    int maxX, maxY;
    int maxUp, maxLeft,maxRight, maxDown;
};
