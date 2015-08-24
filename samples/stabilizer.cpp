#include "stabilizer.hpp"
#include "cv.h"


bool Stabilizer::init( const cv::Mat& frame)
{
    prevFrame = frame.clone();
	return true;
}

bool Stabilizer::track( const cv::Mat& frame)
{
    return true;
}

bool generateFinalShift()
{
    return true;
}