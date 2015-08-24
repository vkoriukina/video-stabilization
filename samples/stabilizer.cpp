#include "stabilizer.hpp"


bool Stabilizer::init( const cv::Mat& frame)
{
    return true;
}

bool Stabilizer::track( const cv::Mat& frame)
{
    return true;
}

bool generateFinalShift()
{
    cv::Mat image;
    int radius=5;
    for (int i=0; i<xshift.size(); i++)
    {
        image.at<Point2f>(
        cv::circle(image,cvPoint(mypoints[i].x,mypoints[i].y),radius,CV_RGB(100,0,0),-1,8,0);
    }
    return true;
}