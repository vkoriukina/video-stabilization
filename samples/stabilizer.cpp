#include "stabilizer.hpp"

#include <cmath>

#include "opencv2/highgui/highgui.hpp"


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


void Stabilizer::resizeVideo(const cv::Mat& frame, int number, cv::Mat& outputFrame){
    cv::Rect rect(maxX - xsmoothed[number],maxY - ysmoothed[number],frame.size().width - 2*maxX,frame.size().height - 2*maxY);
    outputFrame = frame(rect);
}


void Stabilizer::caclMaxShifts(){
    int x = 0,y = 0;
    for (int i = 0 ; i < xsmoothed.size(); i++){
        if (abs(xsmoothed[i] - xshift[i]) > x){
            x = abs(xsmoothed[i] - xshift[i]);
        }
        if (abs(ysmoothed[i] - yshift[i]) > y){
            y = abs(ysmoothed[i] - yshift[i]);
        }
    }
    maxX = x; maxY = y;
}
