#include "stabilizer.hpp"
#include "math.h"
#include"opencv2\highgui\highgui.hpp"


bool Stabilizer::init( const cv::Mat& frame)
{
    return true;
}

bool Stabilizer::track( const cv::Mat& frame)
{
    return true;
}

bool Stabilizer :: generateFinalShift()
{
    double xsum = 0;
    double ysum = 0;
    double avg_x = 0;
    double avg_y = 0;
    int count = 0;
    int radius = 3;
    for(int j=-radius; j <= radius; j++)
    {
        for(int i=0; i < xshift.size(); i++)
        {
            if(i+j >= 0 && i+j < xshift.size())
            {
               xsum += xshift[i];
               ysum += yshift[i];
               count++;
            }
            
            avg_x = xsum / count;
            avg_y = ysum / count;

            xsmoothed.push_back(avg_x);    
            ysmoothed.push_back(avg_y);
        }
    }
    return true;
}


void Stabilizer::resizeVideo(const cv::Mat& frame, int number, cv::Mat& outputFrame){
    cv::Rect rect(xsmoothed[number],ysmoothed[number],frame.size().width - maxX,frame.size().height - maxY);
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