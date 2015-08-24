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

void Stabilizer :: generateFinalShift()
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
}

void Stabilizer :: drawPlots()
{
    int plotWidth = 200; 
    int plotHeight = 200; 

    std::vector<cv::Point> plot; 

    //for(int i=0; i < xshift.size(); i++)
    //{
    //    plot.push_back(cv::Point(i,xsmoothed[i]));
    //}
    cv::Mat img;

    for(unsigned int i=1; i<xshift.size(); ++i)
    {

        cv::Point2f p1; p1.x = i-1; p1.y = plot[i-1].x;
        cv::Point2f p2; p2.x = i;   p2.y = plot[i].x;
        cv::line(img, p1, p2, 'r', 5, 8, 0);
    }

    //the image to be plotted 
   // cv::Mat img = cv::Mat::zeros(plotHeight,plotWidth, CV_8UC3); 

    cv::namedWindow("Plot", CV_WINDOW_AUTOSIZE); 
    cv::imshow("Plot", img); //display the image which is stored in the 'img' in the "MyWindow" window
    cv::waitKey(0);

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