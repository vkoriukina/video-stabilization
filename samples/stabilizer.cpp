#include "stabilizer.hpp"
#include <cmath>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/video/video.hpp"
#include <fstream>

#include <iostream>

using namespace std;

bool Stabilizer::init( const cv::Mat& frame)
{
    prevFrame = frame.clone();
    cv::Mat gray4cor;
    cv::cvtColor(frame, gray4cor, cv::COLOR_BGR2GRAY);
   
    cv::goodFeaturesToTrack(gray4cor, previousFeatures, 500, 0.1, 5);

	return true;
}

namespace
{
    template<typename T>
    T median(const std::vector<T>& x)
    {
        CV_Assert(!x.empty());
        std::vector<T> y(x);
        std::sort(y.begin(), y.end());
        return y[y.size() / 2];
    }
}

bool Stabilizer::track( const cv::Mat& frame)
{
    cv::Mat gray4cor;
    cv::cvtColor(prevFrame, gray4cor, cv::COLOR_BGR2GRAY);

    size_t n = previousFeatures.size();
    CV_Assert(n);
   
    bool flag = true;
    //if (n < 500) {
        cv::goodFeaturesToTrack(gray4cor, previousFeatures, 500, 0.01, 5);
        flag = false;
    //}
   
    // Compute optical flow in selected points.
    std::vector<cv::Point2f> currentFeatures;
    std::vector<uchar> state;
    std::vector<float> error;

    cv::calcOpticalFlowPyrLK(prevFrame, frame, previousFeatures, currentFeatures, state, error);

    float median_error = median<float>(error);

    std::vector<cv::Point2f> good_points;
    std::vector<cv::Point2f> curr_points;
    for (size_t i = 0; i < n; ++i)
    {
        if (state[i] && (error[i] <= median_error))
        {
            good_points.push_back(previousFeatures[i]);
            curr_points.push_back(currentFeatures[i]);
        }
    }

    size_t s = good_points.size();
    CV_Assert(s == curr_points.size());
    

    // Find points shift.
    std::vector<float> shifts_x(s);
    std::vector<float> shifts_y(s);

    

    for (size_t i = 0; i < s; ++i)
    {
        shifts_x[i] = curr_points[i].x - good_points[i].x;
        shifts_y[i] = curr_points[i].y - good_points[i].y;
    }
    
    std::sort(shifts_x.begin(), shifts_x.end());
    std::sort(shifts_y.begin(), shifts_y.end());

    // Find median shift.
    cv::Point2f median_shift(shifts_x[s / 2], shifts_y[s / 2]);
    xshift.push_back(median_shift.x);
    yshift.push_back(median_shift.y);

    if (flag) {
        //previousFeatures = currentFeatures;
    }

    prevFrame = frame.clone();
    
    return true;
}

void Stabilizer :: generateFinalShift()
{
    int radius = 30;

    for(int i = 1; i < xshift.size(); i ++)
        xshift[i] += xshift[i - 1];
    for(int i = 1; i < yshift.size(); i ++)
        yshift[i] += yshift[i - 1];

    
    for(int i=0; i < radius; i++)
    {
        xsmoothed.push_back(xshift[i]);
        ysmoothed.push_back(yshift[i]);
    }

    for(int i=radius; i < xshift.size() - radius; i++)
    {
        double xsum = 0;
        double ysum = 0;

        for(int j=-radius; j <= radius; j++)
        {
            xsum += xshift[i + j];
            ysum += yshift[i + j];
        }
        xsmoothed.push_back(xsum / (2 * radius + 1));
        ysmoothed.push_back(ysum / (2 * radius + 1));    
    }

    for(int i = xshift.size() - radius; i < xshift.size(); i++)
    {
        xsmoothed.push_back(xshift[i]);
        ysmoothed.push_back(yshift[i]);
    }

}


void Stabilizer::resizeVideo(cv::VideoCapture cap){
    cv::Mat frame;
    cap >> frame;
    int k, number = 0;
    while (true)
    {
        cv::Mat result(frame.size().height+200,frame.size().width + 200,CV_8UC3);
        cap >> frame;
        if(frame.empty())
            break;
        cv::Rect rect(int(maxX + (xsmoothed[number] - xshift[number])),int(maxY + (ysmoothed[number] - yshift[number])),frame.size().width,frame.size().height);
        cv::Rect rectFrame(maxX,maxY,frame.size().width,frame.size().height);
        frame.copyTo(result(rect));
        cv::imshow("Video", frame);
        cv::imshow("VideoNew", result(rectFrame));
        k = cv::waitKey(1);
        if(k == 27)
            break;
        number++;
    }
}


void Stabilizer::caclMaxShifts(){
    generateFinalShift();
    float x = 0,y = 0;
    for (int i = 0 ; i < xshift.size(); i++){
        if (abs(xsmoothed[i] - xshift[i]) > x){
            x = abs(xsmoothed[i] - xshift[i]);
        }
        if (abs(ysmoothed[i] - yshift[i]) > y){
            y = abs(ysmoothed[i] - yshift[i]);
        }
    }
    maxX = x + 30; maxY = y + 30;
}


void Stabilizer::responce(){
    ofstream out_Xtrajectories("Xtrajectory.txt");
    ofstream out_Ytrajectories("Ytrajectory.txt");
    for (int i = 0 ; i < xshift.size(); i++) {
        out_Xtrajectories << xshift[i] << " " <<  xsmoothed[i] << endl; 
        out_Ytrajectories << yshift[i] << " " <<  ysmoothed[i] << endl;
    }
}
