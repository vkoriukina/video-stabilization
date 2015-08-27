#include "stabilizer.hpp"

#include <cmath>
#include <iostream>
#include <fstream>

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/video/video.hpp"

using namespace std;

bool Stabilizer::init( const cv::Mat& frame)
{
    prevFrame = frame.clone();
    cv::Mat previous_frame_gray;
    cv::cvtColor(frame, previous_frame_gray, cv::COLOR_BGR2GRAY);
   
    cv::goodFeaturesToTrack(previous_frame_gray, previousFeatures, 500, 0.1, 5);

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

bool Stabilizer::track(const cv::Mat& frame)
{
    cv::Mat previous_frame_gray;
    cv::cvtColor(prevFrame, previous_frame_gray, cv::COLOR_BGR2GRAY);
  
    cv::goodFeaturesToTrack(previous_frame_gray, previousFeatures, 500, 0.01, 5);

    size_t n = previousFeatures.size();
    CV_Assert(n);
    
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

    prevFrame = frame.clone(); 
    return true;
}


bool Stabilizer::forward_backward_track(const cv::Mat& frame)
{
    cv::Mat previous_frame_gray;
    cv::cvtColor(prevFrame, previous_frame_gray, cv::COLOR_BGR2GRAY);
  
    cv::goodFeaturesToTrack(previous_frame_gray, previousFeatures, 500, 0.01, 5);

    size_t n = previousFeatures.size();
    CV_Assert(n);
    
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
    
    //Compute backward optical flow
    std::vector<cv::Point2f> backwardPoints;
    std::vector<uchar> backState;
    std::vector<float> backError;

    cv::calcOpticalFlowPyrLK(frame, prevFrame, curr_points, backwardPoints, backState, backError);
    float median_back_error = median<float>(backError);

    CV_Assert(s == backwardPoints.size());
    std::vector<float> diff(s);

    for (size_t i = 0; i < s; ++i)
    {
        diff[i] = cv::norm(good_points[i] - backwardPoints[i]);
        // diff[i] = (good_points[i].x - backwardPoints[i].x) * (good_points[i].x - backwardPoints[i].x) + (good_points[i].y - backwardPoints[i].y) * (good_points[i].y - backwardPoints[i].y);
    }

    for (int i = s - 1; i >= 0; --i)
    {
        if (!backState[i] || (backError[i] <= median_back_error) || (diff[i] > 400))
        {
            good_points.erase(good_points.begin() + i);
            curr_points.erase(curr_points.begin() + i);
        }
    }

    s = good_points.size();

    // Find points shift.
    std::vector<float> shifts_x(s);
    std::vector<float> shifts_y(s);
    
    for (size_t i = 0; i < s; ++i)
    {
        shifts_x[i] = curr_points[i].x - good_points[i].x;
        shifts_y[i] = curr_points[i].y - good_points[i].y;
    }
    
    // Find median shift.
    cv::Point2f median_shift(median<float>(shifts_x), median<float>(shifts_y));
    xshift.push_back(median_shift.x);
    yshift.push_back(median_shift.y);

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
        cv::Mat result(1.5*frame.size().height,1.5*frame.size().width,CV_8UC3);
        cap >> frame;
        if(frame.empty())
            break;
        cv::Rect rect(int(maxX + (xsmoothed[number] - xshift[number])),int(maxY + (ysmoothed[number] - yshift[number])),frame.size().width,frame.size().height);
        cv::Rect rectFrame(maxX,maxY,frame.size().width,frame.size().height);
        frame.copyTo(result(rect));
        cv::imshow("Video", frame);
        cv::imshow("VideoNew", result(rectFrame));
        k = cv::waitKey(25);
        if(k == 27)
            break;
        number++;
    }
}


void Stabilizer::saveStabedVideo(const std::string& in_file, const std::string& out_file) const {
    cv::VideoCapture video_reader(in_file);
    CV_Assert(video_reader.isOpened());
    cv::Mat frame;
    video_reader >> frame;
    CV_Assert(!frame.empty());

    int fourcc_code = CV_FOURCC('X', 'V', 'I', 'D');
    cv::VideoWriter video_writer(out_file, fourcc_code, 30.0, frame.size());
    
    int number = 0;
    while (true)
    {
        cv::Mat result(frame.size(), CV_8UC3);
        CV_Assert(number < xsmoothed.size());
        CV_Assert(number < xshift.size());
        CV_Assert(number < ysmoothed.size());
        CV_Assert(number < yshift.size());
        // Shifted frame area.
        cv::Rect roi(cv::Point(xsmoothed[number] - xshift[number], ysmoothed[number] - yshift[number]), frame.size());
        // Region of the resulting image that is to be filled by the original one.
        cv::Rect result_roi = roi & cv::Rect(cv::Point(), frame.size());
        roi.x = -roi.x;
        roi.y = -roi.y;
        // Region of the original frame that is to be copied to the resulting one.
        cv::Rect frame_roi = roi & cv::Rect(cv::Point(), frame.size());
        frame(frame_roi).copyTo(result(result_roi));
        video_writer << result;
        video_reader >> frame;
        if (frame.empty() || number + 1 == xshift.size()) {
            break;
        }
        ++number;
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
    maxX = x; maxY = y;
}


void Stabilizer::responce(){
    ofstream out_Xtrajectories("Xtrajectory.txt");
    ofstream out_Ytrajectories("Ytrajectory.txt");
    for (int i = 0 ; i < xshift.size(); i++) {
        out_Xtrajectories << xshift[i] << " " <<  xsmoothed[i] << endl; 
        out_Ytrajectories << yshift[i] << " " <<  ysmoothed[i] << endl;
    }
}
