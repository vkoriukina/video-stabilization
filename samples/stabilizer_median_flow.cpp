#include "stabilizer.hpp"
#include "opencv2\highgui\highgui.hpp"
#include <iostream>
#include <string>
#include <time.h>

const char* params =
     "{ | help       | false | print help               }"
     "{ | video      |       | video file to stabilize  }"
     "{ | type       |       | type of stabilization    }"
     "{ | medianflow | false | whether to use median flow instead of just optical flow }";


int main( int argc, char** argv )
{
    cv::CommandLineParser parser(argc, argv, params);
    if (parser.get<bool>("help")) {
        parser.printParams();
        return 0;
    }
    std::string video_file = parser.get<std::string>("video");
    std::string type = parser.get<std::string>("type");

    Stabilizer stab;

    if (type == "offline")
    {
        cv::VideoCapture cap;
        cap.open( video_file );
        if( !cap.isOpened() )
        {
            std::cout << "Error: could not initialize video capturing...\n";
            return 1;
        }

        cv::Mat frame;   
        cap >> frame;
        
        stab.init(frame);

        cap >> frame;
        int k;
        int time = clock();

        while (true)
        {
            if (parser.get<bool>("medianflow")) {
                stab.forward_backward_track(frame);
            } else {
                stab.track(frame);
            }

            cap >> frame;
            if(frame.empty())
                break;

            cv::imshow("Video", frame);
            k = cv::waitKey(1);
            if (k == 27){
                break;
            }

            std::cout << clock() - time << "\n";
            time = clock();

        }

        stab.calcMaxShifts();
        std::cout << clock() - time << "\n";
    }

    if (type == "online" )
    {
        cv::VideoCapture cap2(video_file);
        cap2.open( video_file );
        stab.onlineProsessing(cap2);
    }

    if (type == "fast" )
    {
        cv::VideoCapture cap2(video_file);
        cap2.open( video_file );
        stab.fastOfflineProsessing(cap2);
    }

    stab.saveStabedVideo(video_file, "stab.avi");

    return 0;
}