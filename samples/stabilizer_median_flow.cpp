#include "stabilizer.hpp"
#include "opencv2\highgui\highgui.hpp"
#include <iostream>
#include <string>
#include <time.h>

const char* params =
     "{   | video |       | video file to stabilize  }"
     "{   | type  |       | type of stabilization    }";


int main( int argc, char** argv )
{
    cv::CommandLineParser parser(argc, argv, params);
    std::string video_file = parser.get<std::string>("video");
    std::string type = parser.get<std::string>("type");


    cv::VideoCapture cap;
    cap.open( video_file );
    if( !cap.isOpened() )
    {
        std::cout << "Error: could not initialize video capturing...\n";
        return 1;
    }

    cv::Mat frame;   
    cap >> frame;

    Stabilizer stab;
    stab.init(frame);

    if (type == "offline")
    {
        cap >> frame;
        int k;
        int time = clock();

        while (true)
        {

            stab.track(frame);

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

        cv::VideoCapture cap2;
        cap2.open( video_file );
        stab.resizeVideo(cap2);
    }

    if (type == "online" )
    {
        cv::VideoCapture cap2;
        cap2.open( video_file );
        stab.onlineProsessing(cap2);
    }

    if (type == "fast" )
    {
        cv::VideoCapture cap2;
        cap2.open( video_file );
        stab.fastOfflineProsessing(cap2);
    }

    return 0;
}