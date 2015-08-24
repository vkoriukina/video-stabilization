#include "stabilizer.hpp"
#include "opencv2\highgui\highgui.hpp"
#include <iostream>
#include <string>

const char* params =
     "{ h | video    |       | video file to stabilize                       }";


int main( int argc, char** argv )
{
    cv::CommandLineParser parser(argc, argv, params);
    std::string video_file = parser.get<std::string>("video");


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

    cap >> frame;
    while (!frame.empty())
    {

        stab.track(frame);

        cap >> frame;

        cv::imshow("Video", frame);
        cv::waitKey(1);
    }


    return 0;
}