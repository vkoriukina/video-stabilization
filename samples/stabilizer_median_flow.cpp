#include "stabilizer.hpp"
#include "opencv2\highgui\highgui.hpp"
#include <iostream>
#include <string>

const char* params =
     "{   | video    |       | video file to stabilize                       }";


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


    char k = 0;
    cap >> frame;
    while (true)
    {

        stab.track(frame);

        cap >> frame;
        if(frame.empty())
            break;

        cv::imshow("Video", frame);
        k = cv::waitKey(1);

        if(k == 27)
            break;
    }
    stab.generateFinalShift();
    stab.drawPlots();

    return 0;
}