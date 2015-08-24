#include "stabilizer.hpp"
#include "opencv2\highgui\highgui.hpp"
#include <iostream>


int main( int argc, char** argv )
{
    cv::VideoCapture cap;
    cap.open( argv[1] );
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

    }
    return 0;
}