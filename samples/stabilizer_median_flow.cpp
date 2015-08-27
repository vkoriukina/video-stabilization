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
    cv::Mat newframe; 
    cap >> frame;
    cap >> newframe;

    Stabilizer stab1, stab2;
    stab1.init(frame);
    stab2.init(newframe);


    char k = 0;
    cap >> frame;
    while (true)
    {

        stab1.track(frame);
        stab2.forward_backward_track(frame);
        cv::imshow("Video", frame);
        k = cv::waitKey(1);

        cap >> frame;
        if (frame.empty() || k == 27) {
            break;
        }
    }

    stab1.caclMaxShifts();
    stab1.saveStabedVideo(video_file, "forward_stab.avi");
    stab1.responce();

    stab2.caclMaxShifts();
    stab2.saveStabedVideo(video_file, "forward_backward_stab.avi");
    stab2.responce();

    {
        cv::VideoCapture cap1("forward_stab.avi");
        CV_Assert(cap1.isOpened());
        cv::Mat frame1;
        cap1 >> frame1;

        cv::VideoCapture cap2("forward_backward_stab.avi");
        CV_Assert(cap2.isOpened());
        cv::Mat frame2;
        cap2 >> frame2;

        while (!frame1.empty() && !frame2.empty()) {
            imshow("forward_stab", frame1);
            imshow("forward_backward_stab", frame2);
            cv::waitKey(25);
            cap1 >> frame1;
            cap2 >> frame2;
        }
    }

    return 0;
}