#include <opencv2/imgproc.hpp>  // Gaussian Blur
#include <opencv2/core.hpp>        // Basic OpenCV structures (cv::Mat, Scalar)
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>  // OpenCV window I/O
#include <opencv2/features2d.hpp>
#include <opencv2/objdetect.hpp>

#include <stdio.h>
#include <iostream>
#include <algorithm>

/**
 * Much of this code is from the opencv facedetect_ex.cpp sample code!
 **/ 

using namespace std;
using namespace cv;

const string windowName("Image Collector");
const string imagePath("C:/Users/rongw/Documents/Code/VSCode_C++/recognize_rong/faces/training/rong/");

const int x_offset = 50;
const int y_offset = 100;

const int numImages = 6000;

class CascadeClassifierAdapter : public DetectionBasedTracker::IDetector 
{
    public:
        CascadeClassifierAdapter(cv::Ptr<cv::CascadeClassifier> detector) : IDetector(), detector(detector) {
            CV_Assert(detector);
        }

        void detect(const cv::Mat &Image, std::vector<cv::Rect> &objects){
            detector->detectMultiScale(Image, objects, scaleFactor, minNeighbours, 0, minObjSize, maxObjSize);
        }

        virtual ~CascadeClassifierAdapter() {}

    private:
        CascadeClassifierAdapter();
        cv::Ptr<cv::CascadeClassifier> detector;
};


int main(int , char**)
{
    namedWindow(windowName);

    VideoCapture stream(0);

    if(!stream.isOpened()){
        cout << "Camera not opened" << endl;
        return 1;
    }
    
    std::string cascadeFileName = samples::findFile("data/lbpcascades/lbpcascade_frontalface_improved.xml");
    cv::Ptr<cv::CascadeClassifier> cclassifier = cv::makePtr<cv::CascadeClassifier>(cascadeFileName);
    if(cclassifier->empty()){
        cout << "Cannot load cascade xml file" << endl;
        return 1;
    }

    cv::Ptr<DetectionBasedTracker::IDetector> detector = cv::makePtr<CascadeClassifierAdapter>(cclassifier);

    cclassifier = cv::makePtr<cv::CascadeClassifier>(cascadeFileName);
    if(cclassifier->empty()){
        cout << "Cannot load cascade xml file" << endl;
        return 1;
    }
    cv::Ptr<DetectionBasedTracker::IDetector> tracking = cv::makePtr<CascadeClassifierAdapter>(cclassifier);

    DetectionBasedTracker::Parameters params;
    DetectionBasedTracker tracker(detector, tracking, params);

    if(!tracker.run()){
        cout << "DetectionBasedTracker init error" << endl;
        return 1;
    }

    Mat referenceFrame;
    Mat grayFrame;
    vector<Rect> faces;

    int curimage = 0;

    do{
        stream >> referenceFrame;
        cvtColor(referenceFrame, grayFrame, COLOR_BGR2GRAY);
        tracker.process(grayFrame);
        tracker.getObjects(faces);

        Mat copy;

        for(int i = 0; i < faces.size(); ++i){
            try{
                Rect r;
                r.x = faces[i].x - x_offset;
                r.y = faces[i].y - y_offset;
                r.width = faces[i].width + (2 * x_offset);
                r.height = faces[i].height + (2 * y_offset);
                copy = referenceFrame(r);
                
                imwrite(imagePath + "rong_" + to_string(400 + curimage++) + ".jpg", copy);
            }
            catch(...){ }
        }
        imshow(windowName, referenceFrame);
        cout << curimage << endl;
    } while(waitKey(100) < 0 && curimage < numImages);

    tracker.stop();

    return 0;
}