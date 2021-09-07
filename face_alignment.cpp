// -10 points, where's your header? 
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/image_io.h>
#include <dlib/gui_widgets.h>
#include <dlib/opencv.h>
#include <opencv2/core/core.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

#include <iostream>
#include <filesystem>
#include <string>

// -100: Poor style! never use namespace. 
using namespace dlib;

namespace fs = std::filesystem;

// -10000: these paths are hardcodes smh. 
const std::string pathToFaces("C:/Users/rongw/Documents/Code/VSCode_C++/recognize_rong/faces/test/rong");
const std::string faceLandmarkDat("C:/Users/rongw/Documents/Code/VSCode_C++/recognize_rong/shape_predictor_68_face_landmarks.dat");
const std::string destination("C:/Users/rongw/Documents/Code/VSCode_C++/recognize_rong/faces/aligned/rong/");

void alignFaces(std::string path, std::string dest, frontal_face_detector &detector, shape_predictor &sp);

// -10 points for no function header! 
int main(int argc, char **argv)
{
    frontal_face_detector detector = get_frontal_face_detector();

    shape_predictor sp;
    deserialize(faceLandmarkDat) >> sp;

    alignFaces(pathToFaces, destination, detector, sp);
    alignFaces("C:/Users/rongw/Documents/Code/VSCode_C++/recognize_rong/faces/training/notrong", 
                "C:/Users/rongw/Documents/Code/VSCode_C++/recognize_rong/faces/aligned/notrong/", detector, sp);

    return 0;
}

void alignFaces(std::string path, std::string dest, frontal_face_detector &detector, shape_predictor &sp)
{
    image_window win, win_face;
    int count = 0;

    // loop through files in a directory
    for (const auto &file : fs::directory_iterator(path))
    {
        array2d<rgb_pixel> img;
        load_image(img, file.path().string());
        pyramid_up(img);

        win.clear_overlay();
        win.set_image(img);

        std::vector<rectangle> faces = detector(img);
        std::vector<full_object_detection> shapes;

        for (int i = 0; i < faces.size(); ++i)
        {
            full_object_detection shape = sp(img, faces[i]);
            std::cout << file.path().string() << " complete" << std::endl;

            shapes.push_back(shape);
            array<array2d<rgb_pixel>> face_chips;
            extract_image_chips(img, get_face_chip_details(shapes), face_chips);
            win_face.set_image(tile_images(face_chips));

            matrix<rgb_pixel> alignedImg = tile_images(face_chips);
            win_face.set_image(alignedImg);

            // convert aligned image to Mat for easier saving
            cv::Mat cvImg = toMat(alignedImg);
            cv::imwrite(dest + "face_" + std::to_string(count++) + ".jpg", cvImg);
        }

        if(count > 10) break;
    }
}
