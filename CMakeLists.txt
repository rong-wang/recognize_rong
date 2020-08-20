cmake_minimum_required(VERSION 2.8.12)

project(face_recognition)

add_subdirectory(C:/Users/rongw/Documents/Code/libraries/dlib-19.20/dlib dlib_build)

set(OpenCV_DIR "C:/Users/rongw/Documents/Code/libraries/opencv/build")
find_package( OpenCV REQUIRED )
INCLUDE_DIRECTORIES( "C:/Users/rongw/Documents/Code/libraries/opencv/build/include")

add_executable(run generate_images.cpp)

target_link_libraries(run "C:/Users/rongw/Documents/Code/libraries/opencv/build/x64/vc15/lib/opencv_world420d.lib" )