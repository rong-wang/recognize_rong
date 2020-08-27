cmake_minimum_required(VERSION 3.1)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

project(face_recognition)

add_subdirectory(C:/Users/rongw/Documents/Code/libraries/dlib-19.20/dlib dlib_build)

set(OpenCV_DIR "C:/Users/rongw/Documents/Code/libraries/opencv/build")
find_package( OpenCV REQUIRED )
INCLUDE_DIRECTORIES( "C:/Users/rongw/Documents/Code/libraries/opencv/build/include")

add_executable(run face_alignment.cpp)

target_link_libraries(run "C:/Users/rongw/Documents/Code/libraries/opencv/build/x64/vc15/lib/opencv_world420d.lib" )
target_link_libraries(run dlib::dlib)
