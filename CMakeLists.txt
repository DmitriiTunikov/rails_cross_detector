cmake_minimum_required(VERSION 3.15)
project(hough_detector)

include_directories(.)

find_package( OpenCV REQUIRED )

set(CMAKE_CXX_STANDARD 17)

add_executable(hough_detector main.cpp
        hough_detector/hough_algo.cpp hough_detector/hough_algo.h
        utils/files_utils.h utils/files_utils.cpp
        utils/cv_supp.cpp utils/cv_supp.h
        utils/math_utils.cpp utils/math_utils.h)

target_link_libraries( ${PROJECT_NAME} ${OpenCV_LIBS} )