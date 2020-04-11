#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "utils/files_utils.h"
#include "hough_detector/hough_algo.h"
#include <chrono>

using namespace cv;
using namespace std;

void run_test(int argc, char** argv) {
    if (argc < 3) {
        std::cout << "Wrong arguments count, program accept 2 arguments: " <<
                  R"(path_to_images_dir path_to_results_dir)" << std::endl;

        throw std::runtime_error(reinterpret_cast<const char *>("wrong input args"));
    }

    int canny_treashhold1 = 150, canny_treashhold2 = 250;

    string images_dir = argv[1], results_dir = argv[2];
    files_utils::remove_dir(results_dir);
    files_utils::create_dir(results_dir);

    vector<files_utils::FileName> image_file_names = files_utils::get_files_from_dir(images_dir);
    for (files_utils::FileName& image_file_name : image_file_names) {
        cv::Mat image = imread(image_file_name.path, IMREAD_COLOR);

        //get file name without file extention
        stringstream ss(image_file_name.name);
        string file_name;
        std::getline(ss, file_name, '.');

        HoughDetector detector(image, canny_treashhold1, canny_treashhold2, 30, 7);
        detector.get_cross_result();
        detector.save_results(results_dir + "/" + file_name + ".txt");
    }
}

int main(int argc, char** argv) {
    if (argc < 2)
    {
        std::cout << "Wrong arguments count, program accept one argument: " <<
                  R"(path to input file, for example: C:\Users\dimat\Downloads\cross_detect_data_set_src\img0002.jpg)" << std::endl;

        return -1;
    }
    if (argc > 2)
    {
        run_test(argc, argv);
        return 0;
    }

    std::string file_name = std::string(argv[1]);

    Mat image, gray_img, canny_res;
    image = imread(file_name, IMREAD_COLOR);
    if (!image.data)
    {
        cout << "Could not open or find input file" << endl;
        return -1;
    }

    cv::cvtColor(image, gray_img, COLOR_BGR2GRAY);
    int canny_treashhold1 = 150, canny_treashhold2 = 250;
    cv::Canny(image, canny_res, canny_treashhold1, canny_treashhold2);
    std::chrono::milliseconds start_time = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch());

    HoughDetector detector(image, canny_treashhold1, canny_treashhold2, 30, 7);
    detector.get_cross_result();
    detector.draw_cross_res();

    std::chrono::milliseconds diff = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()) - start_time;
    std::cout << "time: " << diff.count() << "ms" << std::endl;

    imshow("original", image);
    imshow("canny", canny_res);

    waitKey(0);

    return 0;
}