//
// Created by dmitrii on 01.04.2020.
//

#ifndef CROSSDETECTOR_HOUGH_ALGO_H
#define CROSSDETECTOR_HOUGH_ALGO_H

#include <vector>
#include <opencv2/core/mat.hpp>
#include <utils/cv_supp.h>

class HoughDetector{
public:
    HoughDetector(cv::Mat& image, int canny_treshhold1 = 150, int canny_treshhold2 = 250, int max_strip_len = 30, int min_strip_len = 10);

    std::vector<cv::Point2i> get_cross_result();
    void draw_cross_res();
private:
    const int m_x_didd = 10;
    constexpr static const float m_parallel_cos_diff = 0.3;

    struct Cell {
        cv_supp::Line line;
        std::vector<std::shared_ptr<Cell>> neighs;
        int y_min;
        int y_max;
        bool has_parent;
        explicit Cell(const cv_supp::Line& line_, const int y_min_, const int y_max_) : line(line_), y_min(y_min_), y_max(y_max_), has_parent(false) {};

        bool has_same_direction_neigh();
        std::shared_ptr<Cell> get_same_direction_neigh();
        static bool is_different_direction_lines(std::shared_ptr<Cell> c1, std::shared_ptr<Cell> c2);
    };

    using CellPtr = std::shared_ptr<Cell>;
    std::vector<std::vector<CellPtr>> m_grid;
    cv::Mat m_image;
    int m_canny_treshhold1;
    int m_canny_treshhold2;
    int m_max_strip_len;
    int m_min_strip_len;
    std::vector<cv::Point> m_cross_res;

    void solve();

    std::vector<cv_supp::Line> get_lines_on_cropped(int y_min, int y_max);
};

namespace hough_algo {
    void find_lines(cv::Mat& image, int crop_count, int canny_treashhold1, int canny_treashhold2);

    std::vector<cv::Mat> get_cropped_images(const cv::Mat& image, int crop_count);

    void find_lines_on_cropped(std::vector<cv::Mat>& crop_images, int canny_treashhold1, int canny_treashhold2);

    void draw_lines(const std::vector<cv::Vec2f>& lines, cv::Mat& image);
};


#endif //CROSSDETECTOR_HOUGH_ALGO_H
