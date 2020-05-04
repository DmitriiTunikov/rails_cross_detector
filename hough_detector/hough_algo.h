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
    void save_results(const std::string& path_to_file);
private:
    constexpr static const float m_not_parallel_cos_diff = 0.4;
    constexpr static const float m_parallel_cos_diff = 0.3;
    constexpr static const int m_approximate_diff = 10;
    constexpr static const int m_big_package_dist = 30;
    constexpr static const int m_small_package_dist = 10;
    constexpr static const int m_min_approximate_diff = 8;
    constexpr static const int m_max_approximate_diff = 15;


    struct Cell {
        cv_supp::Line line;
        std::vector<std::shared_ptr<Cell>> neighs;
        std::vector<std::shared_ptr<Cell>> parents;
        int y_min;
        int y_max;
        bool has_parent;
        bool has_intersection;
        explicit Cell(const cv_supp::Line& line_, const int y_min_, const int y_max_) : line(line_), y_min(y_min_), y_max(y_max_), has_parent(false),
                                                                                        has_intersection(false) {};

        bool has_same_direction(std::vector<std::shared_ptr<Cell>> elems, double same_dir_diff = m_parallel_cos_diff);
        std::shared_ptr<Cell> get_same_direction(std::vector<std::shared_ptr<Cell>> elems);
        static bool is_different_direction_lines(std::shared_ptr<Cell> c1, std::shared_ptr<Cell> c2);
    };

    using CellPtr = std::shared_ptr<Cell>;
    std::vector<std::vector<CellPtr>> m_grid;
    cv::Mat m_all_lines;
    cv::Mat m_image;
    cv::Mat m_canny_image;
    cv::Mat m_original_image;
    int m_canny_treshhold1;
    int m_canny_treshhold2;
    int m_max_strip_len;
    int m_min_strip_len;
    std::vector<cv::Point> m_cross_res;

    std::vector<std::vector<cv::Point>> m_cross_bags;

    void solve();
    void package_same_cross_points();
    std::vector<cv_supp::Line> get_lines_on_cropped(int y_min, int y_max);
    void add_result_point(const cv::Point2i& point);
    bool is_intersection_neighs(CellPtr c1, CellPtr c2, int same_direction_depth, int neighs_check_depth,
            HoughDetector::CellPtr came_from1 = CellPtr(), HoughDetector::CellPtr came_from2 = CellPtr());
    static bool is_same_direction_lines(cv_supp::Line l1, cv_supp::Line l2, double same_dir_diff = m_parallel_cos_diff);
    int get_size_by_y(int min_len, int max_len, int y);
};


#endif //CROSSDETECTOR_HOUGH_ALGO_H
