//
// Created by dmitrii on 01.04.2020.
//

#include <cmath>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <map>
#include "hough_algo.h"
#include "utils/cv_supp.h"
#include "utils/math_utils.h"
#include <fstream>

using namespace cv;

std::vector<cv::Mat> hough_algo::get_cropped_images(const cv::Mat &image, int crop_count) {
    const int max_size = 30;
    const int min_size = 10;
    int cur_size = min_size;
    //cur_size = min_size + (float(max_size) - min_size) / (m_img.rows - max_size) * cur_y;
    std::vector<cv::Mat> res;

    for (int i = 1; i <= crop_count; i++) {
        int y_max = image.rows * i / crop_count;
        int y_min = y_max - image.rows / crop_count;

        cv::Rect crop(Point(0, y_min), Point(image.cols, y_max));

        line(image, Point(0, y_min), Point(image.cols, y_min), Scalar(255, 0, 0), 2);

        res.push_back(image(crop));
    }

    return res;
}



void hough_algo::find_lines_on_cropped(std::vector<cv::Mat> &crop_images, int canny_treashhold1, int canny_treashhold2) {
    for (int i = 0; i < crop_images.size(); i++) {
        cv::Mat& cur_image = crop_images[i];

        //find edges by canny
//        Mat gx, gx_abs, vertical_edges;
//        Sobel(cur_image, gx, CV_64F, 1, 0, 1);
//        convertScaleAbs(gx, gx_abs );
//        addWeighted( gx_abs, 0, gx_abs, 1, 0, vertical_edges );

        cv::Mat canny;
        cv::Canny(cur_image, canny, canny_treashhold1, canny_treashhold2);

        //find lines by hough
        std::vector<Vec2f> lines;
        HoughLines(canny, lines, 1, CV_PI/180, 20, 0, 0);

        //draw lines
        draw_lines(lines, cur_image);
    }
}

void hough_algo::draw_lines(const std::vector<cv::Vec2f> &lines, cv::Mat &image) {
    for( size_t j = 0; j < lines.size(); j++)
    {
        cv_supp::draw_polar_line(image, lines[j], Scalar(0, 0, 255));
    }
}

void hough_algo::find_lines(cv::Mat &image, int crop_count, int canny_treashhold1, int canny_treashhold2) {
    std::vector<cv::Mat> croped_images = get_cropped_images(image, crop_count);

    find_lines_on_cropped(croped_images, canny_treashhold1, canny_treashhold2);
}

HoughDetector::HoughDetector(cv::Mat &image, int canny_treshhold1, int canny_treshhold2, int max_strip_len,
                             int min_strip_len) : m_canny_treshhold1(canny_treshhold1), m_canny_treshhold2(canny_treshhold2),
                             m_max_strip_len(max_strip_len), m_min_strip_len(min_strip_len) {
    image.copyTo(m_original_image);
    //crop 1/5 of iamge
    cv::Rect crop(Point(0, image.rows / 5), Point(image.cols, image.rows));
    m_image = image(crop);
}

std::vector<cv::Point> HoughDetector::get_cross_result() {
    m_grid.clear();
    m_cross_res.clear();

    solve();

    return m_cross_res;
}

void HoughDetector::solve() {
    int cur_y = m_image.rows - 1;
    int cur_size = 0;
    const int diff_eps = 8;


    cv::Mat all_lines;
    m_image.copyTo(all_lines);

    //generate grid
    int q = -1;
    while (true) {
        q++;
        cur_size = m_min_strip_len + (float(m_max_strip_len) - m_min_strip_len) / (m_image.rows - m_max_strip_len) * cur_y;

        if (cur_y - cur_size < 0)
            break;

        std::vector<cv_supp::Line> cur_lines = get_lines_on_cropped(cur_y - cur_size, cur_y);
        line(all_lines, Point2i(0, cur_y), Point2i(m_image.cols, cur_y), Scalar(255, 0, 0), 1);
        cv::putText(all_lines, std::to_string(q), Point2i(0, cur_y), 1, 1, Scalar(255 ,0, 0));

        std::map<int, std::vector<CellPtr>> new_cells_map;
        std::vector<CellPtr> cur_lines_as_cells;
        std::vector<CellPtr> prev_lines_as_cells = !m_grid.empty() ? m_grid[m_grid.size() - 1] : std::vector<CellPtr>();
        for (cv_supp::Line &line : cur_lines) {
            cv::line(all_lines, line.cartesLine.max, line.cartesLine.min, Scalar(0, 0, 255), 1);

            CellPtr cur_line_as_cell = std::make_shared<Cell>(line, cur_y - cur_size, cur_y);

            for (CellPtr prev_line_as_cell : prev_lines_as_cells) {
                if (abs(cur_line_as_cell->line.cartesLine.max.x - prev_line_as_cell->line.cartesLine.min.x) < diff_eps) {
                    prev_line_as_cell->neighs.push_back(cur_line_as_cell);
                    cur_line_as_cell->has_parent = true;
                }
            }

            cur_lines_as_cells.push_back(cur_line_as_cell);
        }

        m_grid.push_back(cur_lines_as_cells);
        cur_y -= cur_size;
    }

    //find crosses between lines on grid
    for (int i = 0; i < m_grid.size() - 2; i++) {
        for (int cell_idx = 0; cell_idx < m_grid[i].size(); cell_idx++) {
            CellPtr cur_cell = m_grid[i][cell_idx];

            if (i == 20 && cur_cell->line.cartesLine.min.x > 300 && cur_cell->line.cartesLine.min.x < 320)
                int a = 2;

            //if has neighs with different directions and every neigh has neigh with same direction as neigh, then it is intersection point!
            std::vector<CellPtr>& neighs = cur_cell->neighs;
            if (neighs.size() > 1) {
                for (int k = 0; k < neighs.size(); k++) {
                    for (int m = k + 1; m < neighs.size(); m++) {
                        if (HoughDetector::Cell::is_different_direction_lines(neighs[k], neighs[m])) {
                            if (neighs[k]->has_same_direction_neigh() && neighs[m]->has_same_direction_neigh()) {
                                m_cross_res.push_back(neighs[k]->line.cartesLine.max);
                            }
                        }
                        else {
                            CellPtr neigh_of_neigh_k = neighs[k]->get_same_direction_neigh();
                            CellPtr neigh_of_neigh_m = neighs[m]->get_same_direction_neigh();

                            if (neigh_of_neigh_k && neigh_of_neigh_m && HoughDetector::Cell::is_different_direction_lines(neigh_of_neigh_k, neigh_of_neigh_m)) {
                                if (neigh_of_neigh_k->has_same_direction_neigh() && neigh_of_neigh_m->has_same_direction_neigh()){
                                    m_cross_res.push_back(neighs[k]->line.cartesLine.max);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    //remove same cross res points
    const int min_dist_between_same_points = 20;
    int cur_cross_size = m_cross_res.size();
    for (int i = 0; i < cur_cross_size - 1; i++) {
        cv::circle(all_lines, m_cross_res[i], 2, Scalar(0, 255, 0), 2);
        for (int j = i + 1; j < cur_cross_size; j++) {
            int x_diff = abs(m_cross_res[i].x - m_cross_res[j].x), y_diff = abs(m_cross_res[i].y - m_cross_res[j].y);
            int dist = sqrt(x_diff * x_diff + y_diff * y_diff);
            if (dist < min_dist_between_same_points) {
                m_cross_res.erase(m_cross_res.begin() + j);
                j--;
                cur_cross_size--;
            }
        }
    }

    imshow("all lines", all_lines);
}


std::vector<cv_supp::Line> HoughDetector::get_lines_on_cropped(int y_min, int y_max) {
    std::vector<cv_supp::Line> res_lines;

    cv::Rect crop(Point(0, y_min), Point(m_image.cols, y_max));
    cv::Mat cur_image = m_image(crop);

    cv::Mat canny;
    cv::Canny(cur_image, canny, m_canny_treshhold1, m_canny_treshhold2);

    //find lines by hough
    std::vector<Vec2f> lines;
    HoughLines(canny, lines, 1, CV_PI/180, (y_max - y_min) * 2/3, 0, 0, -CV_PI / 3.5, CV_PI / 3.5);

    //conver to cartesian
    std::vector<cv_supp::CartesLine> cartesLines;
    for (Vec2f& line : lines) {
        cartesLines.push_back(cv_supp::from_polar_to_cartesian(line, y_max - y_min, 0));
        cartesLines[cartesLines.size() - 1].max.y = y_max;
        cartesLines[cartesLines.size() - 1].min.y = y_min;
    }

    //approximate same lines by only one line
    int cur_size = cartesLines.size();
    for (int i = 0; i < cur_size; i++) {
        cv_supp::CartesLine& line_i = cartesLines[i];
        int res_x1 = line_i.max.x;
        int res_x2 = line_i.min.x;

        int same_lines_count = 1;
        for (int j = i + 1; j < cur_size; j++){
            cv_supp::CartesLine& line_j = cartesLines[j];

            if (math_utils::get_diff(line_i.max.x, line_j.max.x) < m_x_diff && math_utils::get_diff(line_i.min.x, line_j.min.x) < m_x_diff) {
                same_lines_count++;
                res_x1 += line_j.max.x;
                res_x2 += line_j.min.x;
                cartesLines.erase(cartesLines.begin() + j);
                lines.erase(lines.begin() + j);
                j--;
                cur_size--;
            }
        }

        line_i.max.x = res_x1 / same_lines_count;
        line_i.min.x = res_x2 / same_lines_count;
    }

    for (int i = 0; i < cartesLines.size(); i++)
        res_lines.push_back(cv_supp::Line{cartesLines[i], lines[i]});

    return res_lines;
}

void HoughDetector::draw_cross_res() {
    for (cv::Point2i& point : m_cross_res) {
        cv::circle(m_image, point, 3, Scalar(0, 255, 0), 2);
    }
}

void HoughDetector::save_results(const std::string& path_to_file) {
    std::ofstream f(path_to_file);
    if (!f.is_open())
        throw std::runtime_error("can't open " + path_to_file);

    for (cv::Point& res : m_cross_res) {
        f << res.x << " " << res.y  + m_original_image.rows / 5 << std::endl;
    }
}

bool HoughDetector::Cell::has_same_direction_neigh() {
    for (CellPtr neigh : neighs) {
        if (math_utils::get_diff(cv_supp::get_line_cos(line.cartesLine.max, line.cartesLine.min),
                cv_supp::get_line_cos(neigh->line.cartesLine.max, neigh->line.cartesLine.min)) < m_parallel_cos_diff)
            return true;
    }

    return false;
}

std::shared_ptr<HoughDetector::Cell> HoughDetector::Cell::get_same_direction_neigh() {
    for (CellPtr neigh : neighs) {
        if (math_utils::get_diff(cv_supp::get_line_cos(line.cartesLine.max, line.cartesLine.min),
                                 cv_supp::get_line_cos(neigh->line.cartesLine.max, neigh->line.cartesLine.min)) < m_parallel_cos_diff)
            return neigh;
    }

    return std::shared_ptr<Cell>();
}

bool HoughDetector::Cell::is_different_direction_lines(std::shared_ptr<HoughDetector::Cell> c1, std::shared_ptr<HoughDetector::Cell> c2) {
    float c1_cos = cv_supp::get_line_cos(c1->line.cartesLine.max, c1->line.cartesLine.min);
    float c2_cos = cv_supp::get_line_cos(c2->line.cartesLine.max, c2->line.cartesLine.min);

    float cos_diff = math_utils::get_diff(c1_cos, c2_cos);

    return cos_diff > m_parallel_cos_diff;
}
