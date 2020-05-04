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

HoughDetector::HoughDetector(cv::Mat &image, int canny_treshhold1, int canny_treshhold2, int max_strip_len,
                             int min_strip_len) : m_canny_treshhold1(canny_treshhold1), m_canny_treshhold2(canny_treshhold2),
                             m_max_strip_len(max_strip_len), m_min_strip_len(min_strip_len) {
    image.copyTo(m_original_image);

    //crop 1/5 of iamge
    cv::Rect crop(Point(0, image.rows / 5), Point(image.cols, image.rows));
    m_image = image(crop);

    //make canny edge detection and remove all horisontal lines
    Canny(m_image, m_canny_image, canny_treshhold1, canny_treshhold2);

    cv_supp::remove_horizontal_edges(m_canny_image);

    imshow("canny image", m_canny_image);
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

    m_image.copyTo(m_all_lines);

    //generate grid
    int q = -1;
    while (true) {
        q++;
        cur_size = m_min_strip_len + (float(m_max_strip_len) - m_min_strip_len) / (m_image.rows - m_max_strip_len) * cur_y;

        if (cur_y - cur_size < 0)
            break;

        std::vector<cv_supp::Line> cur_lines = get_lines_on_cropped(cur_y - cur_size, cur_y);
        line(m_all_lines, Point2i(0, cur_y), Point2i(m_image.cols, cur_y), Scalar(255, 0, 0), 1);
        cv::putText(m_all_lines, std::to_string(q), Point2i(0, cur_y), 1, 1, Scalar(255 ,0, 0));

        std::map<int, std::vector<CellPtr>> new_cells_map;
        std::vector<CellPtr> cur_lines_as_cells;
        std::vector<CellPtr> prev_lines_as_cells = !m_grid.empty() ? m_grid[m_grid.size() - 1] : std::vector<CellPtr>();
        for (cv_supp::Line &line : cur_lines) {
            if (line.cartesLine.min.x > 400 && line.cartesLine.min.x < 405 && line.cartesLine.min.y == 259)
                int a = 2;

            cv::line(m_all_lines, line.cartesLine.max, line.cartesLine.min, Scalar(0, 0, 255), 1);

            CellPtr cur_line_as_cell = std::make_shared<Cell>(line, cur_y - cur_size, cur_y);
            for (CellPtr prev_line_as_cell : prev_lines_as_cells) {
                //const int diff_eps = 3;
                if (abs(cur_line_as_cell->line.cartesLine.max.x - prev_line_as_cell->line.cartesLine.min.x) < get_size_by_y(3, 5, cur_y)) {
                    prev_line_as_cell->neighs.push_back(cur_line_as_cell);
                    cur_line_as_cell->parents.push_back(prev_line_as_cell);
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
            if (cur_cell->line.cartesLine.min.x > 400 && cur_cell->line.cartesLine.min.x < 405 && cur_cell->line.cartesLine.min.y == 259)
                int a = 2;

//            if (!cur_cell->has_same_direction(cur_cell->parents, m_parallel_cos_diff))
//                continue;

            //if has neighs with different directions and every neigh has neigh with same direction as neigh, then it is intersection point!
            std::vector<CellPtr> &neighs = cur_cell->neighs;
            if (neighs.size() > 1) {
                for (int k = 0; k < neighs.size(); k++) {
                    for (int m = k + 1; m < neighs.size(); m++) {
//                        if (is_same_direction_lines(cur_cell->line, neighs[k]->line) ||
//                            is_same_direction_lines(cur_cell->line, neighs[m]->line)) {
                        if (is_intersection_neighs(neighs[k], neighs[m], 2, 2)) {
                            add_result_point(cur_cell->line.cartesLine.min);
                            cv::line(m_all_lines, cur_cell->line.cartesLine.min, cur_cell->line.cartesLine.max,
                                     Scalar(255, 255, 255));
                        }
                    }
                }
            }
        }
    }
    //package_same_cross_points();
    imshow("all lines", m_all_lines);
}


std::vector<cv_supp::Line> HoughDetector::get_lines_on_cropped(int y_min, int y_max) {
    std::vector<cv_supp::Line> res_lines;

    cv::Rect crop(Point(0, y_min), Point(m_image.cols, y_max));
    cv::Mat cur_canny_image = m_canny_image(crop);

    //find lines by hough
    std::vector<Vec2f> lines;
    HoughLines(cur_canny_image, lines, 1, CV_PI / 180, (y_max - y_min) * 2 / 3, 0, 0, -CV_PI / 3.5, CV_PI / 3.5);

    //conver to cartesian
    std::vector<cv_supp::CartesLine> cartesLines;
    for (Vec2f& line : lines) {
        cartesLines.push_back(cv_supp::from_polar_to_cartesian(line, y_max - y_min, 0));
        cartesLines[cartesLines.size() - 1].max.y = y_max;
        cartesLines[cartesLines.size() - 1].min.y = y_min;
    }

    //approximate same lines by only one line
    /*
    int cur_size = cartesLines.size();
    for (int i = 0; i < cur_size; i++) {
        cv_supp::CartesLine& line_i = cartesLines[i];
        int res_x1 = line_i.max.x;
        int res_x2 = line_i.min.x;

        int same_lines_count = 1;
        for (int j = i + 1; j < cur_size; j++){
            cv_supp::CartesLine& line_j = cartesLines[j];

            if (math_utils::get_diff(line_i.max.x, line_j.max.x) < m_approximate_diff && math_utils::get_diff(line_i.min.x, line_j.min.x) < m_approximate_diff) {
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
    }*/

    for (int i = 0; i < cartesLines.size(); i++)
        res_lines.push_back(cv_supp::Line{cartesLines[i], lines[i]});

    return res_lines;
}

void HoughDetector::draw_cross_res() {
    for (cv::Point2i& point : m_cross_res) {
        cv::circle(m_image, point, 3, Scalar(0, 255, 0), 2);
    }
    imshow("original", m_image);
}

void HoughDetector::save_results(const std::string& path_to_file) {
    std::ofstream f(path_to_file);
    if (!f.is_open())
        throw std::runtime_error("can't open " + path_to_file);

    for (cv::Point& res : m_cross_res) {
        f << res.x << " " << res.y  /*+ m_original_image.rows / 5*/ << std::endl;
    }
}

void HoughDetector::add_result_point(const cv::Point2i& p) {
    /*const int min_dist_between_same_points = p.y > 0.2 * m_image.rows ? 25 : 10;

    bool was_used = false;
    for (cv::Point2i& res_point : m_cross_res) {
            int x_diff = abs(res_point.x - p.x), y_diff = abs(res_point.y - p.y);
            int dist = sqrt(x_diff * x_diff + y_diff * y_diff);
            if (dist < min_dist_between_same_points) {
                res_point.x = (res_point.x + p.x) / 2, res_point.y = (res_point.y + p.y) / 2;
                was_used = true;
            }
    }

    if (!was_used)*/
    m_cross_res.push_back(p);
}

bool HoughDetector::is_intersection_neighs(CellPtr c1, CellPtr c2, int same_direction_depth, int neighs_check_depth, CellPtr came_from1, CellPtr came_from2) {

    if (c1->has_intersection || c2->has_intersection)
        return false;

    if (same_direction_depth == 0)
    {
        came_from1->has_intersection = true;
        came_from2->has_intersection = true;
        c1->has_intersection = true;
        c1->has_intersection = true;

        cv::line(m_all_lines, came_from1->line.cartesLine.min, came_from1->line.cartesLine.max, Scalar(0, 255, 255));
        cv::line(m_all_lines, came_from2->line.cartesLine.min, came_from2->line.cartesLine.max, Scalar(0, 255, 255));
        return true;
    }

    if (HoughDetector::Cell::is_different_direction_lines(c1, c2)) {
//        cv::line(m_all_lines, c1->line.cartesLine.min, c1->line.cartesLine.max, Scalar(0, 0, 0));
//        cv::line(m_all_lines, c2->line.cartesLine.min, c2->line.cartesLine.max, Scalar(0, 0, 0));
//        return true;
        bool both_has_same_direction_neighs = c1->has_same_direction(c1->neighs) && c2->has_same_direction(c2->neighs);
        if (both_has_same_direction_neighs) {
            CellPtr c1_neigh = c1->get_same_direction(c1->neighs);
            CellPtr c2_neigh = c2->get_same_direction(c2->neighs);
            if (c1_neigh->has_same_direction(c1_neigh->neighs) && c2_neigh->has_same_direction(c2_neigh->neighs))
            {
                 return is_intersection_neighs(c1->get_same_direction(c1->neighs), c2->get_same_direction((c2->neighs)),
                                          same_direction_depth - 1, neighs_check_depth, c1, c2);
            }
        }
    }
    if (neighs_check_depth > 1){
        CellPtr c1_neigh = c1->get_same_direction(c1->neighs);
        CellPtr c2_neigh = c2->get_same_direction(c2->neighs);
        if (c1_neigh && c2_neigh)
            return is_intersection_neighs(c1_neigh, c2_neigh, 1, neighs_check_depth - 1, c1, c2);
    }

    return false;
}

int HoughDetector::get_size_by_y(int min_len, int max_len, int y) {
    //m_min_strip_len + (float(m_max_strip_len) - m_min_strip_len) / (m_image.rows - m_max_strip_len) * cur_y;
    return min_len + (float(max_len) - min_len) / (m_image.rows - max_len) * y;
}

std::shared_ptr<HoughDetector::Cell> HoughDetector::Cell::get_same_direction(std::vector<std::shared_ptr<Cell>> elems) {
    for (CellPtr neigh : elems) {
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

    return cos_diff > m_not_parallel_cos_diff;
}

bool HoughDetector::is_same_direction_lines(cv_supp::Line l1, cv_supp::Line l2, double same_dir_diff) {
    return math_utils::get_diff(cv_supp::get_line_cos(l1.cartesLine.max, l1.cartesLine.min),
                                cv_supp::get_line_cos(l2.cartesLine.max, l2.cartesLine.min)) < same_dir_diff;
}

void HoughDetector::package_same_cross_points() {
    std::vector<std::vector<cv::Point>> bags;

    for (auto it1 = m_cross_res.begin(); it1 != m_cross_res.end(); it1++) {
        bags.push_back(std::vector<cv::Point>{*it1});
        for (auto it2 = it1 + 1; it2 != m_cross_res.end(); it2++) {
            int x_diff = abs(it1->x - it2->x), y_diff = abs(it1->y - it2->y);
            int dist = sqrt(x_diff * x_diff + y_diff * y_diff);
            int min_dist_in_bag = (it1->y > 0.2 * m_image.rows ? m_big_package_dist : m_small_package_dist);
            if (dist < min_dist_in_bag) {
                bags[bags.size() - 1].push_back(*it2);
                m_cross_res.erase(it2);
                it2--;
            }
        }
    }

    m_cross_res.clear();
    bool all_bags_separate = true;
    for (std::vector<cv::Point>& bag : bags) {
        if (bag.size() > 1)
            all_bags_separate = false;

        cv::Point res(0, 0);

        for (cv::Point bag_point : bag) {
            res.x += bag_point.x;
            res.y += bag_point.y;
        }
        res.x /= bag.size();
        res.y /= bag.size();
        m_cross_res.push_back(res);
    }

    if (!all_bags_separate)
        package_same_cross_points();
}

bool HoughDetector::Cell::has_same_direction(std::vector<std::shared_ptr<Cell>> elems, double same_dir_diff) {
    for (CellPtr parent : elems) {
        if (HoughDetector::is_same_direction_lines(line, parent->line))
            return true;
    }

    return false;
}
