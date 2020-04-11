//
// Created by dmitrii on 31.03.2020.
//

#ifndef CROSSDETECTOR_FILES_UTILS_H
#define CROSSDETECTOR_FILES_UTILS_H

#include <vector>
#include <string>

namespace files_utils {
    struct FileName {
        std::string path;
        std::string name;
    };

    std::vector<FileName> get_files_from_dir(std::string dir_name);
    void remove_dir(const std::string& dir_path);
    void create_dir(const std::string& dir_path);
};


#endif //CROSSDETECTOR_FILES_UTILS_H
