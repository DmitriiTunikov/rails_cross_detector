//
// Created by dmitrii on 31.03.2020.
//

#include "files_utils.h"
#include <filesystem>

using namespace std;
namespace fs = std::filesystem;

std::vector<files_utils::FileName> files_utils::get_files_from_dir(std::string dir_path) {
    vector<FileName> file_names;

    for (const auto& file_name : fs::directory_iterator(dir_path)){
        file_names.push_back(FileName{file_name.path(), file_name.path().filename()});
    }

    return file_names;
}

void files_utils::remove_dir(const std::string& dir_path) {
    fs::remove_all(dir_path);
}

void files_utils::create_dir(const std::string &dir_path) {
    fs::create_directories(dir_path);
}
