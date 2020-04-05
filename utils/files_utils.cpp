//
// Created by dmitrii on 31.03.2020.
//

#include "files_utils.h"
#include <filesystem>

using namespace std;
namespace fs = std::filesystem;

std::vector<std::string> files_utils::get_files_from_dir(std::string dir_path) {
    vector<string> file_names;

    for (const auto& file_name : fs::directory_iterator(dir_path))
        file_names.push_back(file_name.path());

    return file_names;
}
