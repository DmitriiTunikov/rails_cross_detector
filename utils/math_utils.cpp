//
// Created by dmitrii on 02.04.2020.
//

#include <cmath>
#include <opencv2/core/types.hpp>
#include "math_utils.h"

float math_utils::get_diff(float v1, float v2) {
    if (v1 * v2 > 0) {
        return fabs(v2 - v1);
    }
    else {
        if (v1 < 0)
            return -v1 + v2;
        else
            return -v2 + v1;
    }
}