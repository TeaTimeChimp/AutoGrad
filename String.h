#pragma once

#include "NDShape.h"
#include <string>
#include <vector>


std::string to_string(const NDShape& v);

std::vector<std::string> Split(const std::string& str,const char sep);