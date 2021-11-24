//
// Created by hjp on 2021/11/4.
//

#ifndef MYNET_CORE_FRAMEWORK_COMMON_HPP_
#define MYNET_CORE_FRAMEWORK_COMMON_HPP_

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <utility>  // pair
#include <vector>

#ifndef GFLAGS_GFLAGS_H_
namespace gflags = google;
#endif

#define DISABLE_COPY_AND_ASSIGN(classname) \
  classname(const classname&)=delete;\
  classname& operator=(const classname&)=delete

#endif //MYNET_CORE_FRAMEWORK_COMMON_HPP_
