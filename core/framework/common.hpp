//
// Created by hjp on 2021/11/4.
//

#ifndef MYNET_CORE_FRAMEWORK_COMMON_HPP_
#define MYNET_CORE_FRAMEWORK_COMMON_HPP_

#include <cstdint>
#include <glog/logging.h>
#include <gflags/gflags.h>
#include <gtest/gtest.h>

#include <cstring>
#include <iostream>
#include <fstream>
#include <sstream>
#include <set>
#include <map>
#include <string>
#include <vector>
#include <utility> // pair

#ifndef GFLAGS_GFLAGS_H_
namespace gflags = google;
#endif

#define DISABLE_COPY_AND_ASSIGN(classname) \
  private:
  classname(const classname&);\
  classname &operator=(const classname&)

#endif //MYNET_CORE_FRAMEWORK_COMMON_HPP_
