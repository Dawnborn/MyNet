add_library(common common.cpp)
target_link_libraries(common gflags glog gtest gtest_main)

add_library(Tensor Tensor.cc)
target_link_libraries(common MyNet_fbs)