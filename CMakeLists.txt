cmake_minimum_required(3.10)

project(MyNet)

set(GLOG_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/third-party/glog")
set(GFLAGS_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/third-party/gflags")
set(FLATBUFFERS_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/third-party/flatbuffers")

include(third-party/glog)
include(third-party/gflags)
include(third-party/flattbuffers)

find_package(glog REQUIRED CONFIG HINTS "${GLOG_ROOT}" NO_DEFAULT_PATH)

find_package(gflags REQUIRED CONFIG HINTS "${GFLAGS_ROOT}" NO_DEFAULT_PATH)

find_package(flattbuffers REQUIRED CONFIG HINTS "${FLATBUFFERS_ROOT" NO_DEFAULT_PATH)

    
add_subdirectory(core/framework)
add_subdirectory(core/schema)

add_subdirectory(test)