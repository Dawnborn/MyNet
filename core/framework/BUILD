load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library","cc_test")

cc_library(
    name = "Tensor",
    srcs = ["Tensor.cc"],
    hdrs = ["Tensor.hpp"],
    visibility = ["//visibility:public"],
    deps = [
        "//core/schema:MyNet_fbs",
        ":common",
    ],
)

cc_library(
    name = "common",
    srcs = ["common.cpp"],
    hdrs = ["common.hpp"],
    visibility = ["//visibility:public"],
    deps = [
        "@com_github_gflags_gflags//:gflags",
        "@com_github_google_glog//:glog",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "syncedmem_test",
    srcs = ["syncedmem_test.cpp"],
    deps = [":syncedmem"],
)

cc_library(
    name = "syncedmem",
    srcs = ["syncedmem.cpp"],
    hdrs = ["syncedmem.hpp"],
    deps = [":common"],
)
