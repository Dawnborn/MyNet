load("@rules_cc//cc:defs.bzl", "cc_binary","cc_library")

cc_library(
    name = "Tensor",
    srcs = ["Tensor.cc"],
    hdrs = ["Tensor.hpp"],
    deps = [
        "//core/schema:MyNet_fbs",
    ],
    visibility = ["//visibility:public"],
)