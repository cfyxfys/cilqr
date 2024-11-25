cc_library(
    name = "cilqr_cost",
    srcs = ["cilqr_cost.cpp"],
    hdrs = ["cilqr_cost.hpp"],
    includes = [
        "@eigen",
        "cilqr_define.hpp",
    ],
    deps = ["@eigen"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "cilqr_constraint",
    srcs = ["cilqr_constraint.cpp"],
    hdrs = ["cilqr_constraint.hpp"],
    includes = [
        "@eigen",
        "cilqr_define.hpp",
    ],
    deps = ["@eigen"],
    visibility = ["//visibility:public"],
)


cc_library(
    name = "cilqr_core",
    srcs = ["cilqr_core.cpp"],
    hdrs = ["cilqr_core.hpp"],
    includes = [
        "cilqr_define.hpp",
    ],
    deps = [
        ":cilqr_cost",
        ":cilqr_constraint",
    ],
)
