workspace(
    name = "cilqr",
)

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

http_archive(
    name = "eigen",
    urls = ["https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz"],  # Eigen 3.4.0 的下载地址
    strip_prefix = "eigen-3.4.0",  # 解压后的目录前缀
    build_file_content = """
cc_library(
    name = "eigen",
    hdrs = glob(["Eigen/**/*.h", "unsupported/**/*.h"]),  # 添加所有头文件
    includes = ["."],  # 指定头文件的根目录
    visibility = ["//visibility:public"],  # 设置公共可见性
)
""",
)