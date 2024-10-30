# 设置 vcpkg 工具链文件（根据你的 vcpkg 安装位置修改路径）
set(CMAKE_TOOLCHAIN_FILE "${CMAKE_CURRENT_SOURCE_DIR}/../vcpkg/scripts/buildsystems/vcpkg.cmake"
    CACHE STRING "Vcpkg toolchain file")

cmake_minimum_required(VERSION 3.10)
project(CILQR CXX)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 查找 Eigen3
find_package(Eigen3 CONFIG REQUIRED)

# 收集源文件
file(GLOB_RECURSE SOURCES 
    "src/*.cpp"
    "src/*.hpp"
    "include/*.hpp"
)

# 创建库文件（静态库）
add_library(${PROJECT_NAME} STATIC ${SOURCES})

# 添加头文件路径
target_include_directories(${PROJECT_NAME}
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# 链接 Eigen3
target_link_libraries(${PROJECT_NAME} PRIVATE Eigen3::Eigen)
