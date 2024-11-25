# 璁剧疆 vcpkg 宸ュ叿閾炬枃浠讹紙鏍规嵁浣犵殑 vcpkg 瀹夎浣嶇疆淇敼璺緞锛?
set(CMAKE_TOOLCHAIN_FILE "${CMAKE_CURRENT_SOURCE_DIR}/../vcpkg/scripts/buildsystems/vcpkg.cmake"
    CACHE STRING "Vcpkg toolchain file")

cmake_minimum_required(VERSION 3.10)
project(CILQR CXX)

# 璁剧疆 C++ 鏍囧噯
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 鏌ユ壘 Eigen3
find_package(Eigen3 CONFIG REQUIRED)

# 鏀堕泦婧愭枃浠?
file(GLOB_RECURSE SOURCES 
    "src/*.cpp"
    "src/*.hpp"
    "include/*.hpp"
)

# 鍒涘缓搴撴枃浠讹紙闈欐€佸簱锛?
add_library(${PROJECT_NAME} STATIC ${SOURCES})

# 娣诲姞澶存枃浠惰矾寰?
target_include_directories(${PROJECT_NAME}
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# 閾炬帴 Eigen3
target_link_libraries(${PROJECT_NAME} PRIVATE Eigen3::Eigen)
