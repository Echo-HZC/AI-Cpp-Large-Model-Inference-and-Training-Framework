cmake_minimum_required(VERSION 3.15)
project(AI VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 编译选项
if(MSVC)
    add_compile_options(/W4 /O2 /EHsc)
    # 开启多线程编译
    add_compile_options(/MP)
else()
    add_compile_options(-Wall -Wextra -O3 -march=native)
    # 可选：开启OpenMP加速
    # find_package(OpenMP)
    # if(OpenMP_CXX_FOUND)
    #     add_compile_options(-fopenmp)
    #     link_libraries(OpenMP::OpenMP_CXX)
    # endif()
endif()

# 头文件目录
include_directories(${CMAKE_SOURCE_DIR}/include)

# 核心库源文件
set(AI_CORE_SOURCES
    src/tensor.cpp
    src/math.cpp
    src/utils.cpp
    src/layers.cpp
    src/transformer.cpp
    src/model.cpp
    src/engine.cpp
)

# 静态库
add_library(ai_core STATIC ${AI_CORE_SOURCES})

# 主程序
add_executable(AI src/main.cpp)
target_link_libraries(AI ai_core)

# 测试程序（可选）
option(BUILD_TESTS "Build tests" OFF)
if(BUILD_TESTS)
    enable_testing()
    add_executable(test_model tests/test_model.cpp)
    target_link_libraries(test_model ai_core)
    add_test(NAME ModelTest COMMAND test_model)
endif()

# 安装规则
install(TARGETS AI DESTINATION bin)
install(DIRECTORY include/ai DESTINATION include)
