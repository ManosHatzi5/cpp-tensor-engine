cmake_minimum_required(VERSION 3.20)
project(ML_Systems_Engine)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 1. Fetch GoogleTest directly from the source
include(FetchContent)
FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG release-1.12.1
)
FetchContent_MakeAvailable(googletest)

# 2. Tell CMake where your header files will live
include_directories(include)

# 3. Create your core mathematical library (this compiles your actual code)
# As you add more .cpp files, add them to this list.
add_library(ml_engine src/Tensor2D.cpp)

# 4. Create the Test Executable
enable_testing()
add_executable(engine_tests tests/test_tensor.cpp
        include/Tensor2D.h
        src/Tensor2D.cpp
        tests/test_tensor.cpp
        include/Loss.h
        src/Loss.cpp
        include/Layer.h
        include/ReLULayer.h
        src/ReLULayer.cpp
        include/LinearLayer.h
        src/LinearLayer.cpp
        include/Sequential.h
        src/Sequential.cpp
        include/SigmoidLayer.h
        src/SigmoidLayer.cpp)

# 5. Link GoogleTest and your core library to the test executable
target_link_libraries(engine_tests gtest_main ml_engine)

# 6. Enable CLion's native testing UI
include(GoogleTest)
gtest_discover_tests(engine_tests)