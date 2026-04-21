cmake_minimum_required(VERSION 3.10)
project(ComputingF)

# 1. C++ Standard (Required for structured bindings auto [u, v])
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 2. Find External Packages
# This looks for the GSL library files and headers
find_package(GSL REQUIRED)

# 3. Include Directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/CppUnitLite)
# Add GSL headers to the include path
include_directories(${GSL_INCLUDE_DIRS})

# 4. Source Files
set(SOURCE_FILES
    main.cpp
    FNS.cpp
    CppUnitLite/Test.cpp
    CppUnitLite/TestResult.cpp
    CppUnitLite/TestRegistry.cpp
    CppUnitLite/Failure.cpp
    CppUnitLite/SimpleString.cpp  
)

# 5. Define the Executable
add_executable(F_estimator ${SOURCE_FILES})

# 6. Link Libraries
# We link GSL and its required CBLAS component
target_link_libraries(F_estimator PRIVATE GSL::gsl GSL::gslcblas)

if(UNIX)
    target_link_libraries(F_estimator PRIVATE m)
endif()