# cmake_minimum_required(VERSION 3.10)
# project(ComputingF)

# # 1. C++ Standard (Required for structured bindings auto [u, v])
# set(CMAKE_CXX_STANDARD 17)
# set(CMAKE_CXX_STANDARD_REQUIRED ON)

# # 2. Find External Packages
# # This looks for the GSL library files and headers
# find_package(GSL REQUIRED)

# # 3. Include Directories
# include_directories(${CMAKE_CURRENT_SOURCE_DIR})
# include_directories(${CMAKE_CURRENT_SOURCE_DIR}/CppUnitLite)
# # Add GSL headers to the include path
# include_directories(${GSL_INCLUDE_DIRS})

# # 4. Source Files
# set(SOURCE_FILES
#     main.cpp
#     FNS.cpp
#     HEIV.cpp
#     Renorm.cpp
#     GaussNewton.cpp
#     Initialization.cpp
#     CppUnitLite/Test.cpp
#     CppUnitLite/TestResult.cpp
#     CppUnitLite/TestRegistry.cpp
#     CppUnitLite/Failure.cpp
#     CppUnitLite/SimpleString.cpp  
# )

# # 5. Define the Executable
# add_executable(F_estimator ${SOURCE_FILES})

# # 6. Link Libraries
# # We link GSL and its required CBLAS component
# target_link_libraries(F_estimator PRIVATE GSL::gsl GSL::gslcblas)

# if(UNIX)
#     target_link_libraries(F_estimator PRIVATE m)
# endif()

cmake_minimum_required(VERSION 3.10)
project(ComputingF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ------------------------------------------------------------
# Dependencies
# ------------------------------------------------------------

find_package(GSL REQUIRED)
find_package(Threads REQUIRED)
find_package(Eigen3 REQUIRED)


# ------------------------------------------------------------
# Sources
# ------------------------------------------------------------

set(SOURCE_FILES
    main.cpp
    FNS.cpp
    HEIV.cpp
    Renorm.cpp
    GaussNewton.cpp
    Initialization.cpp
    CppUnitLite/Test.cpp
    CppUnitLite/TestResult.cpp
    CppUnitLite/TestRegistry.cpp
    CppUnitLite/Failure.cpp
    CppUnitLite/SimpleString.cpp
)

add_executable(F_estimator ${SOURCE_FILES})

# ------------------------------------------------------------
# Includes
# ------------------------------------------------------------

target_include_directories(F_estimator PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/CppUnitLite
    ${GSL_INCLUDE_DIRS}
)

# ------------------------------------------------------------
# Linking
# ------------------------------------------------------------

target_link_libraries(F_estimator PRIVATE
    ${GSL_LIBRARIES}
    Threads::Threads
    Eigen3::Eigen
)

if(UNIX)
    target_link_libraries(F_estimator PRIVATE m)
endif()