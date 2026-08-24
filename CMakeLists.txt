cmake_minimum_required(VERSION 3.10)
project(ClassicalRelativePoseEstimation)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ------------------------------------------------------------
# Dependencies
# ------------------------------------------------------------

find_package(GSL REQUIRED)
find_package(Threads REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Imagine REQUIRED COMPONENTS LinAlg Images)


# ------------------------------------------------------------
# Sources
# ------------------------------------------------------------

set(SOURCE_FILES
    src/main.cpp
    src/FNS.cpp
    src/HEIV.cpp
    src/Renorm.cpp
    src/GaussNewton.cpp
    src/GetInliers.cpp
    src/Pipeline.cpp
    src/ORSAWrapper.cpp
    src/PoseEstimation.cpp
    src/Initialization.cpp

    external/Imagine/SIFT_VL.cpp
    external/Imagine/vl/generic.c
    external/Imagine/vl/host.c 
    external/Imagine/vl/imop.c 
    external/Imagine/vl/sift.c

    external/CppUnitLite/Test.cpp
    external/CppUnitLite/TestResult.cpp
    external/CppUnitLite/TestRegistry.cpp
    external/CppUnitLite/Failure.cpp
    external/CppUnitLite/SimpleString.cpp

    # --- libOrsa ---
    external/libOrsa/fundamental_model.cpp
    external/libOrsa/homography_model.cpp    
    external/libOrsa/orsa.cpp
    external/libOrsa/ransac.cpp
    external/libOrsa/lrtsac.cpp
    external/libOrsa/model_estimator.cpp
    external/libOrsa/conditioning.cpp
    external/libOrsa/sampling.cpp
    external/libOrsa/libNumerics/numerics.cpp
)

add_executable(Realtive_Pose_Estimator ${SOURCE_FILES})

# ------------------------------------------------------------
# Includes
# ------------------------------------------------------------

target_include_directories(Realtive_Pose_Estimator PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/external
    ${CMAKE_CURRENT_SOURCE_DIR}/external/CppUnitLite
    ${CMAKE_CURRENT_SOURCE_DIR}/external/libOrsa       
    ${CMAKE_CURRENT_SOURCE_DIR}/external/libOrsa/libNumerics
    ${GSL_INCLUDE_DIRS}
)

# ------------------------------------------------------------
# Linking
# ------------------------------------------------------------

target_link_libraries(Realtive_Pose_Estimator PRIVATE
    ${GSL_LIBRARIES}
    Threads::Threads
    Eigen3::Eigen
    Imagine::LinAlg 
    Imagine::Images
)

if(UNIX)
    target_link_libraries(Realtive_Pose_Estimator PRIVATE m)
endif()

include_directories("C:/ENPC/Imagine++-6.0.0/include")
link_directories("C:/ENPC/Imagine++-6.0.0/lib")