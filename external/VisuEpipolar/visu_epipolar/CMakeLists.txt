set(CMAKE_AUTOMOC ON)
find_package(Qt6 COMPONENTS Widgets REQUIRED)
add_subdirectory(libNumerics)

add_executable(visu_epipolar visu_epipolar.cpp ImagePoints.cpp SynchroF.cpp)
target_link_libraries(visu_epipolar VisuQt Numerics)
