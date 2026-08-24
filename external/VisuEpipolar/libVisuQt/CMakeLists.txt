set(CMAKE_AUTOMOC ON)
find_package(Qt6 COMPONENTS Widgets Gui REQUIRED)

add_library(VisuQt ContrastControl.cpp ImageZoom.cpp
  MovieControl.cpp MovieViewer.cpp MovieSequence.cpp RawStream.cpp)
target_link_libraries(VisuQt PUBLIC Qt6::Widgets Qt6::Gui)
target_include_directories(VisuQt PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
