cmake_minimum_required(VERSION 2.8)
project(lane_car)
#Import the headers
include_directories(include)
include_directories(YoloNet)
include_directories(camera_cal)

macro(use_cxx11)
  if (CMAKE_VERSION VERSION_LESS "3.1")
    if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
      set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++11")
    endif ()
  else ()
    set (CMAKE_CXX_STANDARD 11)
  endif ()
endmacro(use_cxx11)

use_cxx11()

#Import OpenCV libraries
find_package(OpenCV REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS} )

set(SOURCES src/camera.cpp src/image_mask.cpp src/lane_detection.cpp src/main.cpp src/car_detection.cpp)

add_executable(lane_detection ${SOURCES})
target_link_libraries(lane_detection ${OpenCV_LIBRARIES})
