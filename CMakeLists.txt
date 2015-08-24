CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

set(PROJECT_NAME video-stabilization)
project(${PROJECT_NAME})

set(CMAKE_CONFIGURATION_TYPES "Debug;Release" CACHE STRING "Configs" FORCE)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE})

find_package(OpenCV REQUIRED)
set(LIBRARY_DEPS ${OpenCV_LIBS})


# BUILD
add_subdirectory(samples)
# Add you directory here
# add_subdirectory(sample_YOUR_NAME)

# REPORT
message( STATUS "")
message( STATUS "General configuration for ${PROJECT_NAME}")
message( STATUS "======================================")
message( STATUS "")
message( STATUS "   Configuration:        ${CMAKE_BUILD_TYPE}")
message( STATUS "   OpenCV build path:    ${OpenCV_DIR}")
message( STATUS "")
