cmake_minimum_required(VERSION 3.18)
project(mvs_camera_backend LANGUAGES CXX)

include(FetchContent)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(MVS_SDK_ROOT "$ENV{MVS_SDK_ROOT}" CACHE PATH "Path to the Hikrobot MVS SDK root")
if(NOT MVS_SDK_ROOT)
  set(MVS_SDK_ROOT "/opt/MVS" CACHE PATH "Path to the Hikrobot MVS SDK root" FORCE)
endif()

find_path(
  MVS_SDK_INCLUDE_DIR
  NAMES MvCameraControl.h
  PATHS
    "${MVS_SDK_ROOT}/Development/Includes"
    "${MVS_SDK_ROOT}/Includes"
    "${MVS_SDK_ROOT}/include"
  REQUIRED
)

find_library(
  MVS_SDK_LIBRARY
  NAMES MvCameraControl libMvCameraControl
  PATHS
    "${MVS_SDK_ROOT}/Development/Libraries/64"
    "${MVS_SDK_ROOT}/Development/Libraries"
    "${MVS_SDK_ROOT}/lib/64"
    "${MVS_SDK_ROOT}/lib"
  REQUIRED
)
get_filename_component(MVS_SDK_LIBRARY_DIR "${MVS_SDK_LIBRARY}" DIRECTORY)

if(CMAKE_VERSION VERSION_LESS 3.18)
  set(DEV_MODULE Development)
else()
  set(DEV_MODULE Development.Module)
endif()

find_package(Python 3.9 COMPONENTS Interpreter ${DEV_MODULE} REQUIRED)
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_VARIABLE NANOBIND_CMAKE_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
  ERROR_QUIET
  RESULT_VARIABLE NANOBIND_CMAKE_DIR_RESULT
)
if(NANOBIND_CMAKE_DIR_RESULT EQUAL 0 AND NANOBIND_CMAKE_DIR)
  list(PREPEND CMAKE_PREFIX_PATH "${NANOBIND_CMAKE_DIR}")
endif()

find_package(nanobind CONFIG QUIET)
if(NOT nanobind_FOUND)
  FetchContent_Declare(
    nanobind
    GIT_REPOSITORY https://github.com/wjakob/nanobind.git
    GIT_TAG 4bfeca03aa67acb211a8a1355a3b11c05fbee8a9
    GIT_SHALLOW TRUE
    GIT_SUBMODULES_RECURSE TRUE
  )
  FetchContent_MakeAvailable(nanobind)
endif()

add_library(mvs_camera_backend STATIC src/mvs_camera_backend.cpp)
target_include_directories(
  mvs_camera_backend
  PRIVATE
    "${MVS_SDK_INCLUDE_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
)
target_link_libraries(mvs_camera_backend PRIVATE "${MVS_SDK_LIBRARY}")
set_target_properties(
  mvs_camera_backend
  PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)

nanobind_add_module(_mvs_camera NB_STATIC src/mvs_camera_module.cpp)
target_include_directories(
  _mvs_camera
  PRIVATE
    "${MVS_SDK_INCLUDE_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
)
target_link_libraries(_mvs_camera PRIVATE mvs_camera_backend "${MVS_SDK_LIBRARY}")
set_target_properties(
  _mvs_camera
  PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)

if(UNIX AND NOT APPLE)
  set_target_properties(
    _mvs_camera
    PROPERTIES
      BUILD_RPATH "$ORIGIN;${MVS_SDK_LIBRARY_DIR}"
      INSTALL_RPATH "$ORIGIN;${MVS_SDK_LIBRARY_DIR}"
  )
endif()
