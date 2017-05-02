
cmake_minimum_required(VERSION 3.1)

PROJECT(model-pool CXX)

#set(CMAKE_VERBOSE_MAKEFILE 1)
#set(CMAKE_BUILD_TYPE "Debug")
SET(CMAKE_BUILD_TYPE Release)

#INCLUDE_DIRECTORIES({PROJECT_SOURCE_DIR})

###########################################
#collect all of header files for #include
###########################################

include_directories(src/core/tensor)
include_directories(src/nn/modules)
include_directories(src/nn/modules/container)
include_directories(src/nn/modules/convolutionLayers)
include_directories(src/nn/modules/simpleLayers)
include_directories(src/nn/modules/tableLayers)
include_directories(src/nn/modules/transfer)
include_directories(src/nn/criterion)


#######################################################################################################
### followings are cmake command line option for checking exectution time on different environments ###
#######################################################################################################

add_definitions("-Wall -std=c++14")

#############################
### Define user options   ###
#############################


option(USE_SSE        "Build tiny-dnn with SSE library support"    OFF)
option(USE_AVX        "Build tiny-dnn with AVX library support"    OFF)
option(USE_AVX2       "Build tiny-dnn with AVX2 library support"   OFF)
option(USE_TBB        "Build tiny-dnn with TBB library support"    OFF)
option(USE_OMP        "Build tiny-dnn with OMP library support"    ON)
option(USE_NNPACK     "Build tiny-dnn with NNPACK library support" OFF)
option(USE_OPENCL     "Build tiny-dnn with OpenCL library support" OFF)
option(USE_LIBDNN     "Build tiny-dnn with GreenteaLibDNN library support" OFF)
option(USE_SERIALIZER "Build tiny-dnn with Serialization support"   ON)
option(USE_DOUBLE     "Build tiny-dnn with double precision computations"  OFF)
option(USE_IMAGE_API  "Build tiny-dnn with Image API support"       ON)
option(USE_GEMMLOWP   "Build tiny-dnn with gemmlowp support"       OFF)

option(BUILD_TESTS    "Set to ON to build tests"              OFF)
option(BUILD_EXAMPLES "Set to ON to build examples"           OFF)
option(BUILD_DOCS     "Set to ON to build documentation"      OFF)
option(COVERALLS      "Set to ON to build with code coverage" OFF)
option(PROFILE        "Set to ON to build with profiling"     OFF)

if(NOT USE_SERIALIZER)
  add_definitions(-DCNN_NO_SERIALIZATION)
endif()

if(USE_DOUBLE)
  add_definitions(-DCNN_USE_DOUBLE)
endif()

if(USE_IMAGE_API)
  add_definitions(-DDNN_USE_IMAGE_API)
endif()

if(USE_GEMMLOWP)
  add_definitions(-DUSE_GEMMLOWP)
endif()

#OPTION (USE_OpenMP "Use OpenMP" ON)

message("${USE_OMP}")

# Find Open Multi-Processing (OpenMP)
find_package(OpenMP REQUIRED)

#if(USE_OMP AND OPENMP_FOUND)
if(OPENMP_FOUND)
  message(STATUS "Found OpenMP!!!")
  
  # In case that OMP is found we force to disable Intel TBB since
  # tiny-dnn does not support mutiple multithreading backends.
  set(USE_TBB OFF)
  add_definitions(-DCNN_USE_OMP)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
elseif(USE_OMP AND NOT OPENMP_FOUND)
  # In case the user sets the flag USE_OMP to ON, the CMake build-tree
  # will require to find OMP in your system. Otherwise, the user can
  # set the CMAKE_C_FLAGS and CMAKE_CXX_FLAGS by hand.
  message(FATAL_ERROR "Can't find OpenMP. Please set OpenMP_C_FLAGS & "
    "OpenMP_CXX_FLAGS")
  
endif()
  

set(PIKA,"PIKACHU")

#configure_file( /examples1.cpp.in main.cpp @ONLY )
add_definitions(-D_POKEMON="oya")


################################################
### following is for generating a example script
################################################

SET(PRO
  examples/example4.cpp
  )

SET(TEST_NAME example1)

add_executable(${TEST_NAME}  
  ${PRO}
  )


#########################################################################
### following is a typical way to have header only lib using interface
### this would be uncomment replacing simple header only installation for
### reflecting parameters of hardware.
#########################################################################

#ADD_LIBRARY(${PROJECT_NAME} INTERFACE)
#target_include_directories(${PROJECT_NAME} INTERFACE ...)
#target_sources(${PROJECT_NAME} INTERFACE
#  ${CMAKE_CURRENT_SOURCE_DIR}/init.hpp
#  )


###############################################
### following is for installation
###############################################


set(project_library_target_name ${PROJECT_NAME})

if(WIN32)
  set(include_install_dir Include)
  set(include_install_dir_full Include)
  set(config_install_dir CMake)
  
elseif(UNIX)
  
  set(include_install_dir include)
  set(include_install_dir_postfix "${project_library_target_name}")
  set(include_install_dir_full    "${include_install_dir}/${include_install_dir_postfix}")
  set(config_install_dir share/${PACKAGE_NAME})
  
else()
  
  message(FATAL_ERROR "Not supported system type. Options: UNIX or WIN32.")
  
endif()


######################################
### collect all of header files   ####
######################################

file(GLOB_RECURSE HEADER
  "${PROJECT_SOURCE_DIR}/src/core/*.hpp"
  "${PROJECT_SOURCE_DIR}/src/nn/modules/*.hpp"
  "${PROJECT_SOURCE_DIR}/src/nn/modules/container/*.hpp"
  "${PROJECT_SOURCE_DIR}/src/nn/modules/convolutionLayers/*.hpp"
  "${PROJECT_SOURCE_DIR}/src/nn/modules/simpleLayers/*.hpp"
  "${PROJECT_SOURCE_DIR}/src/nn/modules/tableLayers/*.hpp"
  "${PROJECT_SOURCE_DIR}/src/nn/modules/transfer/*.hpp"
  "${PROJECT_SOURCE_DIR}/src/nn/criterion/*.hpp"
  
  )

file(GLOB_RECURSE HEADER_INIT
  "${PROJECT_SOURCE_DIR}/init.hpp"
  )

message("following header files are installed")
message("${HEADER}")

message("${include_install_dir_full}")

#######################################################
### if you install every codes, uncomment following ###
#######################################################

#install(
#  DIRECTORY ${PROJECT_SOURCE_DIR}
#  DESTINATION tempo
#  )

#######################################################
### following is for putting all header files on    ###
### /usr/local/include/${include install dir full}  ###
#######################################################


INSTALL(FILES ${HEADER_INIT} DESTINATION ${include_install_dir})
INSTALL(FILES ${HEADER} DESTINATION ${include_install_dir_full})


