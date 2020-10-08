cmake_minimum_required(VERSION 2.8)
project(Spectral_Stokes_IBMethod CXX)

# lib and bin will be in top source
set(CMAKE_INSTALL_PREFIX "${CMAKE_SOURCE_DIR}")
# set runtime path for shared libraries
SET(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS "7.5.0")
    message(FATAL_ERROR "Insufficient gcc version. Must be 7.5.0 or greater")
  endif()
endif()

# make sure we can link to openmp
find_package(OpenMP REQUIRED)

# TODO add find LAPACKE

# add support for FFTW to find_package
configure_file(cmake.in/download_FindFFTW.cmake.in findFFTW-download/CMakeLists.txt)
execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
        RESULT_VARIABLE result
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/findFFTW-download )
if(result)
    message(FATAL_ERROR "CMake step for findFFTW failed: ${result}")
    else()
    message("CMake step for findFFTW completed (${result}).")
endif()
execute_process(COMMAND ${CMAKE_COMMAND} --build .
        RESULT_VARIABLE result
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/findFFTW-download )
if(result)
    message(FATAL_ERROR "Build step for findFFTW failed: ${result}")
endif()

set(findFFTW_DIR ${CMAKE_CURRENT_BINARY_DIR}/findFFTW-src)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${findFFTW_DIR}")

# now find fftw
find_package(FFTW REQUIRED)


# general compiler flags
option(debug "Build with debugger" ON)
# disabling testing for now
#option(test "Build all tests" ON)


# setting include dir and general compile flags
if(NOT debug)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Iinclude -march=native")
elseif(debug)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Iinclude -g -DDEBUG -march=native")
endif()

# include dirs
include_directories(include)
file(GLOB h_FILES include/*.h)
set(spreadInterp_INC_FILES ${h_FILES})
install(FILES ${spreadInterp_INC_FILES} DESTINATION include)

# setting lib src files

set(chebSRC src/Quadrature.cpp)
set(gridSRC src/Grid.cpp wrapper/GridWrapper.cpp)
set(particlesSRC src/ParticleList.cpp wrapper/ParticleListWrapper.cpp)
set(spreadInterpSRC src/SpreadInterp.cpp wrapper/SpreadInterpWrapper.cpp)
set(transformSRC src/Transform.cpp wrapper/TransformWrapper.cpp)
set(spreadInterpTPTestSRC testing/test_spread_TP.cpp)
set(linSolveSRC src/LinearSolvers.cpp)
set(spreadInterpDPTestSRC testing/test_spread_DP.cpp)
set(chebTestSRC testing/test_cheb.cpp)
set(transformTestSRC testing/test_transform_TP.cpp)

# set build output paths
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# building lib
add_library(cheb SHARED ${chebSRC})
set_source_files_properties(${chebSRC} PROPERTIES COMPILE_FLAGS "-O3 -fopt-info -fPIC -fopenmp")
target_link_libraries(cheb gomp)

add_library(grid SHARED ${gridSRC})
set_source_files_properties(${gridSRC} PROPERTIES COMPILE_FLAGS "-O3 -fPIC")
target_link_libraries(grid cheb fftw3)

add_library(particles SHARED ${particlesSRC})
set_source_files_properties(${particlesSRC} PROPERTIES COMPILE_FLAGS "-O3 -fopt-info -fPIC -fopenmp")
target_link_libraries(particles grid cheb)

add_library(spreadInterp SHARED ${spreadInterpSRC})
set_source_files_properties(${spreadInterpSRC} PROPERTIES COMPILE_FLAGS "-O3 -fopt-info -fPIC -fopenmp")
target_link_libraries(spreadInterp particles)

add_library(transform SHARED ${transformSRC})
set_source_files_properties(${transformSRC} PROPERTIES COMPILE_FLAGS "-lfftw3 -lm -fPIC -fopenmp")
target_link_libraries(transform fftw3 fftw3_omp gomp)

add_library(linSolve SHARED ${linSolveSRC})
set_source_files_properties(${linSolveSRC} PROPERTIES COMPILE_FLAGS "-lm -llapacke -lblas -fopenmp -fPIC")

# install libs
install(TARGETS spreadInterp ARCHIVE DESTINATION lib LIBRARY DESTINATION lib)
install(TARGETS cheb ARCHIVE DESTINATION lib LIBRARY DESTINATION lib)
install(TARGETS grid ARCHIVE DESTINATION lib LIBRARY DESTINATION lib)
install(TARGETS particles ARCHIVE DESTINATION lib LIBRARY DESTINATION lib)
install(TARGETS transform ARCHIVE DESTINATION lib LIBRARY DESTINATION lib)
install(TARGETS linSolve ARCHIVE DESTINATION lib LIBRARY DESTINATION lib)
## build executables ##

  
# building test
add_executable(test_spread_TP ${spreadInterpTPTestSRC})
set_source_files_properties(${spreadInterpTPTestSRC} PROPERTIES COMPILE_FLAGS "-fopenmp")
target_link_libraries(test_spread_TP spreadInterp fftw3_omp)

add_executable(test_spread_DP ${spreadInterpDPTestSRC})
set_source_files_properties(${spreadInterpDPTestSRC} PROPERTIES COMPILE_FLAGS "-fopenmp")
target_link_libraries(test_spread_DP spreadInterp fftw3_omp)


add_executable(test_cheb ${chebTestSRC})
set_source_files_properties(${chebTestSRC} PROPERTIES COMPILE_FLAGS "-fopenmp")
target_link_libraries(test_cheb cheb)

add_executable(test_transform_TP ${transformTestSRC})
target_link_libraries(test_transform_TP transform spreadInterp)

# install exec for test data creation
install(TARGETS test_spread_TP RUNTIME DESTINATION bin/testing)
install(TARGETS test_spread_DP RUNTIME DESTINATION bin/testing)
install(TARGETS test_cheb RUNTIME DESTINATION bin/testing)
install(TARGETS test_transform_TP RUNTIME DESTINATION bin/testing)

# disabling testing for now
#if (test)
#  # Download and unpack googletest at configure time
#  configure_file(cmake.in/download_googltest.cmake.in googletest-download/CMakeLists.txt)
#  execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
#    RESULT_VARIABLE result
#    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/googletest-download )
#  if(result)
#    message(FATAL_ERROR "CMake step for googletest failed: ${result}")
#  endif()
#  execute_process(COMMAND ${CMAKE_COMMAND} --build .
#    RESULT_VARIABLE result
#    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/googletest-download )
#  if(result)
#    message(FATAL_ERROR "Build step for googletest failed: ${result}")
#  endif()
#  
#  # Prevent overriding the parent project's compiler/linker
#  # settings on Windows
#  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
#  
#  # Add googletest directly to our build. This defines
#  # the gtest and gtest_main targets.
#  add_subdirectory(${CMAKE_CURRENT_BINARY_DIR}/googletest-src
#                   ${CMAKE_CURRENT_BINARY_DIR}/googletest-build
#                   EXCLUDE_FROM_ALL)
#  
#  # The gtest/gtest_main targets carry header search path
#  # dependencies automatically when using CMake 2.8.11 or
#  # later. Otherwise we have to add them here ourselves.
#  if (CMAKE_VERSION VERSION_LESS 2.8.11)
#    include_directories(${gtest_SOURCE_DIR}/include)
#  endif()
#
#  enable_testing()
#  
#  set(OLD_RUNTIME_OUTPUT_DIR ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
#  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/testing)
#  add_executable(testSpreadInterpNoPBC ${CMAKE_SOURCE_DIR}/testing/spreadInterpNoPBC.cpp)
#  add_executable(testSpreadInterpPBC ${CMAKE_SOURCE_DIR}/testing/spreadInterpPBC.cpp)
#  target_link_libraries(testSpreadInterpNoPBC spreadInterp io init gomp gtest_main)
#  target_link_libraries(testSpreadInterpPBC spreadInterp io init gomp gtest_main)
#  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${OLD_RUNTIME_OUTPUT_DIR})
#  add_subdirectory(testing)
#  install(TARGETS testSpreadInterpNoPBC RUNTIME DESTINATION ${CMAKE_SOURCE_DIR}/testing)
#  install(TARGETS testSpreadInterpPBC RUNTIME DESTINATION ${CMAKE_SOURCE_DIR}/testing)
#endif()
