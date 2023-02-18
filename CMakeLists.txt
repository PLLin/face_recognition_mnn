cmake_minimum_required(VERSION 3.0)

set(CMAKE_C_STANDARD 99)
set(CMAKE_CXX_STANDARD 11)
set(MNN_PATH "/opt/MNN")

if(CMAKE_SYSTEM_NAME MATCHES "^Linux")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D__STRICT_ANSI__")
    if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
      # This is to workaround libgcc.a
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++")
    endif()
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "^armv7")
        add_definitions(-mfpu=neon)    #please define in project/cross-compile/arm.toolchain.cmake
    endif()
endif()

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS}
                    ${MNN_PATH}/include/
                    ${MNN_PATH}/source/
                    ${MNN_PATH}/express/
                    ${MNN_PATH}/tools/
                    ${MNN_PATH}/codegen/
                    ${MNN_PATH}/schema/current/
                    ${MNN_PATH}/3rd_party/
                    ${MNN_PATH}/3rd_party/flatbuffers/include
                    ${MNN_PATH}/3rd_party/half
                    ${MNN_PATH}/3rd_party/imageHelper
                    ${MNN_PATH}/3rd_party/OpenCLHeaders/
                    ${CMAKE_CURRENT_LIST_DIR}/include/)

aux_source_directory(${CMAKE_CURRENT_LIST_DIR}/src/ DIR_SRCS)

link_directories(${MNN_PATH}/build)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=gnu99")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

add_executable(fr_trial_run ${CMAKE_CURRENT_LIST_DIR}/fr_trial_run.cpp)
target_link_libraries(fr_trial_run MNN)

add_executable(fr_main ${CMAKE_CURRENT_LIST_DIR}/fr_main.cpp ${DIR_SRCS})
target_link_libraries(fr_main MNN ${OpenCV_LIBS})