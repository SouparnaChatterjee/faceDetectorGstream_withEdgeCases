cmake_minimum_required(VERSION 3.16)
project(FaceDetection VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

if(OpenCV_FOUND)
    message(STATUS "OpenCV found: ${OpenCV_VERSION}")
    message(STATUS "Include dirs: ${OpenCV_INCLUDE_DIRS}")
else()
    message(FATAL_ERROR "OpenCV not found!")
endif()

add_executable(face_detection src/main.cpp)
target_include_directories(face_detection PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(face_detection PRIVATE ${OpenCV_LIBS})

add_custom_command(TARGET face_detection POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_SOURCE_DIR}/data
        $<TARGET_FILE_DIR:face_detection>/data
    COMMENT "Copying cascade files to build directory"
)

message(STATUS "Project configured successfully!")
