cmake_minimum_required(VERSION 3.24)
# Include native GPU code and forward-compatible PTX without model-specific branches.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "75-real;75-virtual")
endif()
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
project(PSISimulation LANGUAGES CXX CUDA)
if(NOT WIN32)
    message(FATAL_ERROR "The viewer requires Windows and an NVIDIA GPU.")
endif()
find_package(CUDAToolkit REQUIRED)
add_executable(psi main.cu hpsi.cuh mspsi.cuh reconstruction.cuh simulation.cuh viewer.h)
target_compile_features(psi PRIVATE cxx_std_17 cuda_std_17)
set_target_properties(psi PROPERTIES CUDA_RUNTIME_LIBRARY Static
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/Releases"
    RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_CURRENT_SOURCE_DIR}/Releases")
target_compile_options(psi PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3>)
target_link_libraries(psi PRIVATE user32 gdi32 opengl32)
