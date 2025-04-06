cmake_minimum_required(VERSION 3.30)
# Generate compile_commands.json for clangd
set(CMAKE_EXPORT_COMPILE_COMMANDS ON CACHE INTERNAL "")

project(heat-equation VERSION 1.0.0 LANGUAGES C CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Find MPI package with required components
find_package(MPI REQUIRED COMPONENTS C CXX)

# Find OpenMP package
find_package(OpenMP REQUIRED)

# Print information about the MPI installation found
if(MPI_FOUND)
  message(STATUS "Found MPI:")
  message(STATUS "  Version: ${MPI_VERSION}")
  message(STATUS "  C Compiler: ${MPI_C_COMPILER}")
  message(STATUS "  CXX Compiler: ${MPI_CXX_COMPILER}")
  message(STATUS "  Include path: ${MPI_CXX_INCLUDE_DIRS}")
endif()

if(OpenMP_FOUND)
  message(STATUS "Found OpenMP:")
  message(STATUS "  Version: ${OpenMP_VERSION}")
  message(STATUS "  Include path: ${OpenMP_CXX_INCLUDE_DIRS}")
endif()

# Create an executable target
add_executable(a1_mpi src/a1-serial.cpp)

# Link against MPI libraries using the imported target
target_link_libraries(a1_mpi PRIVATE MPI::MPI_CXX)
target_link_libraries(a1_mpi PRIVATE OpenMP::OpenMP_CXX)

# Add include directories if needed for additional headers
target_include_directories(a1_mpi PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_include_directories(a1_mpi PUBLIC ${OpenMP_CXX_INCLUDE_DIRS})

# Installation rules
install(TARGETS a1_mpi
        RUNTIME DESTINATION bin)

# Configuration summary
message(STATUS "Configuration Summary:")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  Install prefix: ${CMAKE_INSTALL_PREFIX}")
message(STATUS "  Testing enabled: ${BUILD_TESTING}")
