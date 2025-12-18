# ATenSpace - Tensor-based AtomSpace
cmake_minimum_required(VERSION 3.5)

# Find ATen
find_package(Torch REQUIRED)

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/..)
include_directories(${TORCH_INCLUDE_DIRS})

# Example executable
add_executable(atomspace_example example.cpp)
target_link_libraries(atomspace_example ${TORCH_LIBRARIES})
set_property(TARGET atomspace_example PROPERTY CXX_STANDARD 17)

# Test executable
add_executable(atomspace_test test.cpp)
target_link_libraries(atomspace_test ${TORCH_LIBRARIES})
set_property(TARGET atomspace_test PROPERTY CXX_STANDARD 17)

# Installation
install(FILES 
    Atom.h 
    AtomSpace.h 
    ATenSpace.h
    README.md
    DESTINATION include/ATen/atomspace
)
