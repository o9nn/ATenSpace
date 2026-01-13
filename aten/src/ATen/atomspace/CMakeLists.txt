# ATenSpace - Tensor-based AtomSpace
cmake_minimum_required(VERSION 3.5)

# Find ATen
find_package(Torch REQUIRED)

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${TORCH_INCLUDE_DIRS})

# Python bindings option
option(BUILD_PYTHON_BINDINGS "Build Python bindings" OFF)

if(BUILD_PYTHON_BINDINGS)
    # Find Python and pybind11
    find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
    find_package(pybind11 CONFIG)
    
    if(NOT pybind11_FOUND)
        # Try to use pip installed pybind11
        execute_process(
            COMMAND ${Python3_EXECUTABLE} -c "import pybind11; print(pybind11.get_cmake_dir())"
            OUTPUT_VARIABLE pybind11_DIR
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
        )
        if(pybind11_DIR)
            find_package(pybind11 CONFIG PATHS ${pybind11_DIR})
        endif()
    endif()
    
    if(pybind11_FOUND)
        message(STATUS "Building Python bindings with pybind11")
        pybind11_add_module(atenspace python_bindings.cpp)
        target_link_libraries(atenspace PRIVATE ${TORCH_LIBRARIES})
        set_property(TARGET atenspace PROPERTY CXX_STANDARD 17)
        
        # Install Python module
        install(TARGETS atenspace LIBRARY DESTINATION .)
    else()
        message(WARNING "pybind11 not found. Skipping Python bindings. Install with: pip install pybind11")
    endif()
endif()

# Example executable
add_executable(atomspace_example example.cpp)
target_link_libraries(atomspace_example ${TORCH_LIBRARIES})
set_property(TARGET atomspace_example PROPERTY CXX_STANDARD 17)

# Advanced example executable
add_executable(atomspace_example_advanced example_advanced.cpp)
target_link_libraries(atomspace_example_advanced ${TORCH_LIBRARIES})
set_property(TARGET atomspace_example_advanced PROPERTY CXX_STANDARD 17)

# PLN example executable
add_executable(atomspace_example_pln example_pln.cpp)
target_link_libraries(atomspace_example_pln ${TORCH_LIBRARIES})
set_property(TARGET atomspace_example_pln PROPERTY CXX_STANDARD 17)

# ECAN example executable
add_executable(atomspace_example_ecan example_ecan.cpp)
target_link_libraries(atomspace_example_ecan ${TORCH_LIBRARIES})
set_property(TARGET atomspace_example_ecan PROPERTY CXX_STANDARD 17)

# Cognitive Engine example executable
add_executable(atomspace_example_cognitive example_cognitive.cpp)
target_link_libraries(atomspace_example_cognitive ${TORCH_LIBRARIES})
set_property(TARGET atomspace_example_cognitive PROPERTY CXX_STANDARD 17)

# NLU example executable
add_executable(atomspace_example_nlu example_nlu.cpp)
target_link_libraries(atomspace_example_nlu ${TORCH_LIBRARIES})
set_property(TARGET atomspace_example_nlu PROPERTY CXX_STANDARD 17)

# Vision example executable
add_executable(atomspace_example_vision example_vision.cpp)
target_link_libraries(atomspace_example_vision ${TORCH_LIBRARIES})
set_property(TARGET atomspace_example_vision PROPERTY CXX_STANDARD 17)

# Neural Network (ATenNN) example executable
add_executable(atomspace_example_nn example_nn.cpp)
target_link_libraries(atomspace_example_nn ${TORCH_LIBRARIES})
set_property(TARGET atomspace_example_nn PROPERTY CXX_STANDARD 17)

# Test executable
add_executable(atomspace_test test.cpp)
target_link_libraries(atomspace_test ${TORCH_LIBRARIES})
set_property(TARGET atomspace_test PROPERTY CXX_STANDARD 17)

# Advanced features test executable
add_executable(atomspace_test_advanced test_advanced.cpp)
target_link_libraries(atomspace_test_advanced ${TORCH_LIBRARIES})
set_property(TARGET atomspace_test_advanced PROPERTY CXX_STANDARD 17)

# PLN test executable
add_executable(atomspace_test_pln test_pln.cpp)
target_link_libraries(atomspace_test_pln ${TORCH_LIBRARIES})
set_property(TARGET atomspace_test_pln PROPERTY CXX_STANDARD 17)

# ECAN test executable
add_executable(atomspace_test_ecan test_ecan.cpp)
target_link_libraries(atomspace_test_ecan ${TORCH_LIBRARIES})
set_property(TARGET atomspace_test_ecan PROPERTY CXX_STANDARD 17)

# Cognitive Engine test executable
add_executable(atomspace_test_cognitive test_cognitive.cpp)
target_link_libraries(atomspace_test_cognitive ${TORCH_LIBRARIES})
set_property(TARGET atomspace_test_cognitive PROPERTY CXX_STANDARD 17)

# NLU test executable
add_executable(atomspace_test_nlu test_nlu.cpp)
target_link_libraries(atomspace_test_nlu ${TORCH_LIBRARIES})
set_property(TARGET atomspace_test_nlu PROPERTY CXX_STANDARD 17)

# Vision test executable
add_executable(atomspace_test_vision test_vision.cpp)
target_link_libraries(atomspace_test_vision ${TORCH_LIBRARIES})
set_property(TARGET atomspace_test_vision PROPERTY CXX_STANDARD 17)

# Neural Network test executable
add_executable(atomspace_test_nn test_nn.cpp)
target_link_libraries(atomspace_test_nn ${TORCH_LIBRARIES})
set_property(TARGET atomspace_test_nn PROPERTY CXX_STANDARD 17)

# Installation
install(FILES 
    Atom.h 
    AtomSpace.h 
    ATenSpace.h
    TimeServer.h
    AttentionBank.h
    Serializer.h
    PatternMatcher.h
    TruthValue.h
    ForwardChainer.h
    BackwardChainer.h
    ECAN.h
    TensorLogicEngine.h
    CognitiveEngine.h
    NLU.h
    Vision.h
    ATenNN.h
    PretrainedModels.h
    README.md
    DESTINATION include/ATen/atomspace
)
