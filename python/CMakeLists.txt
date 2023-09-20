
# Python module
if(TRITON_BUILD_PYTHON_MODULE)
    message(STATUS "Adding VK Backend Python module")
    set(PYTHON_SRC_PATH ${CMAKE_CURRENT_SOURCE_DIR}/csrc)
    set(PYTHON_SRC ${PYTHON_SRC_PATH}/main.cc ${PYTHON_SRC_PATH}/vk_backend_for_triton.cc)
    include_directories("." ${PYTHON_SRC_PATH})
    include_directories(../include)

    if(PYTHON_INCLUDE_DIRS)
        include_directories(${PYTHON_INCLUDE_DIRS})
    else()
        find_package(Python3 REQUIRED COMPONENTS Development Interpreter)
        include_directories(${Python3_INCLUDE_DIRS})
        link_directories(${Python3_LIBRARY_DIRS})
        link_libraries(${Python3_LIBRARIES})
        add_link_options(${Python3_LINK_OPTIONS})
    endif()

    add_library(vk_backend_for_triton SHARED ${PYTHON_SRC})

    get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
    get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

    set(VK_EXTENSION_LIBRARIES
            TritonAnalysis
            TritonTransforms
            TritonGPUTransforms
            TritonSPIRV
            TritonGPUToSPIRV
            ${dialect_libs}
            ${conversion_libs}

            # optimizations
            MLIRPass
            MLIRTransforms
            MLIRSupport
            MLIRIR
            ${dialect_libs}
            ${conversion_libs}
            )
    target_link_libraries(vk_backend_for_triton PRIVATE ${VK_EXTENSION_LIBRARIES})
    target_link_libraries(vk_backend_for_triton PRIVATE ${LLVM_LIBRARIES})

    link_libraries(stdc++fs)

    target_link_options(vk_backend_for_triton PRIVATE ${LLVM_LDFLAGS})

    file(INSTALL
         ${CMAKE_CURRENT_SOURCE_DIR}/utils/sycl.cpp
         ${CMAKE_CURRENT_SOURCE_DIR}/utils/pipeline.cpp
         DESTINATION ${PYTHON_THIRD_PARTY_PATH}/cpu/utils)

    file(INSTALL
         ${CMAKE_CURRENT_SOURCE_DIR}/__init__.py

         DESTINATION ${PYTHON_THIRD_PARTY_PATH}/cpu/)
endif()