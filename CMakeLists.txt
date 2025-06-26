cmake_minimum_required(VERSION 3.14)
project(rocsolver_benchmark CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)
FetchContent_Declare(
    argparse
    GIT_REPOSITORY https://github.com/p-ranav/argparse.git
)
FetchContent_MakeAvailable(argparse)

set(TARGETS
    bench_rocsolver_dgeqrf_strided_batched
    bench_rocsolver_dgeqrf_batched
    bench_rocsolver_ssyevj_strided_batched
    bench_rocsolver_sgesvdj_strided_batched
    bench_openblas_ssyevj_strided_batched
    bench_openblas_sgesvdj_strided_batched
)

set(CMAKE_CXX_COMPILER /opt/rocm/bin/hipcc)

foreach(TARGET ${TARGETS})
    add_executable(
        ${TARGET}
        ${CMAKE_CURRENT_SOURCE_DIR}/src/${TARGET}.cpp
    )

    target_compile_options(
        ${TARGET} PRIVATE
        -fopenmp
        -Wno-unused-result
    )

    target_include_directories(
        ${TARGET} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        /opt/rocm/include
    )
    
    # Add OpenBLAS include directories for CPU benchmarks
    if(${TARGET} MATCHES "bench_openblas_.*")
        target_include_directories(
            ${TARGET} PRIVATE
            /usr/include/openblas
            /usr/include
        )
    endif()

    target_link_libraries(
        ${TARGET} PRIVATE
        -L/opt/rocm/lib -lrocsolver -lrocblas
        -fopenmp
        argparse
    )
    
    # Add OpenBLAS for CPU benchmarks if needed
    if(${TARGET} MATCHES "bench_openblas_.*")
        target_link_libraries(
            ${TARGET} PRIVATE
            openblas lapacke
        )
    endif()
endforeach(TARGET)
