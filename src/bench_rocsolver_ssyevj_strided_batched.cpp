#include <hip/hip_runtime_api.h> // for hip functions
#include <rocsolver/rocsolver.h> // for all the rocsolver C interfaces and type declarations
#include <stdio.h>   // for printf
#include <stdlib.h> // for malloc
#include <random> // for random number generation
#include <vector> // for storing timing results
#include <cmath> // for sqrt in standard deviation calculation
#include <iostream> // for cout/cerr

#include <argparse/argparse.hpp>

// Example: Compute the eigenvalues and eigenvectors of an array of symmetric matrices on the GPU

float *create_symmetric_matrices(rocblas_int *n_out,
                               rocblas_int *lda_out,
                               rocblas_stride *strideA_out,
                               rocblas_int *batch_count_out,
                               int random_seed) {
  // allocate space for input matrix data on CPU
  float *hA = (float*)malloc(sizeof(float) * (*strideA_out) * (*batch_count_out));

  // generate random symmetric matrices
  std::mt19937 gen(random_seed);
  std::uniform_real_distribution<float> dis(-10.0, 10.0);
  
  for (rocblas_int b = 0; b < *batch_count_out; ++b) {
    for (rocblas_int i = 0; i < *n_out; ++i) {
      // Diagonal elements
      hA[i + i * (*lda_out) + b * (*strideA_out)] = dis(gen) * 10.0; // Make diagonal dominant
      
      // Off-diagonal elements (ensure symmetry)
      for (rocblas_int j = i + 1; j < *n_out; ++j) {
        float value = dis(gen);
        hA[i + j * (*lda_out) + b * (*strideA_out)] = value;
        hA[j + i * (*lda_out) + b * (*strideA_out)] = value; // Symmetric counterpart
      }
    }
  }

  return hA;
}

// Use rocsolver_ssyevj_strided_batched to compute eigenvalues and eigenvectors of an array of real symmetric matrices.
int main(int argc, char *argv[]) {
  // ArgumentParserの設定
  argparse::ArgumentParser program("bench_rocsolver_ssyevj_strided_batched");
  
  program.add_argument("-n", "--size")
      .help("Matrix size (N x N)")
      .default_value(10)
      .scan<'i', int>();
      
  program.add_argument("-l", "--lda")
      .help("Leading dimension (lda)")
      .default_value(10)
      .scan<'i', int>();
      
  program.add_argument("-s", "--stride")
      .help("Stride between matrices (default: lda * N)")
      .scan<'i', int>();
      
  program.add_argument("-b", "--batch-count")
      .help("Batch count")
      .default_value(2)
      .scan<'i', int>();
      
  program.add_argument("-r", "--random-seed")
      .help("Random seed for matrix generation")
      .default_value(42)
      .scan<'i', int>();
      
  program.add_argument("-i", "--iterations")
      .help("Number of iterations for timing")
      .default_value(10)
      .scan<'i', int>();
      
  program.add_argument("-w", "--warmup-time")
      .help("Warm-up time in milliseconds before timing")
      .default_value(1000)
      .scan<'i', int>();
      
  program.add_argument("-t", "--tolerance")
      .help("Tolerance for Jacobi method")
      .default_value(1e-7f)
      .scan<'f', float>();
      
  program.add_argument("-m", "--max-sweeps")
      .help("Maximum number of sweeps for Jacobi method")
      .default_value(100)
      .scan<'i', int>();
  
  // 引数の解析
  try {
    program.parse_args(argc, argv);
  } catch (const std::exception& err) {
    std::cerr << err.what() << std::endl;
    std::cerr << program;
    return 1;
  }
  
  // 値の取得
  rocblas_int N = program.get<int>("--size");
  rocblas_int lda = program.get<int>("--lda");
  rocblas_int batch_count = program.get<int>("--batch-count");
  int random_seed = program.get<int>("--random-seed");
  int iterations = program.get<int>("--iterations");
  int warmup_time = program.get<int>("--warmup-time");
  float tolerance = program.get<float>("--tolerance");
  rocblas_int max_sweeps = program.get<int>("--max-sweeps");

  if (lda < N) lda = N;
  
  // ストライドの計算（指定されていない場合はlda * Nを使用）
  rocblas_stride strideA;
  if (program.present("--stride")) {
    strideA = program.get<int>("--stride");
  } else {
    strideA = lda * N;
  }
  
  // create_symmetric_matrices関数の呼び出し
  float *hA = create_symmetric_matrices(&N, &lda, &strideA, &batch_count, random_seed);

  // initialization
  rocblas_handle handle;
  rocblas_create_handle(&handle);

  // calculate the sizes of our arrays
  size_t size_A = strideA * (size_t)batch_count;   // elements in array for matrices
  rocblas_stride strideW = N;                      // stride of eigenvalues
  size_t size_W = strideW * (size_t)batch_count;   // elements in array for eigenvalues
  size_t size_info = batch_count;                  // elements in info array

  // allocate memory on GPU
  float *dA, *dW, *dResidual;
  rocblas_int *dInfo, *dNSweeps;
  hipMalloc((void**)&dA, sizeof(float)*size_A);
  hipMalloc((void**)&dW, sizeof(float)*size_W);
  hipMalloc((void**)&dInfo, sizeof(rocblas_int)*size_info);
  hipMalloc((void**)&dResidual, sizeof(float)*batch_count);
  hipMalloc((void**)&dNSweeps, sizeof(rocblas_int)*batch_count);

  // copy data to GPU
  hipMemcpy(dA, hA, sizeof(float)*size_A, hipMemcpyHostToDevice);

  // create events for timing
  hipEvent_t start, stop;
  hipEventCreate(&start);
  hipEventCreate(&stop);
  
  // vector to store timing results
  std::vector<float> timings;
  
  // time-based warm-up phase
  printf("Performing warm-up for %d ms...\n", warmup_time);

  hipEvent_t warmup_start, warmup_current;
  hipEventCreate(&warmup_start);
  hipEventCreate(&warmup_current);
  hipEventRecord(warmup_start, 0);

  float warmup_elapsed = 0.0f;
  int warmup_count = 0;

  // Set up parameters for syevj
  rocblas_esort esort = rocblas_esort_ascending; // Sort eigenvalues in ascending order
  rocblas_evect evect = rocblas_evect_original;  // Compute both eigenvalues and eigenvectors
  rocblas_fill uplo = rocblas_fill_upper;        // Use upper triangular part of the matrix

  while (warmup_elapsed < warmup_time || warmup_count == 0) {
    // run the computation (without timing)
    rocsolver_ssyevj_strided_batched(handle, esort, evect, uplo, N, dA, lda, strideA, 
                                    tolerance, dResidual, max_sweeps, dNSweeps, 
                                    dW, strideW, dInfo, batch_count);
    
    warmup_count++;
    
    // check elapsed time
    hipEventRecord(warmup_current, 0);
    hipEventSynchronize(warmup_current);
    hipEventElapsedTime(&warmup_elapsed, warmup_start, warmup_current);
  }

  printf("Completed %d warm-up iterations in %.2f ms\n", warmup_count, warmup_elapsed);
  hipEventDestroy(warmup_start);
  hipEventDestroy(warmup_current);
  
  // run the computation multiple times for timing
  for (int iter = 0; iter < iterations; ++iter) {
    // start timing
    hipEventRecord(start, 0);
    
    // compute the eigenvalues and eigenvectors on the GPU
    rocsolver_ssyevj_strided_batched(handle, esort, evect, uplo, N, dA, lda, strideA, 
                                    tolerance, dResidual, max_sweeps, dNSweeps, 
                                    dW, strideW, dInfo, batch_count);
    
    // stop timing
    hipEventRecord(stop, 0);
    hipEventSynchronize(stop);
    
    // calculate elapsed time
    float elapsed_time;
    hipEventElapsedTime(&elapsed_time, start, stop);
    timings.push_back(elapsed_time);
  }
  
  // calculate statistics
  float avg_time = 0.0f;
  for (float t : timings) avg_time += t;
  avg_time /= timings.size();
  
  float std_dev = 0.0f;
  for (float t : timings) std_dev += (t - avg_time) * (t - avg_time);
  std_dev = sqrt(std_dev / timings.size());
  
  // print timing results
  printf("\n===== Performance Results =====\n");
  printf("Matrix size: %d x %d\n", N, N);
  printf("Batch count: %d\n", batch_count);
  printf("Tolerance: %e\n", tolerance);
  printf("Max sweeps: %d\n", max_sweeps);
  printf("Warm-up time: %d ms (completed %d iterations)\n", warmup_time, warmup_count);
  printf("Timing iterations: %d\n", iterations);
  printf("Average execution time: %.3f ms\n", avg_time);
  printf("Standard deviation: %.3f ms\n", std_dev);
  printf("==============================\n\n");

  // clean up
  hipFree(dA);
  hipFree(dW);
  hipFree(dInfo);
  hipFree(dResidual);
  hipFree(dNSweeps);
  free(hA);
  hipEventDestroy(start);
  hipEventDestroy(stop);
  rocblas_destroy_handle(handle);
}
