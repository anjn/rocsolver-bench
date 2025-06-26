#include <stdio.h>   // for printf
#include <stdlib.h> // for malloc
#include <random> // for random number generation
#include <vector> // for storing timing results
#include <cmath> // for sqrt in standard deviation calculation
#include <iostream> // for cout/cerr
#include <chrono> // for high-resolution timing
#include <cstring> // for memcpy
#include <cblas.h> // for OpenBLAS
#include <lapacke.h> // for LAPACKE

#include <argparse/argparse.hpp>

// Example: Compute the eigenvalues and eigenvectors of an array of symmetric matrices on the CPU using OpenBLAS

float *create_matrices(lapack_int N,
                      lapack_int lda,
                      size_t strideA,
                      lapack_int batch_count,
                      int random_seed) {
  // allocate space for input matrix data on CPU
  float *hA = (float*)malloc(sizeof(float) * strideA * batch_count);

  // generate random symmetric matrices
  std::mt19937 gen(random_seed);
  std::uniform_real_distribution<float> dis(-10.0, 10.0);
  
  for (lapack_int b = 0; b < batch_count; ++b) {
    for (lapack_int i = 0; i < N; ++i) {
      // Diagonal elements
      hA[i + i * lda + b * strideA] = dis(gen) * 10.0; // Make diagonal dominant
      
      // Off-diagonal elements (ensure symmetry)
      for (lapack_int j = i + 1; j < N; ++j) {
        float value = dis(gen);
        hA[i + j * lda + b * strideA] = value;
        hA[j + i * lda + b * strideA] = value; // Symmetric counterpart
      }
    }
  }

  return hA;
}

// Use LAPACKE_ssyev to compute eigenvalues and eigenvectors of an array of real symmetric matrices.
int main(int argc, char *argv[]) {
  // ArgumentParserの設定
  argparse::ArgumentParser program("bench_openblas_ssyev");
  
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
  
  // 引数の解析
  try {
    program.parse_args(argc, argv);
  } catch (const std::exception& err) {
    std::cerr << err.what() << std::endl;
    std::cerr << program;
    return 1;
  }
  
  // 値の取得
  lapack_int N = program.get<int>("--size");
  lapack_int lda = program.get<int>("--lda");
  lapack_int batch_count = program.get<int>("--batch-count");
  int random_seed = program.get<int>("--random-seed");
  int iterations = program.get<int>("--iterations");
  int warmup_time = program.get<int>("--warmup-time");

  if (lda < N) lda = N;
  
  // ストライドの計算（指定されていない場合はlda * Nを使用）
  size_t strideA;
  if (program.present("--stride")) {
    strideA = program.get<int>("--stride");
  } else {
    strideA = lda * N;
  }
  
  float *hA = create_matrices(N, lda, strideA, batch_count, random_seed);

  // calculate the sizes of our arrays
  size_t size_A = strideA * (size_t)batch_count;   // elements in array for matrices
  size_t strideW = N;                              // stride of eigenvalues
  size_t size_W = strideW * (size_t)batch_count;   // elements in array for eigenvalues

  // allocate memory for eigenvalues
  float *hW = (float*)malloc(sizeof(float) * size_W);
  
  // Create a copy of the original matrices for each iteration
  float *hA_copy = (float*)malloc(sizeof(float) * size_A);
  
  // Query the optimal workspace size
  lapack_int lwork = -1;  // Signal to query optimal size
  float work_query;
  lapack_int info = LAPACKE_ssyev_work(LAPACK_COL_MAJOR, 'V', 'U', 
                                      N, NULL, lda, NULL, &work_query, lwork);
  
  // Get the optimal workspace size
  lwork = (lapack_int)work_query;
  
  // vector to store timing results
  std::vector<float> timings;
  
  // time-based warm-up phase
  printf("Performing warm-up for %d ms...\n", warmup_time);

  auto warmup_start = std::chrono::high_resolution_clock::now();
  auto warmup_current = warmup_start;
  float warmup_elapsed = 0.0f;
  int warmup_count = 0;

  while (warmup_elapsed < warmup_time || warmup_count == 0) {
    // Copy the original matrices for this warm-up iteration
    memcpy(hA_copy, hA, sizeof(float) * size_A);
    
    // Process each matrix in the batch
    #pragma omp parallel
    {
      // Allocate thread-local workspace
      float *thread_work = (float*)malloc(sizeof(float) * lwork);
      
      #pragma omp for
      for (lapack_int b = 0; b < batch_count; ++b) {
        float* A_batch = hA_copy + b * strideA;
        float* W_batch = hW + b * strideW;
        
        // Compute eigenvalues and eigenvectors using _work variant with thread-local workspace
        // LAPACKE_ssyev_work parameters:
        // - matrix_layout: LAPACK_COL_MAJOR for column-major layout
        // - jobz: 'V' to compute both eigenvalues and eigenvectors
        // - uplo: 'U' to use upper triangular part of the matrix
        // - n: matrix dimension
        // - a: input/output matrix
        // - lda: leading dimension of a
        // - w: output eigenvalues
        // - work: workspace array (thread-local)
        // - lwork: size of workspace
        lapack_int info = LAPACKE_ssyev_work(LAPACK_COL_MAJOR, 'V', 'U', 
                                            N, A_batch, lda, W_batch, thread_work, lwork);
      
        if (info != 0) {
          printf("LAPACKE_ssyev failed for matrix %d with error %d\n", (int)b, (int)info);
        }
      }
      
      // Free thread-local workspace
      free(thread_work);
    }
    
    warmup_count++;
    
    // check elapsed time
    warmup_current = std::chrono::high_resolution_clock::now();
    warmup_elapsed = std::chrono::duration<float, std::milli>(warmup_current - warmup_start).count();
  }

  printf("Completed %d warm-up iterations in %.2f ms\n", warmup_count, warmup_elapsed);
  
  // run the computation multiple times for timing
  for (int iter = 0; iter < iterations; ++iter) {
    // Copy the original matrices for this iteration
    memcpy(hA_copy, hA, sizeof(float) * size_A);
    
    // start timing
    auto start = std::chrono::high_resolution_clock::now();
    
    // Process each matrix in the batch
    #pragma omp parallel
    {
      // Allocate thread-local workspace
      float *thread_work = (float*)malloc(sizeof(float) * lwork);
      
      #pragma omp for
      for (lapack_int b = 0; b < batch_count; ++b) {
        float* A_batch = hA_copy + b * strideA;
        float* W_batch = hW + b * strideW;
        
        // Compute eigenvalues and eigenvectors using _work variant with thread-local workspace
        lapack_int info = LAPACKE_ssyev_work(LAPACK_COL_MAJOR, 'V', 'U', 
                                            N, A_batch, lda, W_batch, thread_work, lwork);
      
        //if (info != 0) {
        //  printf("LAPACKE_ssyev failed for matrix %d with error %d\n", (int)b, (int)info);
        //}
      }
      
      // Free thread-local workspace
      free(thread_work);
    }
    
    // stop timing
    auto stop = std::chrono::high_resolution_clock::now();
    
    // calculate elapsed time
    float elapsed_time = std::chrono::duration<float, std::milli>(stop - start).count();
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
  printf("\n===== Performance Results (CPU - OpenBLAS) =====\n");
  printf("Matrix size: %d x %d\n", (int)N, (int)N);
  printf("Batch count: %d\n", (int)batch_count);
  printf("Warm-up time: %d ms (completed %d iterations)\n", warmup_time, warmup_count);
  printf("Timing iterations: %d\n", iterations);
  printf("Average execution time: %.3f ms\n", avg_time);
  printf("Standard deviation: %.3f ms\n", std_dev);
  printf("==============================================\n\n");

  // clean up
  free(hA);
  free(hA_copy);
  free(hW);
  
  return 0;
}
