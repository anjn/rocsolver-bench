#include <hip/hip_runtime_api.h> // for hip functions
#include <rocsolver/rocsolver.h> // for all the rocsolver C interfaces and type declarations
#include <stdio.h>   // for printf
#include <stdlib.h> // for malloc
#include <random> // for random number generation
#include <vector> // for storing timing results
#include <cmath> // for sqrt in standard deviation calculation
#include <iostream> // for cout/cerr

#include <argparse/argparse.hpp>

// Example: Compute the QR Factorizations of an array of matrices on the GPU

double *create_example_matrices(rocblas_int *M_out,
                                rocblas_int *N_out,
                                rocblas_int *lda_out,
                                rocblas_stride *strideA_out,
                                rocblas_int *batch_count_out,
                                int random_seed) {
  // allocate space for input matrix data on CPU
  double *hA = (double*)malloc(sizeof(double) * (*strideA_out) * (*batch_count_out));

  // generate random matrices
  std::mt19937 gen(random_seed);
  std::uniform_real_distribution<double> dis(-100.0, 100.0);
  
  for (rocblas_int b = 0; b < *batch_count_out; ++b) {
    for (rocblas_int i = 0; i < *M_out; ++i) {
      for (rocblas_int j = 0; j < *N_out; ++j) {
        hA[i + j * (*lda_out) + b * (*strideA_out)] = dis(gen);
      }
    }
  }

  return hA;
}

// Use rocsolver_dgeqrf_strided_batched to factor an array of real M-by-N matrices.
int main(int argc, char *argv[]) {
  // ArgumentParserの設定
  argparse::ArgumentParser program("bench_rocsolver_dgeqrf_strided_batched");
  
  program.add_argument("-m", "--rows")
      .help("Number of rows (M)")
      .default_value(10)
      .scan<'i', int>();
      
  program.add_argument("-n", "--cols")
      .help("Number of columns (N)")
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
  rocblas_int M = program.get<int>("--rows");
  rocblas_int N = program.get<int>("--cols");
  rocblas_int lda = program.get<int>("--lda");
  rocblas_int batch_count = program.get<int>("--batch-count");
  int random_seed = program.get<int>("--random-seed");
  int iterations = program.get<int>("--iterations");
  int warmup_time = program.get<int>("--warmup-time");

  if (lda < M) lda = M;
  
  // ストライドの計算（指定されていない場合はlda * Nを使用）
  rocblas_stride strideA;
  if (program.present("--stride")) {
    strideA = program.get<int>("--stride");
  } else {
    strideA = lda * N;
  }
  
  // create_example_matrices関数の呼び出し
  double *hA = create_example_matrices(&M, &N, &lda, &strideA, &batch_count, random_seed);


  // initialization
  rocblas_handle handle;
  rocblas_create_handle(&handle);

  // preload rocBLAS GEMM kernels (optional)
  // rocblas_initialize();

  // calculate the sizes of our arrays
  size_t size_A = strideA * (size_t)batch_count;   // elements in array for matrices
  rocblas_stride strideP = (M < N) ? M : N;        // stride of Householder scalar sets
  size_t size_piv = strideP * (size_t)batch_count; // elements in array for Householder scalars

  // allocate memory on GPU
  double *dA, *dIpiv;
  hipMalloc((void**)&dA, sizeof(double)*size_A);
  hipMalloc((void**)&dIpiv, sizeof(double)*size_piv);

  // copy data to GPU
  hipMemcpy(dA, hA, sizeof(double)*size_A, hipMemcpyHostToDevice);

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

  while (warmup_elapsed < warmup_time || warmup_count == 0) {
    // run the computation (without timing)
    rocsolver_dgeqrf_strided_batched(handle, M, N, dA, lda, strideA, dIpiv, strideP, batch_count);
    
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
    
    // compute the QR factorizations on the GPU
    rocsolver_dgeqrf_strided_batched(handle, M, N, dA, lda, strideA, dIpiv, strideP, batch_count);
    
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
  printf("Matrix size: %d x %d\n", M, N);
  printf("Batch count: %d\n", batch_count);
  printf("Warm-up time: %d ms (completed %d iterations)\n", warmup_time, warmup_count);
  printf("Timing iterations: %d\n", iterations);
  printf("Average execution time: %.3f ms\n", avg_time);
  printf("Standard deviation: %.3f ms\n", std_dev);
  printf("==============================\n\n");

  // clean up
  hipFree(dA);
  hipFree(dIpiv);
  free(hA);
  hipEventDestroy(start);
  hipEventDestroy(stop);
  rocblas_destroy_handle(handle);
}
