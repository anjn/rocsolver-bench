#include <hip/hip_runtime_api.h> // for hip functions
#include <rocsolver/rocsolver.h> // for all the rocsolver C interfaces and type declarations
#include <stdio.h>   // for printf
#include <stdlib.h> // for malloc
#include <random> // for random number generation
#include <vector> // for storing timing results
#include <cmath> // for sqrt in standard deviation calculation
#include <iostream> // for cout/cerr

#include <argparse/argparse.hpp>

// Example: Compute the singular values and singular vectors of an array of general matrices on the GPU

float *create_matrices_for_sgesvdj_strided_batched(rocblas_int M,
                                                  rocblas_int N,
                                                  rocblas_int lda,
                                                  rocblas_stride strideA,
                                                  rocblas_int batch_count,
                                                  int random_seed) {
  // allocate space for input matrix data on CPU
  float *hA = (float*)malloc(sizeof(float) * strideA * batch_count);

  // generate random general matrices
  std::mt19937 gen(random_seed);
  std::uniform_real_distribution<float> dis(-10.0, 10.0);
  
  for (rocblas_int b = 0; b < batch_count; ++b) {
    for (rocblas_int i = 0; i < M; ++i) {
      for (rocblas_int j = 0; j < N; ++j) {
        hA[i + j * lda + b * strideA] = dis(gen);
      }
    }
  }

  return hA;
}

// Use rocsolver_sgesvdj_strided_batched to compute singular values and singular vectors of an array of general matrices.
int main(int argc, char *argv[]) {
  // ArgumentParserの設定
  argparse::ArgumentParser program("bench_rocsolver_sgesvdj_strided_batched");
  
  program.add_argument("-m", "--rows")
      .help("Number of rows (M)")
      .default_value(10)
      .scan<'i', int>();
      
  program.add_argument("-n", "--cols")
      .help("Number of columns (N)")
      .default_value(8)
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
      
  program.add_argument("-j", "--max-sweeps")
      .help("Maximum number of sweeps for Jacobi method")
      .default_value(100)
      .scan<'i', int>();
      
  program.add_argument("--left-svect")
      .help("Left singular vectors computation (none, singular, all)")
      .default_value(std::string("all"));
      
  program.add_argument("--right-svect")
      .help("Right singular vectors computation (none, singular, all)")
      .default_value(std::string("all"));
  
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
  float tolerance = program.get<float>("--tolerance");
  rocblas_int max_sweeps = program.get<int>("--max-sweeps");
  std::string left_svect_str = program.get<std::string>("--left-svect");
  std::string right_svect_str = program.get<std::string>("--right-svect");

  if (lda < M) lda = M;
  
  // ストライドの計算（指定されていない場合はlda * Nを使用）
  rocblas_stride strideA;
  if (program.present("--stride")) {
    strideA = program.get<int>("--stride");
  } else {
    strideA = lda * N;
  }
  
  // Parse singular vector computation options
  rocblas_svect left_svect, right_svect;
  
  if (left_svect_str == "none") {
    left_svect = rocblas_svect_none;
  } else if (left_svect_str == "singular") {
    left_svect = rocblas_svect_singular;
  } else {
    left_svect = rocblas_svect_all;
  }
  
  if (right_svect_str == "none") {
    right_svect = rocblas_svect_none;
  } else if (right_svect_str == "singular") {
    right_svect = rocblas_svect_singular;
  } else {
    right_svect = rocblas_svect_all;
  }
  
  // create_matrices_for_sgesvdj_strided_batched関数の呼び出し
  float *hA = create_matrices_for_sgesvdj_strided_batched(M, N, lda, strideA, batch_count, random_seed);

  // initialization
  rocblas_handle handle;
  rocblas_create_handle(&handle);

  // calculate the sizes of our arrays
  size_t size_A = strideA * (size_t)batch_count;   // elements in array for matrices
  rocblas_int min_mn = (M < N) ? M : N;            // min(M,N)
  rocblas_stride strideS = min_mn;                 // stride of singular values
  size_t size_S = strideS * (size_t)batch_count;   // elements in array for singular values
  size_t size_info = batch_count;                  // elements in info array
  
  // Determine sizes for U and V matrices based on left_svect and right_svect
  rocblas_int ldu = (left_svect == rocblas_svect_none) ? 1 : M;
  rocblas_int ldv = (right_svect == rocblas_svect_none) ? 1 : 
                    (right_svect == rocblas_svect_singular) ? min_mn : N;
  
  rocblas_stride strideU = (left_svect == rocblas_svect_none) ? 1 : 
                          (left_svect == rocblas_svect_singular) ? ldu * min_mn : ldu * M;
  rocblas_stride strideV = (right_svect == rocblas_svect_none) ? 1 : ldv * N;
  
  size_t size_U = strideU * (size_t)batch_count;
  size_t size_V = strideV * (size_t)batch_count;

  // allocate memory on GPU
  float *dA, *dS, *dU, *dV, *dResidual;
  rocblas_int *dInfo, *dNSweeps;
  hipMalloc((void**)&dA, sizeof(float)*size_A);
  hipMalloc((void**)&dS, sizeof(float)*size_S);
  hipMalloc((void**)&dU, sizeof(float)*size_U);
  hipMalloc((void**)&dV, sizeof(float)*size_V);
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

  while (warmup_elapsed < warmup_time || warmup_count == 0) {
    // run the computation (without timing)
    rocsolver_sgesvdj_strided_batched(handle, left_svect, right_svect, M, N, dA, lda, strideA, 
                                     tolerance, dResidual, max_sweeps, dNSweeps, 
                                     dS, strideS, dU, ldu, strideU, dV, ldv, strideV, 
                                     dInfo, batch_count);
    
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
    // Copy fresh data to GPU for each iteration
    hipMemcpy(dA, hA, sizeof(float)*size_A, hipMemcpyHostToDevice);
    
    // start timing
    hipEventRecord(start, 0);
    
    // compute the SVD on the GPU
    rocsolver_sgesvdj_strided_batched(handle, left_svect, right_svect, M, N, dA, lda, strideA, 
                                     tolerance, dResidual, max_sweeps, dNSweeps, 
                                     dS, strideS, dU, ldu, strideU, dV, ldv, strideV, 
                                     dInfo, batch_count);
    
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
  printf("Left singular vectors: %s\n", left_svect_str.c_str());
  printf("Right singular vectors: %s\n", right_svect_str.c_str());
  printf("Tolerance: %e\n", tolerance);
  printf("Max sweeps: %d\n", max_sweeps);
  printf("Warm-up time: %d ms (completed %d iterations)\n", warmup_time, warmup_count);
  printf("Timing iterations: %d\n", iterations);
  printf("Average execution time: %.3f ms\n", avg_time);
  printf("Standard deviation: %.3f ms\n", std_dev);
  printf("==============================\n\n");

  // clean up
  hipFree(dA);
  hipFree(dS);
  hipFree(dU);
  hipFree(dV);
  hipFree(dInfo);
  hipFree(dResidual);
  hipFree(dNSweeps);
  free(hA);
  hipEventDestroy(start);
  hipEventDestroy(stop);
  rocblas_destroy_handle(handle);
}
