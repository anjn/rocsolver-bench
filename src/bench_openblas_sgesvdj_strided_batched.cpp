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

// Example: Compute the singular values and singular vectors of an array of general matrices on the CPU using OpenBLAS

float *create_matrices_for_sgesvdj_strided_batched(lapack_int M,
                                                  lapack_int N,
                                                  lapack_int lda,
                                                  size_t strideA,
                                                  lapack_int batch_count,
                                                  int random_seed) {
  // allocate space for input matrix data on CPU
  float *hA = (float*)malloc(sizeof(float) * strideA * batch_count);

  // generate random general matrices
  std::mt19937 gen(random_seed);
  std::uniform_real_distribution<float> dis(-10.0, 10.0);
  
  for (lapack_int b = 0; b < batch_count; ++b) {
    for (lapack_int i = 0; i < M; ++i) {
      for (lapack_int j = 0; j < N; ++j) {
        hA[i + j * lda + b * strideA] = dis(gen);
      }
    }
  }

  return hA;
}

// Use LAPACKE_sgesvd to compute singular values and singular vectors of an array of general matrices.
int main(int argc, char *argv[]) {
  // ArgumentParserの設定
  argparse::ArgumentParser program("bench_openblas_sgesvdj_strided_batched");
  
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
  lapack_int M = program.get<int>("--rows");
  lapack_int N = program.get<int>("--cols");
  lapack_int lda = program.get<int>("--lda");
  lapack_int batch_count = program.get<int>("--batch-count");
  int random_seed = program.get<int>("--random-seed");
  int iterations = program.get<int>("--iterations");
  int warmup_time = program.get<int>("--warmup-time");
  std::string left_svect_str = program.get<std::string>("--left-svect");
  std::string right_svect_str = program.get<std::string>("--right-svect");

  if (lda < M) lda = M;
  
  // ストライドの計算（指定されていない場合はlda * Nを使用）
  size_t strideA;
  if (program.present("--stride")) {
    strideA = program.get<int>("--stride");
  } else {
    strideA = lda * N;
  }
  
  // Parse singular vector computation options
  char jobu, jobvt;
  
  if (left_svect_str == "none") {
    jobu = 'N'; // No left singular vectors are computed
  } else if (left_svect_str == "singular") {
    jobu = 'S'; // The first min(M,N) columns of U are returned in the array U
  } else {
    jobu = 'A'; // All M columns of U are returned in the array U
  }
  
  if (right_svect_str == "none") {
    jobvt = 'N'; // No right singular vectors are computed
  } else if (right_svect_str == "singular") {
    jobvt = 'S'; // The first min(M,N) rows of V^T are returned in the array VT
  } else {
    jobvt = 'A'; // All N rows of V^T are returned in the array VT
  }
  
  // create_matrices_for_sgesvdj_strided_batched関数の呼び出し
  float *hA = create_matrices_for_sgesvdj_strided_batched(M, N, lda, strideA, batch_count, random_seed);

  // calculate the sizes of our arrays
  size_t size_A = strideA * (size_t)batch_count;   // elements in array for matrices
  lapack_int min_mn = (M < N) ? M : N;             // min(M,N)
  size_t strideS = min_mn;                         // stride of singular values
  size_t size_S = strideS * (size_t)batch_count;   // elements in array for singular values
  
  // Determine sizes for U and VT matrices based on jobu and jobvt
  lapack_int ldu = (jobu == 'N') ? 1 : M;
  lapack_int ldvt = (jobvt == 'N') ? 1 : N;
  
  size_t strideU = (jobu == 'N') ? 1 : ldu * ((jobu == 'A') ? M : min_mn);
  size_t strideVT = (jobvt == 'N') ? 1 : ldvt * N;
  
  size_t size_U = strideU * (size_t)batch_count;
  size_t size_VT = strideVT * (size_t)batch_count;

  // allocate memory for singular values, singular vectors, and workspace
  float *hS = (float*)malloc(sizeof(float) * size_S);
  float *hU = (float*)malloc(sizeof(float) * size_U);
  float *hVT = (float*)malloc(sizeof(float) * size_VT);
  
  // Create a copy of the original matrices for each iteration
  float *hA_copy = (float*)malloc(sizeof(float) * size_A);
  
  // Allocate workspace for LAPACK
  // For simplicity, we'll use a large workspace
  lapack_int lwork = 5 * std::max(M, N);
  float *work = (float*)malloc(sizeof(float) * lwork);
  
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
    for (lapack_int b = 0; b < batch_count; ++b) {
      float* A_batch = hA_copy + b * strideA;
      float* S_batch = hS + b * strideS;
      float* U_batch = hU + b * strideU;
      float* VT_batch = hVT + b * strideVT;
      
      // Compute SVD
      // LAPACKE_sgesvd parameters:
      // - matrix_layout: LAPACK_COL_MAJOR for column-major layout
      // - jobu: 'A', 'S', or 'N' for left singular vectors computation
      // - jobvt: 'A', 'S', or 'N' for right singular vectors computation
      // - m: number of rows
      // - n: number of columns
      // - a: input/output matrix
      // - lda: leading dimension of a
      // - s: output singular values
      // - u: output left singular vectors
      // - ldu: leading dimension of u
      // - vt: output right singular vectors (transposed)
      // - ldvt: leading dimension of vt
      // - work: workspace
      // - lwork: size of workspace
      lapack_int info = LAPACKE_sgesvd_work(LAPACK_COL_MAJOR, jobu, jobvt, 
                                           M, N, A_batch, lda, S_batch, 
                                           U_batch, ldu, VT_batch, ldvt, 
                                           work, lwork);
      
      if (info != 0) {
        printf("LAPACKE_sgesvd failed for matrix %d with error %d\n", (int)b, (int)info);
      }
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
    for (lapack_int b = 0; b < batch_count; ++b) {
      float* A_batch = hA_copy + b * strideA;
      float* S_batch = hS + b * strideS;
      float* U_batch = hU + b * strideU;
      float* VT_batch = hVT + b * strideVT;
      
      // Compute SVD
      lapack_int info = LAPACKE_sgesvd_work(LAPACK_COL_MAJOR, jobu, jobvt, 
                                           M, N, A_batch, lda, S_batch, 
                                           U_batch, ldu, VT_batch, ldvt, 
                                           work, lwork);
      
      if (info != 0) {
        printf("LAPACKE_sgesvd failed for matrix %d with error %d\n", (int)b, (int)info);
      }
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
  printf("Matrix size: %d x %d\n", (int)M, (int)N);
  printf("Batch count: %d\n", (int)batch_count);
  printf("Left singular vectors: %s\n", left_svect_str.c_str());
  printf("Right singular vectors: %s\n", right_svect_str.c_str());
  printf("Warm-up time: %d ms (completed %d iterations)\n", warmup_time, warmup_count);
  printf("Timing iterations: %d\n", iterations);
  printf("Average execution time: %.3f ms\n", avg_time);
  printf("Standard deviation: %.3f ms\n", std_dev);
  printf("==============================================\n\n");

  // clean up
  free(hA);
  free(hA_copy);
  free(hS);
  free(hU);
  free(hVT);
  free(work);
  
  return 0;
}
