rocblas_status rocsolver_ssyevj_strided_batched(rocblas_handle handle, const rocblas_esort esort, const rocblas_evect evect, const rocblas_fill uplo, const rocblas_int n, float *A, const rocblas_int lda, const rocblas_stride strideA, const float abstol, float *residual, const rocblas_int max_sweeps, rocblas_int *n_sweeps, float *W, const rocblas_stride strideW, rocblas_int *info, const rocblas_int batch_count)  

SYEVJ_STRIDED_BATCHED computes the eigenvalues and optionally the eigenvectors of a batch of real symmetric matrices A_l.

The eigenvalues are found using the iterative Jacobi algorithm and are returned in an order depending on the value of esort. The eigenvectors are computed depending on the value of evect. The computed eigenvectors are orthonormal.

Note

In order to carry out calculations, this method may synchronize the stream contained within the rocblas_handle.

Parameters:

- handle – [in] rocblas_handle.
    
- esort – [in] rocblas_esort. Specifies the order of the returned eigenvalues. If esort is rocblas_esort_ascending, then the eigenvalues are sorted and returned in ascending order. If esort is rocblas_esort_none, then the order of the returned eigenvalues is unspecified.
    
- evect – [in] rocblas_evect. Specifies whether the eigenvectors are to be computed. If evect is rocblas_evect_original, then the eigenvectors are computed. rocblas_evect_tridiagonal is not supported.
    
- uplo – [in] rocblas_fill. Specifies whether the upper or lower part of the symmetric matrices A_l is stored. If uplo indicates lower (or upper), then the upper (or lower) part of A_l is not used.
    
- n – [in] rocblas_int. n >= 0. Number of rows and columns of matrices A_l.
    
- A – [inout] pointer to type. Array on the GPU (the size depends on the value of strideA). On entry, the matrices A_l. On exit, the eigenvectors of A_l if they were computed and the algorithm converged; otherwise the contents of A_l are unchanged.
    
- lda – [in] rocblas_int. lda >= n. Specifies the leading dimension of matrices A_l.
    
- strideA – [in] rocblas_stride. Stride from the start of one matrix A_l to the next one A_(l+1). There is no restriction for the value of strideA. Normal use case is strideA >= lda*n.
    
- abstol – [in] type. The absolute tolerance. The algorithm is considered to have converged once off(A_l) is <= abstol. If abstol <= 0, then the tolerance will be set to machine precision.
    
- residual – [out] pointer to type. Array of batch_count scalars on the GPU. The Frobenius norm of the off-diagonal elements of A_l (i.e. off(A_l)) at the final iteration.
    
- max_sweeps – [in] rocblas_int. max_sweeps > 0. Maximum number of sweeps (iterations) to be used by the algorithm.
    
- n_sweeps – [out] pointer to rocblas_int. Array of batch_count integers on the GPU. The actual number of sweeps (iterations) used by the algorithm for each batch instance.
    
- W – [out] pointer to type. Array on the GPU (the size depends on the value of strideW). The eigenvalues of A_l in increasing order.
    
- strideW – [in] rocblas_stride. Stride from the start of one vector W_l to the next one W_(l+1). There is no restriction for the value of strideW. Normal use case is strideW >= n.
    
- info – [out] pointer to rocblas_int. Array of batch_count integers on the GPU. If info[l] = 0, successful exit for matrix A_l. If info[l] = 1, the algorithm did not converge.
    
- batch_count – [in] rocblas_int. batch_count >= 0. Number of matrices in the batch.
