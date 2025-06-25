rocblas_status rocsolver_sgesvdj_strided_batched(rocblas_handle handle, const rocblas_svect left_svect, const rocblas_svect right_svect, const rocblas_int m, const rocblas_int n, float *A, const rocblas_int lda, const rocblas_stride strideA, const float abstol, float *residual, const rocblas_int max_sweeps, rocblas_int *n_sweeps, float *S, const rocblas_stride strideS, float *U, const rocblas_int ldu, const rocblas_stride strideU, float *V, const rocblas_int ldv, const rocblas_stride strideV, rocblas_int *info, const rocblas_int batch_count)  

GESVDJ_STRIDED_BATCHED computes the singular values and optionally the singular vectors of a batch of general m-by-n matrix A (Singular Value Decomposition).

The SVD of matrix A_l in the batch is given by: A_l = U_l S_l V_l'

where the m-by-n matrix S_l is zero except, possibly, for its min(m,n) diagonal elements, which are the singular values of A_l. U_l and V_l are orthogonal (unitary) matrices. The first min(m,n) columns of U_l and V_l are the left and right singular vectors of A_l, respectively.

The computation of the singular vectors is optional and it is controlled by the function arguments left_svect and right_svect as described below. When computed, this function returns the transpose (or transpose conjugate) of the right singular vectors, i.e. the rows of V_l'.

left_svect and right_svect are rocblas_svect enums that can take the following values:

- rocblas_svect_all: the entire matrix U_l (or V_l') is computed,
    
- rocblas_svect_singular: the singular vectors (first min(m,n) columns of U_l or rows of V_l') are computed, or
    
- rocblas_svect_none: no columns (or rows) of U_l (or V_l') are computed, i.e. no singular vectors.
    

The singular values are computed by applying QR factorization to A_l V_l if m >= n (resp. LQ factorization to U_l' A_l if m < n), where V_l (resp. U_l) is found as the eigenvectors of A_l' A_l (resp. A_l A_l') using the Jacobi eigenvalue algorithm.

Note

In order to carry out calculations, this method may synchronize the stream contained within the rocblas_handle.

Parameters:

- handle – [in] rocblas_handle.
    
- left_svect – [in] rocblas_svect. Specifies how the left singular vectors are computed. rocblas_svect_overwrite is not supported.
    
- right_svect – [in] rocblas_svect. Specifies how the right singular vectors are computed. rocblas_svect_overwrite is not supported.
    
- m – [in] rocblas_int. m >= 0. The number of rows of all matrices A_l in the batch.
    
- n – [in] rocblas_int. n >= 0. The number of columns of all matrices A_l in the batch.
    
- A – [inout] pointer to type. Array on the GPU (the size depends on the value of strideA). On entry, the matrices A_l. On exit, the contents of A_l are destroyed.
    
- lda – [in] rocblas_int. lda >= m. The leading dimension of A_l.
    
- strideA – [in] rocblas_stride. Stride from the start of one matrix A_l to the next one A_(l+1). There is no restriction for the value of strideA. Normal use case is strideA >= lda*n.
    
- abstol – [in] real type. The absolute tolerance. The algorithm is considered to have converged once off(A_l'A_l) is <= norm(A_l'A_l) * abstol [resp. off(A_lA_l') <= norm(A_lA_l') * abstol]. If abstol <= 0, then the tolerance will be set to machine precision.
    
- residual – [out] pointer to real type on the GPU. The Frobenius norm of the off-diagonal elements of A_l'A_l (resp. A_lA_l') at the final iteration.
    
- max_sweeps – [in] rocblas_int. max_sweeps > 0. Maximum number of sweeps (iterations) to be used by the algorithm.
    
- n_sweeps – [out] pointer to rocblas_int. Array of batch_count integers on the GPU. The actual number of sweeps (iterations) used by the algorithm for each batch instance.
    
- S – [out] pointer to real type. Array on the GPU (the size depends on the value of strideS). The singular values of A_l in decreasing order.
    
- strideS – [in] rocblas_stride. Stride from the start of one vector S_l to the next one S_(j+1). There is no restriction for the value of strideS. Normal use case is strideS >= min(m,n).
    
- U – [out] pointer to type. Array on the GPU (the side depends on the value of strideU). The matrices U_l of left singular vectors stored as columns. Not referenced if left_svect is set to none.
    
- ldu – [in] rocblas_int. ldu >= m if left_svect is set to all or singular; ldu >= 1 otherwise. The leading dimension of U_l.
    
- strideU – [in] rocblas_stride. Stride from the start of one matrix U_l to the next one U_(j+1). There is no restriction for the value of strideU. Normal use case is strideU >= ldu*min(m,n) if left_svect is set to singular, or strideU >= ldu*m when left_svect is equal to all.
    
- V – [out] pointer to type. Array on the GPU (the size depends on the value of strideV). The matrices V_l of right singular vectors stored as rows (transposed / conjugate-transposed). Not referenced if right_svect is set to none.
    
- ldv – [in] rocblas_int. ldv >= n if right_svect is set to all; ldv >= min(m,n) if right_svect is set to singular; or ldv >= 1 otherwise. The leading dimension of V.
    
- strideV – [in] rocblas_stride. Stride from the start of one matrix V_l to the next one V_(j+1). There is no restriction for the value of strideV. Normal use case is strideV >= ldv*n.
    
- info – [out] pointer to a rocblas_int on the GPU. If info[l] = 0, successful exit. If info[l] = 1, the algorithm did not converge.
    
- batch_count – [in] rocblas_int. batch_count >= 0. Number of matrices in the batch.
