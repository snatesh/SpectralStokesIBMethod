#include <lapacke.h>
#include <cblas.h>
#include <omp.h>

extern "C"
{
  /* compute LU of banded matrix A as well as X in AX = B
     A is overwritten with LU info on output, and B is 
     overwritten with X. */
  void precomputeBandedLinOps(double* A, double* B, double* C, 
                              double* D, double* G, double* G_inv, int* PIV, int kl, 
                              int ku, int Nyx, int Nz) 
  {
    int nrhs = 2, ldab = 2 * kl + ku + 1;
    #pragma omp parallel for
    for (unsigned int i = 1; i < Nyx; ++i)
    {
      unsigned int offset_a = ldab * Nz * i;
      unsigned int offset_bc = nrhs * Nz * i;
      unsigned int offset_dg = nrhs * nrhs * i;
      unsigned int offset_p = Nz * i;
      double* ab = &(A[offset_a]);
      double* b = &(B[offset_bc]);
      double* c = &(C[offset_bc]);
      double* d = &(D[offset_dg]);
      double* g = &(G[offset_dg]);
      double* g_inv = &(G_inv[offset_dg]);
      int* ipiv = &(PIV[offset_p]);
      // ab is overwritten with LU of ab
      // b is overwritten with ab^-1 b
      LAPACKE_dgbsv_work(LAPACK_COL_MAJOR, Nz, kl, ku, nrhs, ab, ldab, ipiv, b, Nz);
      cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                  nrhs, nrhs, Nz, 1.0, c, nrhs, b, Nz, 0.0, g, nrhs);
      g[0] -= d[0]; g[1] -= d[1]; g[2] -= d[2]; g[3] -= d[3];
      double det = 1.0 / (g[0] * g[3] - g[2] * g[1]);
      g_inv[0] = g[3] * det;
      g_inv[1] = -g[1] * det;
      g_inv[2] = -g[2] * det;
      g_inv[3] = g[0] * det; 
    }
  }

  /* Use the precomputed LU of banded matrix A to solve AX = RHS.
     RHS is overwritten with X */ 
  void bandedSchurSolve(double* LU, double* RHS, int* PIV, double* C, double* GINV, double* AINVB,
                        double* FIMat, double* SIMat, double* Cp, double* Dp, int kl, int ku, int Nyx, int Nz)
  {
    int nrhs = 1, nrhs1 = 2, ldab = 2 * kl + ku + 1;
    #pragma omp parallel for
    for (unsigned int i = 1; i < Nyx; ++i)
    {
      unsigned int offset_lu = ldab * Nz * i;
      unsigned int offset_rhs = Nz * i;
      unsigned int offset_bc = 2 * Nz * i;
      unsigned int offset_p = Nz * i;
      unsigned int offset_g = 2 * 2 * i;
      double* lu = &(LU[offset_lu]);
      int* ipiv = &(PIV[offset_p]);
      double* rhs = &(RHS[offset_rhs]);
      double* c = &(C[offset_bc]);
      double* ainvb = &(AINVB[offset_bc]);
      double* ginv = &(GINV[offset_g]);
      double* cp = &(Cp[offset_rhs]);
      double* dp = &(Dp[offset_rhs]);
      double* x = (double*) malloc((Nz + 2) * sizeof(double));
      double* y = (double*) malloc(2 * sizeof(double));
      // compute lu^{-1}*rhs
      LAPACKE_dgbtrs_work(LAPACK_COL_MAJOR, 'N', Nz, kl, ku,
                          1, lu, ldab, ipiv, rhs, Nz); 
      // compute  c*lu^{-1}*rhs (2x1)
      cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                  2, 1, Nz, 1.0, c, 2, rhs, Nz, 0.0, y, 2);
      // compute g^{-1}*(c*lu^{-1}*rhs)  (2 x 1)
      y[0] = ginv[0] * y[0] + ginv[2] * y[1];
      y[1] = ginv[1] * y[0] + ginv[3] * y[1];
      // compute lu^{-1}*rhs - lu^{-1}*b*y and save into rhs
      cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                  Nz, 1, 2, -1.0, ainvb, Nz, y, 2, 1.0, rhs, Nz);
      // copy into x
      for (unsigned int j = 0; j < Nz; ++j){x[j] = rhs[j];}
      x[Nz] = y[0]; x[Nz+1] = y[1];
      // compute sol and its derivative
      cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 
                  Nz, 1, Nz + 2, 1.0, SIMat, Nz, x, Nz + 2, 0.0, cp, Nz); 
      cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 
                  Nz, 1, Nz + 2, 1.0, FIMat, Nz, x, Nz + 2, 0.0, dp, Nz); 
      free(x);
      free(y);
    }
  }

  /* Use the precomputed LU of banded matrix A to solve AX = RHS.
     RHS is overwritten with X */ 
  void bandedSchurSolve_noD(double* LU, double* RHS, double* bc_RHS, int* PIV, 
                            double* C, double* GINV, double* AINVB, double* SIMat, 
                            double* Cp, int kl, int ku, int Nyx, int Nz)
  {
    int nrhs = 1, nrhs1 = 2, ldab = 2 * kl + ku + 1;
    #pragma omp parallel for
    for (unsigned int i = 1; i < Nyx; ++i)
    {
      unsigned int offset_lu = ldab * Nz * i;
      unsigned int offset_rhs = Nz * i;
      unsigned int offset_bc = 2 * Nz * i;
      unsigned int offset_p = Nz * i;
      unsigned int offset_g = 2 * 2 * i;
      unsigned int offset_g1 = 2 * i;
      double* lu = &(LU[offset_lu]);
      int* ipiv = &(PIV[offset_p]);
      double* rhs = &(RHS[offset_rhs]);
      double* c = &(C[offset_bc]);
      double* ainvb = &(AINVB[offset_bc]);
      double* ginv = &(GINV[offset_g]);
      double* bc_rhs = &(bc_RHS[offset_g1]);
      double* cp = &(Cp[offset_rhs]);
      double* x = (double*) malloc((Nz + 2) * sizeof(double));
      double* y = (double*) malloc(2 * sizeof(double));
      // compute lu^{-1}*rhs
      LAPACKE_dgbtrs_work(LAPACK_COL_MAJOR, 'N', Nz, kl, ku,
                          1, lu, ldab, ipiv, rhs, Nz); 
      // compute  c*lu^{-1}*rhs (2x1)
      cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                  2, 1, Nz, 1.0, c, 2, rhs, Nz, 0.0, y, 2);
      // compute g^{-1}*(c*lu^{-1}*rhs - (alpha,beta))  (2 x 1)
      y[0] = ginv[0] * (y[0] - bc_RHS[0]) + ginv[2] * (y[1] - bc_RHS[1]);
      y[1] = ginv[1] * (y[0] - bc_RHS[0]) + ginv[3] * (y[1] - bc_RHS[1]);
      // compute lu^{-1}*rhs - lu^{-1}*b*y and save into rhs
      cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                  Nz, 1, 2, -1.0, ainvb, Nz, y, 2, 1.0, rhs, Nz);
      // copy into x
      for (unsigned int j = 0; j < Nz; ++j){x[j] = rhs[j];}
      x[Nz] = y[0]; x[Nz+1] = y[1];
      // compute sol and its derivative
      cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 
                  Nz, 1, Nz + 2, 1.0, SIMat, Nz, x, Nz + 2, 0.0, cp, Nz); 

      free(x);
      free(y);
    }
  }


}
