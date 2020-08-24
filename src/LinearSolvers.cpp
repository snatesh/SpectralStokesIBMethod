#include <lapacke.h>
#include <cblas.h>
#include <omp.h>
#include <complex.h>
extern "C"
{
  // compute LU of banded matrix A as well as X in AX = B
  // A is overwritten with LU info on output, and B is 
  // overwritten with X.
  void precomputeBandedLinOps(double* A, double* B, double* C, 
                              double* D, double* G, double* G_inv, int* PIV, int kl, 
                              int ku, int Nyx, int Nz) 
  {
    int nrhs = 2, ldab = 2 * kl + ku + 1;
    #pragma omp parallel for
    for (unsigned int i = 0; i < Nyx; ++i)
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
//(7, 29, 16384)A
//(29, 2, 16384)B
//(2, 29, 16384)C
//(2, 2, 16384)D
//  #(2*kl+ku+1, Nz, Nxy) 

  void bandedSchurSolve(double* LU, double* RHS, int* PIV, int kl, int ku, int Nyx, int Nz)
  {
    int nrhs = 1, ldab = 2 * kl + ku + 1;
    #pragma omp parallel for
    for (unsigned int i = 0; i < Nyx; ++i)
    {
      unsigned int offset_lu = ldab * Nz * i;
      unsigned int offset_rhs = nrhs * Nz * i;
      unsigned int offset_p = Nz * i;
      double* lu = &(LU[offset_lu]);
      int* ipiv = &(PIV[offset_p]);
      double* rhs = &(RHS[offset_rhs]);
      LAPACKE_dgbtrs_work(LAPACK_COL_MAJOR, 'N', Nz, kl, ku,
                          nrhs, lu, ldab, ipiv, rhs, Nz); 
    }
  }
}
