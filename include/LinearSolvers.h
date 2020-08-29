#ifndef _LINEAR_SOLVERS_H
#define _LINEAR_SOLVERS_H
extern "C"
{
  /* compute LU of banded matrix A as well as X in AX = B
     A is overwritten with LU info on output, and B is 
     overwritten with X. */
  void precomputeBandedLinOps(double* A, double* B, double* C, 
                              double* D, double* G, double* G_inv, int* PIV, int kl, 
                              int ku, int Nyx, int Nz);
  /* Use the precomputed LU of banded matrix A to solve AX = RHS.
     RHS is overwritten with X */ 
  void bandedSchurSolve(double* LU, double* RHS, int* PIV, int kl, int ku, int Nyx, int Nz);
}

#endif
