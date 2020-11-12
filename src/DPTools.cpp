#include<math.h> 
#include<fftw3.h> 
#include<cblas.h>
#include <lapacke.h>
#include<omp.h>
#include"DPTools.h"

extern "C"
{
  void evalTheta(const double* in, double* out, double theta, 
                 unsigned int Nyx, unsigned int Nz, unsigned int dof)
  {
    #pragma omp parallel for
    for (unsigned int i = 0; i < Nyx; ++i)
    {
      double* out_xy = &(out[dof * i]);
      for (unsigned int j = 0; j < Nz; ++j)
      {
        const double* in_xyz = &(in[dof * (i + Nyx * j)]);
        double alpha = cos(j * theta);
        #pragma omp simd
        for (unsigned int l = 0; l < dof; ++l)
        {
          out_xy[l] += in_xyz[l] * alpha; 
        }
      }
    }
  }

  void chebTransform(double* in_re, double* in_im, double* out_re, 
                     double* out_im, const fftw_plan plan, unsigned int N)
  {
    unsigned int ext = 2 * N - 2;
    fftw_complex* in = (fftw_complex*) fftw_malloc(ext * sizeof(fftw_complex));
    for (unsigned int i = 0; i < N; ++i)
    {
      in[i][0] = in_re[i];
      in[i][1] = in_im[i];
    }
    for (unsigned int i = N; i < ext; ++i)
    {
      in[i][0] = in_re[ext - i];
      in[i][1] = in_im[ext - i];
    }
    fftw_execute_dft(plan, in, in);
    out_re[0] = in[0][0] / ((double) ext); 
    out_im[0] = in[0][1] / ((double) ext);
    for (unsigned int i = 1; i < N - 1; ++i)
    {
      out_re[i] = (in[i][0] + in[ext - i][0]) / ((double) ext);
      out_im[i] = (in[i][1] + in[ext - i][1]) / ((double) ext);
    } 
    out_re[N-1] = in[N-1][0] / ((double) ext);
    out_im[N-1] = in[N-1][1] / ((double) ext);
    fftw_free(in);
  }

  void evalCorrectionSol_bottomWall(double* Cpcorr_r, double* Cpcorr_i, double* Cucorr_r, 
                                    double* Cucorr_i, double* Cvcorr_r, double* Cvcorr_i,
                                    double* Cwcorr_r, double* Cwcorr_i, const double* fhat_r,
                                    const double* fhat_i, const double* Kx, const double* Ky, 
                                    const double* z, double eta, unsigned int Nyx, 
                                    unsigned int Nz, unsigned int dof)
  {
    // create single fftw forward plan using tmp arrays
    // for re-use in loop
    fftw_init_threads();
    fftw_plan_with_nthreads(1);
    fftw_complex* in = (fftw_complex*) fftw_malloc((2 * Nz - 2) * sizeof(fftw_complex));
    fftw_complex* out = in;
    fftw_plan fplan = fftw_plan_dft_1d(2 * Nz - 2, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_free(in);
  
    #pragma omp parallel for
    for (unsigned int i = 1; i < Nyx; ++i)
    {
      unsigned int ix = dof * i, iy = ix + 1, iz = iy + 1;
      double kx = Kx[i], ky = Ky[i], k = sqrt(kx * kx + ky * ky); 
      double* enkz = (double*) fftw_malloc(Nz * sizeof(double));
      double* zenkz = (double*) fftw_malloc(Nz * sizeof(double));
      double* Cp_r = (double*) fftw_malloc(Nz * sizeof(double));
      double* Cp_i = (double*) fftw_malloc(Nz * sizeof(double));
      double* Cu_r = (double*) fftw_malloc(Nz * sizeof(double));
      double* Cu_i = (double*) fftw_malloc(Nz * sizeof(double));
      double* Cv_r = (double*) fftw_malloc(Nz * sizeof(double));
      double* Cv_i = (double*) fftw_malloc(Nz * sizeof(double));
      double* Cw_r = (double*) fftw_malloc(Nz * sizeof(double));
      double* Cw_i = (double*) fftw_malloc(Nz * sizeof(double));
      #pragma omp simd
      for (unsigned int j = 0; j < Nz; ++j) 
      {
        enkz[j] = exp(-k * z[j]);
        zenkz[j] = z[j] * enkz[j];
        Cp_r[j] = Cp_i[j] = Cu_r[j] = Cu_i[j] = 0;
        Cv_r[j] = Cv_i[j] = Cw_r[j] = Cw_i[j] = 0;
      }
      double alpha_r = kx * fhat_r[ix] + ky * fhat_r[iy] - k * fhat_i[iz];
      double alpha_i = kx * fhat_i[ix] + ky * fhat_i[iy] + k * fhat_r[iz];
      // correction for pressure
      cblas_daxpy(Nz, 2.0 * eta * alpha_i, enkz, 1, Cp_r, 1);
      cblas_daxpy(Nz, -2.0 * eta * alpha_r, enkz, 1, Cp_i, 1); 
      // correction for x vel
      cblas_daxpy(Nz, -kx * alpha_r / k, zenkz, 1, Cu_r, 1);      
      cblas_daxpy(Nz, fhat_r[ix], enkz, 1, Cu_r, 1);
      cblas_daxpy(Nz, -kx * alpha_i / k, zenkz, 1, Cu_i, 1);      
      cblas_daxpy(Nz, fhat_i[ix], enkz, 1, Cu_i, 1);
      // correction for y vel
      cblas_daxpy(Nz, -ky * alpha_r / k, zenkz, 1, Cv_r, 1);      
      cblas_daxpy(Nz, fhat_r[iy], enkz, 1, Cv_r, 1);
      cblas_daxpy(Nz, -ky * alpha_i / k, zenkz, 1, Cv_i, 1);      
      cblas_daxpy(Nz, fhat_i[iy], enkz, 1, Cv_i, 1);
      // correction for z vel
      cblas_daxpy(Nz, alpha_i, zenkz, 1, Cw_r, 1);
      cblas_daxpy(Nz, fhat_r[iz], enkz, 1, Cw_r, 1); 
      cblas_daxpy(Nz, -alpha_r, zenkz, 1, Cw_i, 1);
      cblas_daxpy(Nz, fhat_i[iz], enkz, 1, Cw_i, 1);
      // forward transform in z to get cheb coeffs
      unsigned int offset = i * Nz;
      chebTransform(Cp_r, Cp_i, &(Cpcorr_r[offset]), 
                    &(Cpcorr_i[offset]), fplan, Nz);
      chebTransform(Cu_r, Cu_i, &(Cucorr_r[offset]),
                    &(Cucorr_i[offset]), fplan, Nz);
      chebTransform(Cv_r, Cv_i, &(Cvcorr_r[offset]),
                    &(Cvcorr_i[offset]), fplan, Nz);
      chebTransform(Cw_r, Cw_i, &(Cwcorr_r[offset]),
                    &(Cwcorr_i[offset]), fplan, Nz);
      // clean
      fftw_free(enkz);
      fftw_free(zenkz);
      fftw_free(Cp_r);  
      fftw_free(Cp_i);  
      fftw_free(Cu_r);  
      fftw_free(Cu_i);  
      fftw_free(Cv_r);  
      fftw_free(Cv_i);  
      fftw_free(Cw_r);  
      fftw_free(Cw_i);  
    }
    fftw_destroy_plan(fplan);
  }
 
  inline unsigned int at(unsigned int i, unsigned int j){return i + 8 * j;}

  void evalCorrectionSol_slitChannel(double* Cpcorr_r, double* Cpcorr_i, double* Cucorr_r, 
                                     double* Cucorr_i, double* Cvcorr_r, double* Cvcorr_i,
                                     double* Cwcorr_r, double* Cwcorr_i, const double* fbhat_r, 
                                     const double* fbhat_i, const double* fthat_r, const double* fthat_i, 
                                     const double* Kx, const double* Ky, const double* z, double H, 
                                     double eta, unsigned int Nyx, unsigned int Nz, unsigned int dof)
  {
    // create single fftw forward plan using tmp arrays
    // for re-use in loop
    fftw_init_threads();
    fftw_plan_with_nthreads(1);
    fftw_complex* in = (fftw_complex*) fftw_malloc((2 * Nz - 2) * sizeof(fftw_complex));
    fftw_complex* out = in;
    fftw_plan fplan = fftw_plan_dft_1d(2 * Nz - 2, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_free(in);
    double fac = 1.0 / 2.0 / eta;
    #pragma omp parallel for
    for (unsigned int i = 1; i < Nyx; ++i)
    {
      unsigned int ix = dof * i, iy = ix + 1, iz = iy + 1;
      double kx = Kx[i], ky = Ky[i], k = sqrt(kx * kx + ky * ky);
      // e^(-kz) 
      double* enkz = (double*) fftw_malloc(Nz * sizeof(double));
      // e^(k(z-H))
      double* ekzmh = (double*) fftw_malloc(Nz * sizeof(double));
      // ze^(k(z-H))
      double* zekzmh = (double*) fftw_malloc(Nz * sizeof(double));
      // e^(-kH)
      double enkh = exp(-k * H);
      // ze^(-kz)
      double* zenkz = (double*) fftw_malloc(Nz * sizeof(double));
      // real and imaginary components of pressure and vel
      double* Cp_r = (double*) fftw_malloc(Nz * sizeof(double));
      double* Cp_i = (double*) fftw_malloc(Nz * sizeof(double));
      double* Cu_r = (double*) fftw_malloc(Nz * sizeof(double));
      double* Cu_i = (double*) fftw_malloc(Nz * sizeof(double));
      double* Cv_r = (double*) fftw_malloc(Nz * sizeof(double));
      double* Cv_i = (double*) fftw_malloc(Nz * sizeof(double));
      double* Cw_r = (double*) fftw_malloc(Nz * sizeof(double));
      double* Cw_i = (double*) fftw_malloc(Nz * sizeof(double));
      // matrix to solve for coeffs of exponentials in correction sol
      lapack_complex_double* coeffA = (lapack_complex_double*) fftw_malloc(64 * sizeof(lapack_complex_double));
      // right hand side of coeffA_r c_r = x_r (real and complex parts)
      lapack_complex_double* x = (lapack_complex_double*) fftw_malloc(8 * sizeof(lapack_complex_double));
      lapack_complex_double* c = x;
      // pivot storage for lapack
      int* piv = (int*) fftw_malloc(8 * sizeof(int));
      #pragma omp simd
      for (unsigned int j = 0; j < 64; ++j) {coeffA[j].real = coeffA[j].imag = 0;} 
      #pragma omp simd
      for (unsigned int j = 0; j < Nz; ++j) 
      {
        enkz[j] = exp(-k * z[j]);
        ekzmh[j] = enkh / enkz[j];
        zekzmh[j] = z[j] * ekzmh[j];
        zenkz[j] = z[j] * enkz[j];
        Cp_r[j] = Cp_i[j] = Cu_r[j] = Cu_i[j] = 0;
        Cv_r[j] = Cv_i[j] = Cw_r[j] = Cw_i[j] = 0;
      }
      coeffA[at(0,0)].real = fac;
      coeffA[at(0,1)].real = enkh * fac;
      coeffA[at(0,6)].real = -k;
      coeffA[at(0,7)].real = k * enkh;
      coeffA[at(1,0)].real = (1 - k * H) * enkh * fac;
      coeffA[at(1,1)].real = (1 + k * H) * fac;
      coeffA[at(1,6)].real = -k * enkh;
      coeffA[at(1,7)].real = k;
      coeffA[at(2,2)].real = 1;
      coeffA[at(2,3)].real = enkh;
      coeffA[at(3,4)].real = 1;
      coeffA[at(3,5)].real = enkh;
      coeffA[at(4,6)].real = 1;
      coeffA[at(4,7)].real = enkh;
      coeffA[at(5,0)].imag = -kx * H * enkh * fac / k;
      coeffA[at(5,1)].imag = kx * H * fac / k;
      coeffA[at(5,2)].real = enkh;
      coeffA[at(5,3)].real = 1;
      coeffA[at(6,0)].imag = -ky * H * enkh * fac / k;
      coeffA[at(6,1)].imag = ky * H * fac / k;
      coeffA[at(6,4)].real = enkh;
      coeffA[at(6,5)].real = 1;
      coeffA[at(7,0)].real = H * enkh * fac;
      coeffA[at(7,1)].real = H * fac;
      coeffA[at(7,6)].real = enkh;
      coeffA[at(7,7)].real = 1; 

      x[0].real = kx * fbhat_i[ix] + ky * fbhat_i[iy];
      x[0].imag = -kx * fbhat_r[ix] - ky * fbhat_r[iy]; 
      x[1].real = kx * fthat_i[ix] + ky * fthat_i[iy];
      x[1].imag = -kx * fthat_r[ix] - ky * fthat_r[iy]; 
      x[2].real = fbhat_r[ix]; x[2].imag = fbhat_i[ix];
      x[3].real = fbhat_r[iy]; x[3].imag = fbhat_i[iy];
      x[4].real = fbhat_r[iz]; x[4].imag = fbhat_i[iz];
      x[5].real = fthat_r[ix]; x[5].imag = fthat_i[ix];
      x[6].real = fthat_r[iy]; x[6].imag = fthat_i[iy];
      x[7].real = fthat_r[iz]; x[7].imag = fthat_i[iz];
      // solve for coefficients of exponentials   
      LAPACKE_zgesv(LAPACK_COL_MAJOR, 8, 1, coeffA, 8, piv, x, 8);      
      // correction for pressure
      cblas_daxpy(Nz, c[0].real, enkz, 1, Cp_r, 1);
      cblas_daxpy(Nz, c[1].real, ekzmh, 1, Cp_r, 1); 
      cblas_daxpy(Nz, c[0].imag, enkz, 1, Cp_i, 1);
      cblas_daxpy(Nz, c[1].imag, ekzmh, 1, Cp_i, 1); 
      // correction for x vel
      cblas_daxpy(Nz, c[0].imag * kx * fac / k, zenkz, 1,Cu_r, 1);
      cblas_daxpy(Nz, -c[1].imag * kx * fac / k, zekzmh, 1, Cu_r, 1);
      cblas_daxpy(Nz, c[2].real, enkz, 1, Cu_r, 1);
      cblas_daxpy(Nz, c[3].real, ekzmh, 1, Cu_r, 1);
      cblas_daxpy(Nz, -c[0].real * kx * fac / k, zenkz, 1, Cu_i, 1);
      cblas_daxpy(Nz, c[1].real * kx * fac / k, zekzmh, 1, Cu_i, 1);
      cblas_daxpy(Nz, c[2].imag, enkz, 1, Cu_i, 1);
      cblas_daxpy(Nz, c[3].imag, ekzmh, 1, Cu_i, 1);
      // correction for y vel
      cblas_daxpy(Nz, c[0].imag * ky * fac / k, zenkz, 1, Cv_r, 1);
      cblas_daxpy(Nz, -c[1].imag * ky * fac / k, zekzmh, 1, Cv_r, 1);
      cblas_daxpy(Nz, c[4].real, enkz, 1, Cv_r, 1);
      cblas_daxpy(Nz, c[5].real, ekzmh, 1, Cv_r, 1);
      cblas_daxpy(Nz, -c[0].real * ky * fac / k, zenkz, 1, Cv_i, 1);
      cblas_daxpy(Nz, c[1].real * ky * fac / k, zekzmh, 1, Cv_i, 1);
      cblas_daxpy(Nz, c[4].imag, enkz, 1, Cv_i, 1);
      cblas_daxpy(Nz, c[5].imag, ekzmh, 1, Cv_i, 1);
      // correction for z vel
      cblas_daxpy(Nz, c[0].real * fac, zenkz, 1, Cw_r, 1);
      cblas_daxpy(Nz, c[1].real * fac, zekzmh, 1, Cw_r, 1);
      cblas_daxpy(Nz, c[6].real, enkz, 1, Cw_r, 1);
      cblas_daxpy(Nz, c[7].real, ekzmh, 1, Cw_r, 1);
      cblas_daxpy(Nz, c[0].imag * fac, zenkz, 1, Cw_i, 1);
      cblas_daxpy(Nz, c[1].imag * fac, zekzmh, 1, Cw_i, 1);
      cblas_daxpy(Nz, c[6].imag, enkz, 1, Cw_i, 1);
      cblas_daxpy(Nz, c[7].imag, ekzmh, 1, Cw_i, 1);
      // forward transform in z to get cheb coeffs
      unsigned int offset = i * Nz;
      chebTransform(Cp_r, Cp_i, &(Cpcorr_r[offset]), 
                    &(Cpcorr_i[offset]), fplan, Nz);
      chebTransform(Cu_r, Cu_i, &(Cucorr_r[offset]),
                    &(Cucorr_i[offset]), fplan, Nz);
      chebTransform(Cv_r, Cv_i, &(Cvcorr_r[offset]),
                    &(Cvcorr_i[offset]), fplan, Nz);
      chebTransform(Cw_r, Cw_i, &(Cwcorr_r[offset]),
                    &(Cwcorr_i[offset]), fplan, Nz);
      // clean
      fftw_free(enkz);
      fftw_free(ekzmh);
      fftw_free(zekzmh);
      fftw_free(zenkz);
      fftw_free(Cp_r);  
      fftw_free(Cp_i);  
      fftw_free(Cu_r);  
      fftw_free(Cu_i);  
      fftw_free(Cv_r);  
      fftw_free(Cv_i);  
      fftw_free(Cw_r);  
      fftw_free(Cw_i); 
      fftw_free(coeffA);
      fftw_free(x); 
      fftw_free(piv);
    }
    fftw_destroy_plan(fplan);
  }
}
