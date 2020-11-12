#ifndef _DPTOOLS_H
#define _DPTOOLS_H

extern "C"
{
  /*
    This function is used to evaluate a Chebyshev series at a given
    value of theta (point on the cheb grid)
    
    Parameters: 
      in - the input array (size (Nz * Nyx * dof,1))
         - these are the Fourier-Chebyshev coeffs on the grid
      out - the output array (size (Nyx * dof, 1))
          - these are the Fourier-Chebyshev coeffs on the x-y
          - plane at a given z value 
      theta - determines the slice in z
            - eg) theta = pi is z = 0, theta = 0 is z - Lz 
      Nyx - total number of points in x,y
      Nz  - number of points in z
      dof - degrees of freedom
  */
  void evalTheta(const double* in, double* out, double theta, 
                 unsigned int Nyx, unsigned int Nz, unsigned int dof);

  /* 
    1D chebyshev transformation to go from function values to cheb coeffs 
    
    Parameters:
      in_re, in_im - input real/complex data
      out_re, out_im - output real/complex Chebyshev coeffs of data
      plan - a precomputed forward in-place fftw_plan for use on the in arrays
      Nyx, Nz - total number of points in x,y and num points in z 
  */ 
  void chebTransform(double* in_re, double* in_im, double* out_re, 
                     double* out_im, const fftw_plan plan, unsigned int N);

  /*
    Evaluate the analytical correction to the DP solve to enforce no-slip 
    BCs at the bottom wall
 
    Parameters: 
      C(p,u,v,w)corr_(r,i) - real and complex part of Fourier-Cheb coeffs of
                           - correction sol for  pressure and velocity
                           - these are 0 at time of call
                           - and overwritten during exectution
                           - the k = 0 element remains 0
      fhat_r, fhat_i - real and complex part of Fourier-Cheb coeffs at bottom wall
      z - Chebyshev points in z
      Kx, Ky - tensor prod of wave numbers in x,y
      eta - viscosity
      Nyx, Nz - Nyx = Ny * Nx for Ny,Nx points in x,y and Nz points in z
      dof - degrees of freedom   
  */
  void evalCorrectionSol_bottomWall(double* Cpcorr_r, double* Cpcorr_i, double* Cucorr_r, 
                                    double* Cucorr_i, double* Cvcorr_r, double* Cvcorr_i,
                                    double* Cwcorr_r, double* Cwcorr_i, const double* fhat_r,
                                    const double* fhat_i, const double* Kx, const double* Ky, 
                                    const double* z, double eta, unsigned int Nyx, 
                                    unsigned int Nz, unsigned int dof);
  /*  
    Evaluate the analytical correction to the DP solve to enforce no-slip 
    BCs at the bottom wall
 
    Parameters: 
      C(p,u,v,w)corr_(r,i) - real and complex part of Fourier-Cheb coeffs of
                           - correction sol for  pressure and velocity
                           - these are 0 at time of call
                           - and overwritten during exectution
                           - the k = 0 element remains 0
      fbhat_r, fbhat_i - real and complex part of Fourier-Cheb coeffs at bottom wall
      fthat_r, fthat_i - real and complex part of Fourier-Cheb coeffs at top wall
      z - Chebyshev points in z
      Kx, Ky - tensor prod of wave numbers in x,y
      Lz - extent of z grid
      eta - viscosity
      Nyx, Nz - Nyx = Ny * Nx for Ny,Nx points in x,y and Nz points in z
      dof - degrees of freedom   
  */
  void evalCorrectionSol_slitChannel(double* Cpcorr_r, double* Cpcorr_i, double* Cucorr_r, 
                                     double* Cucorr_i, double* Cvcorr_r, double* Cvcorr_i,
                                     double* Cwcorr_r, double* Cwcorr_i, const double* fbhat_r, 
                                     const double* fbhat_i, const double* fthat_r, const double* fthat_i, 
                                     const double* Kx, const double* Ky, const double* z, double Lz, 
                                     double eta, unsigned int Nyx, unsigned int Nz, unsigned int dof);

}
#endif
