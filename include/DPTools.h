#ifndef _DPTOOLS_H
#define _DPTOOLS_H

extern "C"
{
  /*
    This function is used to evaluate a Chebyshev series at a given
    value of theta (point on the cheb grid)
    
    Inputs: in - the input array (size (Nz * Nyx * dof,1))
            out - the output array (size (Nyx * dof, 1))
            Nyx - total number of points in x,y
            Nz  - number of points in z
            dof - degrees of freedom
  */
  void evalTheta(const double* in, double* out, double theta, 
                 unsigned int Nyx, unsigned int Nz, unsigned int dof);
  
  void chebTransform(double* in_re, double* in_im, double* out_re, 
                     double* out_im, const fftw_plan plan, unsigned int N);

  void evalCorrectionSol_bottomWall(double* Cpcorr_r, double* Cpcorr_i, double* Cucorr_r, 
                                    double* Cucorr_i, double* Cvcorr_r, double* Cvcorr_i,
                                    double* Cwcorr_r, double* Cwcorr_i, const double* fhat_r,
                                    const double* fhat_i, const double* Kx, const double* Ky, 
                                    const double* z, double eta, unsigned int Nyx, 
                                    unsigned int Nz, unsigned int dof);
    
  void evalCorrectionSol_slitChannel(double* Cpcorr_r, double* Cpcorr_i, double* Cucorr_r, 
                                     double* Cucorr_i, double* Cvcorr_r, double* Cvcorr_i,
                                     double* Cwcorr_r, double* Cwcorr_i, const double* fbhat_r, 
                                     const double* fbhat_i, const double* fthat_r, const double* fthat_i, 
                                     const double* Kx, const double* Ky, const double* z, double Lz, 
                                     double eta, unsigned int Nyx, unsigned int Nz, unsigned int dof);

}
#endif
