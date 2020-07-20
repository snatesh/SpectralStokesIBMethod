#ifndef TRANSFORM_H
#define TRANSFORM_H
#include<fftw3.h>
#include<iostream>
#include "exceptions.h"

// Forward and Backward Fourier transform object.
// Currently, only triple periodicity is supported,
// with imminent plans to add double periodicity for
// Fourier-Chebyshev transform.

struct Transform
{
  // real and complex input
  double *in_real, *in_complex;
  // real and complex output (these are aliased to input ptrs)
  double *out_real, *out_complex;
  // forward and backward plans
  fftw_plan pF, pB;
  // structs for configuring mem layout
  fftw_iodim *dims, *howmany_dims;
  unsigned int Nx, Ny, Nz;
  // degrees of freedom in the input, dimension of the problem
  // eg. if 4-component vector field in 3D, dof = 4 and rank = 3
  unsigned int dof, rank;
  // internal flag indicating whether we do a forward or back transform 
  int mode;

  Transform(); 
  // forward transform 
  Transform(const double* in_real, const unsigned int Nx, 
            const unsigned int Ny, const unsigned int Nz, 
            const unsigned int dof);
  // backwrad transform
  Transform(const double* out_real, const double* out_complex,
            const unsigned int Nx, const unsigned int Ny, 
            const unsigned int Nz, const unsigned int dof);
  // configure memory layout
  void configDims();
  void cleanup();
};

/* C wrapper for calling from Python. Any functions
   defined here should also have their prototypes 
   and wrappers defined in Transform.py */
extern "C"
{
  Transform* Ftransform(const double* in_real, const unsigned int Nx,
                        const unsigned int Ny, const unsigned int Nz,
                        const unsigned int dof) 
  {
    if (not fftw_init_threads())
    {
      exitErr("Could not initialize threads for FFTW");
    }
    return new Transform(in_real, Nx, Ny, Nz, dof);
  }
  
  Transform* Btransform(const double* out_real, const double* out_complex,
                        const unsigned int Nx, const unsigned int Ny, 
                        const unsigned int Nz, const unsigned int dof)
  {
    if (not fftw_init_threads())
    {
      exitErr("Could not initialize threads for FFTW");
    }
    return new Transform(out_real, out_complex, Nx, Ny, Nz, dof);
  }

  double* getRealOut(Transform* t) {return t->out_real;}
  double* getComplexOut(Transform* t) {return t->out_complex;}  

  void CleanTransform(Transform* t) {t->cleanup();}
  void DeleteTransform(Transform* t) {if(t) {delete t; t = 0;}}
}

#endif 
