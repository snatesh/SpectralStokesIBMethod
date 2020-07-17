#ifndef TRANSFORM_H
#define TRANSFORM_H
#include<fftw3.h>
#include<iostream>
#include "exceptions.h"

struct Transform
{
  double *in_real, *in_complex;
  double *out_real, *out_complex;
  fftw_plan pF, pB;
  fftw_iodim *dims, *howmany_dims;
  unsigned int Nx, Ny, Nz;
  unsigned int dof, rank; 
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

// C wrapper for calling from Python
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
