#include "Transform.h"

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

