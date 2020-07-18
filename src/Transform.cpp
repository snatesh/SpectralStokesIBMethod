#include "Transform.h"
#include<omp.h>
#include<iostream>

Transform::Transform() : in_real(0),in_complex(0),out_real(0),out_complex(0),
                         Nx(0),Ny(0),Nz(0),dof(0),rank(0) {}


// Constructs forward plan and executes - assumes input has 0 complex part
Transform::Transform(const double* _in_real, const unsigned int _Nx, 
                     const unsigned int _Ny, const unsigned int _Nz, 
                     const unsigned int _dof)
{
  Nx = _Nx; Ny = _Ny; Nz = _Nz; dof = _dof;
  // dimension of the problem TODO: generalize this
  rank = 3;

  // sign for forward transform
  mode = FFTW_FORWARD;

  // set num threads to w/e used by openmp
  fftw_plan_with_nthreads(omp_get_max_threads());
  // configure memory layout
  configDims();
  // allocate input arrays
  in_real = (double*) fftw_malloc(Nz * Ny * Nx * dof * sizeof(double));
  if (!in_real) {exitErr("alloc failed in Transform");}
  in_complex = (double*) fftw_malloc(Nz * Ny * Nx * dof * sizeof(double));  
  if (!in_complex) {exitErr("alloc failed in Transform");}
  // alias out to in for in-place transform
  out_real = in_real; out_complex = in_complex;
  // create plans for forward transform
  // we do this before populating input arrays (but only need to if not using FFTW_ESTIMATE)
  pF = fftw_plan_guru_split_dft(rank, dims, 1, howmany_dims, in_real, in_complex, out_real, out_complex, FFTW_ESTIMATE);
  if (!pF) {exitErr("FFTW forward planning failed");}
  // populate input by copy
  for (unsigned int i = 0; i < Nz * Ny * Nx * dof; ++i)
  {
    in_real[i] = _in_real[i];
    in_complex[i] = 0;
  }
  fftw_execute(pF);
}

// Constructs backward plan and executes 
Transform::Transform(const double* _out_real, const double* _out_complex,
                     const unsigned int _Nx, const unsigned int _Ny, 
                     const unsigned int _Nz, const unsigned int _dof)
{
  Nx = _Nx; Ny = _Ny; Nz = _Nz; dof = _dof;
  // dimension of the problem TODO: generalize this
  rank = 3;
  // sign for backward transform
  mode = FFTW_BACKWARD;

  // set num threads to w/e used by openmp
  fftw_plan_with_nthreads(omp_get_max_threads());
  // configure memory layout
  configDims();
  // allocate input arrays
  out_real = (double*) fftw_malloc(Nz * Ny * Nx * dof * sizeof(double));
  if (!out_real) {exitErr("alloc failed in Transform");}
  out_complex = (double*) fftw_malloc(Nz * Ny * Nx * dof * sizeof(double));  
  if (!out_complex) {exitErr("alloc failed in Transform");}
  // alias out to in for in-place transform
  in_real = out_real; in_complex = out_complex;
  // create plans for backward transform
  // MUST do this before populating in arrays
  pB = fftw_plan_guru_split_dft(rank, dims, 1, howmany_dims, out_complex, out_real, in_complex, in_real, FFTW_ESTIMATE);
  if (!pB) {exitErr("FFTW backward planning failed");}
  // populate input by copy
  for (unsigned int i = 0; i < Nz * Ny * Nx * dof; ++i)
  {
    out_real[i] = _out_real[i];
    out_complex[i] = _out_complex[i];
  }
  fftw_execute(pB);
}

void Transform::configDims()
{
  // set up iodims - we store as (k,j,i,l), l = 0:dof 
  dims = (fftw_iodim*) fftw_malloc(rank * sizeof(fftw_iodim));    
  if (!dims) {exitErr("alloc failed in configDims for Transform");}
  // we want to do 1 fft for the entire dof x 3D array     
  howmany_dims = (fftw_iodim*) fftw_malloc(1 * sizeof(fftw_iodim));
  if (!howmany_dims) {exitErr("alloc failed in configDims for Transform");}
  // size of k
  dims[0].n = Nz;
  // stride for k
  dims[0].is = dof * Nx * Ny;
  dims[0].os = dof * Nx * Ny;
  // size of j
  dims[1].n = Ny;
  // stride for j
  dims[1].is = dof * Nx;
  dims[1].os = dof * Nx;
  // size of i
  dims[2].n = Nx;
  // stride for i
  dims[2].is = dof;
  dims[2].os = dof;
  
  // dof component vec field
  howmany_dims[0].n = dof;
  // stride of 1 b/w each component (interleaved)
  howmany_dims[0].is = 1;
  howmany_dims[0].os = 1;
}

void Transform::cleanup()
{
  // destroy plans
  if (mode == FFTW_FORWARD) fftw_destroy_plan(pF);
  else if (mode == FFTW_BACKWARD) fftw_destroy_plan(pB);

  // free memory
  fftw_free(in_real);
  fftw_free(in_complex);
  fftw_free(dims);
  fftw_free(howmany_dims);
}
