#include "Transform.h"
#include "SpreadInterp.h"
#include "SpeciesList.h"
#include "Grid.h"
#include<iostream>
#include<math.h>

// testing complex to complex forward and backward in-place transforms
// for 3D vector field stored row-major as (k,j,i,l), l = 0:2
// We also check OpenMP integration
int main(int argc, char* argv[])
{
  // initialize threads for fftw
  fftw_init_threads();
  // num points, grid spacing, box size
  const unsigned int Nx = 64, Ny = 64, Nz = 64, dof = 3; 
  const double hx = 0.5, hy = 0.5, hz = 0.5, Lx = Nx * hx, Ly = Ny * hy, Lz = Nz * hz; 

  Grid grid; SpeciesList species;
  grid.makeTP(Lx, Ly, Lz, hx, hy, hz, Nx, Ny, Nz, dof);
  species.randInit(grid, atoi(argv[1]));
  spread(species, grid); 

  Transform forward(grid.fG,Nx,Ny,Nz,dof);
  Transform backward(forward.out_real,forward.out_complex,Nx,Ny,Nz,dof); 
  
  // make sure result is the same as initial input
  double maxerr,err; maxerr = 0;
  unsigned int N = Nx * Ny * Nz;
  for (unsigned int i = 0; i < Nx * Ny * Nz * 3; ++i)
  {
    // compare difference (note we normalize by N)
    err = fabs(backward.out_real[i]/N - grid.fG[i]);
    maxerr = (maxerr >= err ? maxerr : err);
  }
  std::cout << "Max error = " << maxerr << std::endl;
 
  forward.cleanup();
  backward.cleanup();
  species.cleanup();
  grid.cleanup(); 
  return 0;
}
