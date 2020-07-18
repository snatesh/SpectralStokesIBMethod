#include<iostream>
#include<iomanip>
#include<fstream>
#include<fftw3.h>
#include"SpreadInterp.h"
#include"SpeciesList.h"
#include"Grid.h"


int main(int argc, char* argv[])
{
  fftw_init_threads();
  // grid spacing, effective radius, num total columns
  const unsigned int Nx = 64, Ny = 64, Nz = 25, dof = 3; 
  const double hx = 0.5, hy = 0.5, hz = 0.5, Lx = Nx * hx, Ly = Ny * hy, Lz = Nz * hz; 

  Grid grid; SpeciesList species;
  grid.makeTP(Lx, Ly, Lz, hx, hy, hz, Nx, Ny, Nz, dof);
  //grid.makeDP(Lx, Ly, Lz, hx, hy, Nx, Ny, Nz, dof);
  species.randInit(grid, atoi(argv[1]));
  spread(species, grid); 

  interpolate(species, grid);

  species.writeSpecies("particles.txt");
  grid.writeGrid("spread.txt");
  grid.writeCoords("coords.txt");  
 

  species.cleanup();
  grid.cleanup();
  return 0;
}

 

