#include "Grid.h"

/* C wrapper for calling from Python. Any functions
   defined here should also have their prototypes 
   and wrappers defined in Grid.py */
extern "C"
{
  Grid* MakeGrid()
  {
    Grid* grid = new Grid();
    return grid;
  }
  
  void SetupGrid(Grid* grid) {grid->setup();}
  //void SetL(Grid* grid, const double* Ls) {grid->setL(Ls);}
  void SetL(Grid* grid, const double Lx, const double Ly, const double Lz) 
  {
    grid->Lx = Lx; grid->Ly = Ly; grid->Lz = Lz;
  }
  void SetN(Grid* grid, const unsigned int Nx, const unsigned int Ny, 
            const unsigned int Nz) 
  {
    grid->Nx = Nx; grid->Ny = Ny; grid->Nz = Nz;
  }
  void Seth(Grid* grid, const double hx, const double hy, const double hz) 
  { 
    grid->hx = hx; grid->hy = hy; grid->hz = hz;
  }
  void SetZ(Grid* grid, const double* zpts, const double* zwts) {grid->setZ(zpts,zwts);}  
  void SetPeriodicity(Grid* grid, bool x, bool y, bool z) {grid->setPeriodicity(x,y,z);}
  void SetBCs(Grid* grid, unsigned int* BCs) {grid->setBCs(reinterpret_cast<BC*>(BCs));}
  void Setdof(Grid* grid, const unsigned int dof) {grid->dof = dof;}

  void CleanGrid(Grid* g) {g->cleanup();}
  void DeleteGrid(Grid* g) {if(g) {delete g; g = 0;}} 
  double* GetSpread(Grid* g) {return g->fG;}
  void SetSpread(Grid* g, double* f) 
  { 
    // copy
    for (unsigned int i = 0; i < g->Nx * g->Ny * g->Nz * g->dof; ++i)
    { 
      g->fG[i] = f[i];
    }   
  } 
  void WriteGrid(Grid* g, const char* fname) {g->writeGrid(fname);}
  void WriteCoords(Grid* g, const char* fname) {g->writeCoords(fname);}  
}

