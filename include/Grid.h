#ifndef GRID_H
#define GRID_H
#include<ostream>
#include<BoundaryConditions.h>

/* Grid is an SoA describing the domain

 * fG                     - forces on the grid
 * fG_unwrap              - forces on extended grid (used internally for BCs)
 * xG, yG, zG             - grids for each axis (sorted in inc or dec order) (see below)
 * Lx, Ly, Lz, hx, hy, hz - length and grid spacing in each dimension 
 *                        - if hx > 0, xG should be Null (same for y,z)
 *                        - if hx = 0, xG must be allocated (same for y,z)
 * hxeff, hyeff, hzeff    - effective grid spacing (TODO: remove this)
 * Nxeff, Nyeff, Nzeff    - num points in each dimension for EXTENDED grid
 * has_locator            - bool indicated whether a grid locator has been constructed
 * (x,y,z)descend         - bools indicated sorting order of grids, if provided (set internally)
 * firstn, nextn          - enables the lookup of particles in terms of columns of the grid 
                            for column ind, grid.firstn[ind] = i1 is the index of the first particle in the column
                            grid.nextn[i1] = i2 is the index of the next particle in the column, and so on.
*/ 


struct Grid
{
  double *fG, *fG_unwrap, *xG, *yG, *zG, *zG_wts; 
  int *firstn, *nextn;
  unsigned int* number;
  unsigned int Nx, Ny, Nz, dof;
  double Lx, Ly, Lz;
  double hx, hy, hz;
  // these are used if grid is non-uniform
  // to define a uniform partitioning of grid into columns
  double hxeff, hyeff, hzeff;
  unsigned int Nxeff, Nyeff, Nzeff;
  bool has_locator, has_bc, unifZ;
  bool xdescend, ydescend, zdescend;
  // enum for boundary conditions at the ends of each axis
  BC BCs[6];
  
  /* empty/null ctor */
  Grid();
  /* set up Grid based on what caller has provided */
  void setup();
  void setL(const double Lx, const double Ly, const double Lz);
  void setN(const unsigned int Nx, const unsigned int Ny, const unsigned int Nz);
  void seth(const double hx, const double hy, const double hz);
  void setZ(const double* zpts, const double* zwts);  
  void setBCs(const BC* BCs);
  /* Create a valid triply periodic grid. The caller only provides these params */
  void makeTP(const double Lx, const double Ly, const double Lz, 
              const double hx, const double hy, const double hz,
              const unsigned int Nx, const unsigned int Ny, 
              const unsigned int Nz, const unsigned int dof);
  /* Create a valid doubly periodic grid. The caller only provides these params.
     By default, a Chebyshev grid will be constructed and stored in zG, zG_wts */
  void makeDP(const double Lx, const double Ly, const double Lz, 
              const double hx, const double hy, 
              const unsigned int Nx, const unsigned int Ny, 
              const unsigned int Nz, const unsigned int dof);
  /* clean memory */
  void cleanup();
  // helper called in setup to determine max spacing in non-uniform axis
  void configAxis(double& heff, unsigned int& Neff, const double* axis, 
                  const unsigned int N, const bool descend);
  /* check validity of current state */
  bool validState() const;
  
  /* write Grid data F to ostream */
  void writeGrid(std::ostream& outputStream) const;
  void writeGrid(const char* fname) const;
  void writeCoords(std::ostream& outputStream) const;
  void writeCoords(const char* fname) const;
};

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
  void SetBCs(Grid* grid, unsigned int* BCs) {grid->setBCs(reinterpret_cast<BC*>(BCs));}
  void Setdof(Grid* grid, const unsigned int dof) {grid->dof = dof;}

  void CleanGrid(Grid* g) {g->cleanup();}
  void DeleteGrid(Grid* g) {if(g) {delete g; g = 0;}} 
  double* GetGridSpread(Grid* g) {return g->fG;}
  void SetGridSpread(Grid* g, double* f) 
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

#endif
