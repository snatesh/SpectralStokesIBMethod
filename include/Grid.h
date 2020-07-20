#ifndef GRID_H
#define GRID_H
#include<ostream>

/* Grid is an SoA describing the domain

 * fG                     - forces on the grid
 * fG_unwrap              - forces on extended grid (used internally for BCs)
 * xG, yG, zG             - grids for each axis (sorted in inc or dec order) (see below)
 * Lx, Ly, Lz, hx, hy, hz - length and grid spacing in each dimension 
 *                        - if hx > 0, xG should be Null (same for y,z)
 *                        - if hx = 0, xG must be allocated (same for y,z)
 * periodicity            - triply- (3), doubly- (2), singly- (1) or a- (0) periodic mode
 * hxeff, hyeff, hzeff    - effective grid spacing (TODO: remove this)
 * Nxeff, Nyeff, Nzeff    - num points in each dimension for EXTENDED grid
 * has_locator            - bool indicated whether a grid locator has been constructed
 * (x,y,z)descend         - bools indicated sorting order of grids, if provided (set internally)
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
  int periodicity;
  bool has_locator;
  bool xdescend, ydescend, zdescend;

  /* empty/null ctor */
  Grid();
  /* set up Grid based on what caller has provided */
  void setup();
  void makeTP(const double Lx, const double Ly, const double Lz, 
              const double hx, const double hy, const double hz,
              const unsigned int Nx, const unsigned int Ny, 
              const unsigned int Nz, const unsigned int dof);
  void makeDP(const double Lx, const double Ly, const double Lz, 
              const double hx, const double hy, 
              const unsigned int Nx, const unsigned int Ny, 
              const unsigned int Nz, const unsigned int dof);
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
  Grid* MakeGrid(const double Lx, const double Ly, const double Lz,
                 const double hx, const double hy, const double hz,
                 const unsigned int Nx, const unsigned int Ny,
                 const unsigned int Nz, const unsigned int dof,
                 const unsigned int periodicity)
  {
    Grid* grid = new Grid();
    if (periodicity == 3) {grid->makeTP(Lx, Ly, Lz, hx, hy, hz, Nx, Ny, Nz, dof);}
    else if (periodicity == 2) {grid->makeDP(Lx, Ly, Lz, hx, hy, Nx, Ny, Nz, dof);}
    return grid; 
  }

  void CleanGrid(Grid* g) {g->cleanup();}
  void DeleteGrid(Grid* g) {if(g) {delete g; g = 0;}} 
  double* getGridSpread(Grid* g) {return g->fG;}
  void setGridSpread(Grid* g, double* f) 
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
