#ifndef GRID_H
#define GRID_H
#include<ostream>
#include"BoundaryConditions.h"

/* Grid is an SoA describing the domain and its data

 * fG                     - forces on the grid
 * fG_unwrap              - forces on extended grid (used internally for BCs)
 * xG, yG, zG             - grids for each axis (sorted in inc or dec order) (see below)
 * Lx, Ly, Lz, hx, hy, hz - length and grid spacing in each dimension 
 *                        - if hx > 0, xG should be Null (same for y,z)
 *                        - if hx = 0, xG must be allocated (same for y,z)
 * Nxeff, Nyeff, Nzeff    - num points in each dimension for EXTENDED grid
 * has_locator            - bool indicating whether a grid locator has been constructed
 * isperiodic             - bool array indicating whether periodicity is on or off for each axis
 * has_bc                 - bool array indicating whether BCs for each dof are specified
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
  unsigned int Nxeff, Nyeff, Nzeff;
  bool has_locator, has_bc, unifZ;
  // bool array specifying if grid is periodic in direction i
  // and another bool to make sure this array is populated
  bool isperiodic[3], has_periodicity;
  // enum for boundary conditions for each dof at the ends of each axis (dof x 6)
  BC* BCs;
  
  /* empty/null ctor */
  Grid();
  /* set up Grid based on what caller has provided */
  void setup();
  void setL(const double Lx, const double Ly, const double Lz);
  void setN(const unsigned int Nx, const unsigned int Ny, const unsigned int Nz);
  void seth(const double hx, const double hy, const double hz);
  void setZ(const double* zpts, const double* zwts);  
  void setPeriodicity(bool x, bool y, bool z);
  void setBCs(const BC* BCs);
  /* zero the extended grid */
  void zeroExtGrid();
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
  /* check validity of current state */
  bool validState() const;
  
  /* write Grid data F to ostream */
  void writeGrid(std::ostream& outputStream) const;
  void writeGrid(const char* fname) const;
  void writeCoords(std::ostream& outputStream) const;
  void writeCoords(const char* fname) const;
};


#endif
