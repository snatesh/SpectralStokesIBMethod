#include<fstream>
#include<iomanip>
#include<fftw3.h>
#include<omp.h>
#include"Grid.h"
#include"exceptions.h"
#include"Quadrature.h"

Grid::Grid() : fG(0), fG_unwrap(0), xG(0), yG(0), zG(0), firstn(0), 
               nextn(0), number(0), Nx(0), Ny(0), Nz(0), Lx(0), 
               Ly(0), Lz(0), hx(0), hy(0), hz(0), Nxeff(0), 
               Nyeff(0), Nzeff(0), has_locator(false), 
               dof(0), BCs(0), zG_wts(0), has_periodicity(false), 
               has_bc(false), unifZ(false)
{}

void Grid::setup()
{
  // allocate interior grid
  if (!this->fG) 
  {
    this->fG = (double*) fftw_malloc(dof * Nx * Ny * Nz * sizeof(double));
  }
  if (this->validState())
  {
    // these will be added to when the grid is extended based on kernel widths
    Nxeff = Nx; Nyeff = Ny; Nzeff = Nz;
    if (hz > 0) unifZ = true;
  }
  else {exitErr("Grid is invalid.");}
}

void Grid::setL(const double Lx, const double Ly, const double Lz)
{
  this->Lx = Lx;
  this->Ly = Ly;
  this->Lz = Lz;
}

void Grid::setN(const unsigned int Nx, const unsigned int Ny, const unsigned int Nz)
{
  this->Nx = Nx;
  this->Ny = Ny;
  this->Nz = Nz;
}

void Grid::seth(const double hx, const double hy, const double hz)
{
  this->hx = hx;
  this->hy = hy;
  this->hz = hz;
}

void Grid::setPeriodicity(bool x, bool y, bool z)
{
  this->isperiodic[0] = x; this->isperiodic[1] = y;
  this->isperiodic[2] = z; this->has_periodicity = true;
}

void Grid::setBCs(const BC* _BCs)
{
  this->BCs = (BC*) malloc(dof * 6 * sizeof(BC));
  for (unsigned int i = 0; i < 6 * dof; ++i)
  {
    this->BCs[i] = _BCs[i];
  }
  this->has_bc = true; 
}

void Grid::setZ(const double* zpts, const double* zwts)
{
  this->zG = (double*) fftw_malloc(Nz * sizeof(double));
  this->zG_wts = (double*) fftw_malloc(Nz * sizeof(double));   
  for (unsigned int i = 0; i < Nz; ++i)
  {
    this->zG[i] = zpts[i];
    this->zG_wts[i] = zwts[i];
  }
}

void Grid::zeroExtGrid()
{
  if (this->fG_unwrap)
  {
    #pragma omp parallel for
    for (unsigned int i = 0; i < Nxeff * Nyeff * Nzeff * dof; ++i)
    {
      fG_unwrap[i] = 0;
    }
  }
  else
  {
    exitErr("Extended grid has not been allocated.");
  }
}
void Grid::makeTP(const double Lx, const double Ly, const double Lz, 
                  const double hx, const double hy, const double hz,
                  const unsigned int Nx, const unsigned int Ny, 
                  const unsigned int Nz, const unsigned int dof)
{
  this->Lx = Lx; this->Ly = Ly; this->Lz = Lz;
  this->Nx = Nx; this->Ny = Ny; this->Nz = Nz;
  this->hx = hx; this->hy = hy; this->hz = hz;
  this->dof = dof;
  this->fG = (double*) fftw_malloc(dof * Nx * Ny * Nz * sizeof(double));
  this->isperiodic[0] = this->isperiodic[1] = this->isperiodic[2] = true;
  for (unsigned int i = 0; i < 6 * dof; ++i)
  {
    this->BCs[i] = none;
  }
  this->has_bc = true;
  this->setup();
}

void Grid::makeDP(const double Lx, const double Ly, const double Lz, 
                  const double hx, const double hy, const unsigned int Nx, 
                  const unsigned int Ny, const unsigned int Nz, const unsigned int dof)
{
  this->Lx = Lx; this->Ly = Ly; this->Lz = Lz;
  this->Nx = Nx; this->Ny = Ny; this->Nz = Nz;
  this->hx = hx; this->hy = hy; 
  this->dof = dof;
  this->fG = (double*) fftw_malloc(dof * Nx * Ny * Nz * sizeof(double));
  this->zG = (double*) fftw_malloc(Nz * sizeof(double));
  this->zG_wts = (double*) fftw_malloc(Nz * sizeof(double)); 
  clencurt(zG, zG_wts, 0., Lz, Nz);
  this->isperiodic[0] = this->isperiodic[1] = true; this->isperiodic[2] = false;
  for (unsigned int i = 0; i < 6 * dof; ++i)
  {
    this->BCs[i] = none;
  }
  this->has_bc = true;
  this->setup();
}

void Grid::cleanup()
{
  if (this->validState())
  {
    if (fG_unwrap) {fftw_free(fG_unwrap); fG_unwrap = 0;}
    if (firstn) {fftw_free(firstn); firstn = 0;}
    if (nextn) {fftw_free(nextn); nextn = 0;}
    if (number) {fftw_free(number); number = 0;}
    if (fG) {fftw_free(fG); fG = 0;}
    if (zG) {fftw_free(zG); zG = 0;}
    if (zG_wts) {fftw_free(zG_wts); zG_wts = 0;}
  }
  else {exitErr("Could not clean up grid.");}
}


void Grid::writeGrid(std::ostream& outputStream) const
{
  if (this->validState() && outputStream.good()) 
  {
    const int N = Nx * Ny * Nz; 
    for (unsigned int i = 0; i < N; ++i)
    {
      for (unsigned int j = 0; j < this->dof; ++j)
      {
        outputStream << std::setprecision(16) << fG[j + i * dof] << " ";
      }
      outputStream << std::endl;
    }
  }
  else
  {
    exitErr("Unable to write grid data to output stream.");
  }
}

void Grid::writeCoords(std::ostream& outputStream) const
{
  if (this->validState() && outputStream.good())
  {
    if (unifZ)
    {
      for (unsigned int k = 0; k < Nz; ++k)
      {
        for (unsigned int j = 0; j < Ny; ++j)
        {
          for (unsigned int i = 0; i < Nx; ++i)
          {
            outputStream << hx * i << " " << hy * j << " " << hz * k << std::endl;
          }
        }
      }
    }
    else
    {
      for (unsigned int k = 0; k < Nz; ++k)
      {
        for (unsigned int j = 0; j < Ny; ++j)
        {
          for (unsigned int i = 0; i < Nx; ++i)
          {
            outputStream << std::setprecision(16) << hx * i << " " << hy * j << " " << zG[k] << std::endl;
          }
        }   
      }
    }
  }
  else
  {
    exitErr("Unable to write grid coordinates to output stream.");
  }
} 

void Grid::writeGrid(const char* fname) const
{
  std::ofstream file; file.open(fname);
  writeGrid(file); file.close();
}

void Grid::writeCoords(const char* fname) const
{
  std::ofstream file; file.open(fname);
  writeCoords(file); file.close();
}

bool Grid::validState() const
{
  try
  {
    // minimal check
    if (not (fG))
    {
      throw Exception("Grid data array fG must be allocated, \
                       but at least one is null.", __func__, __FILE__, __LINE__);
    }
    // uniform x y z
    if (hx > 0 && hy > 0 && hz > 0 && (xG || yG || zG))
    {
      throw Exception("Detected uniform grids in each direction, \
                       but grids are non-null.", __func__, __FILE__, __LINE__);
    }
    // uniform x y
    if (hx > 0 && hy > 0 && hz == 0 && (not zG))
    {
      throw Exception("Detected uniform grids in x, y, but z grid is null",
                      __func__, __FILE__, __LINE__);
    }
    // ensure hx, hy are provided
    if (hx == 0 || hy == 0)
    {
      throw Exception("Grid in x, y must be uniform, and grid spacing \
                       hx, hy must be provided.", __func__, __FILE__, __LINE__);
    }
    // grid resolution 
    if (not (Nx && Ny && Nz))
    {
      throw Exception("Grid resolutions are unspecified.", __func__, __FILE__, __LINE__);
    }
    // grid extent
    if (not (Lx > 0 && Ly > 0 && Lz > 0))
    {
      throw Exception("Grid extents are unspecified.", __func__, __FILE__, __LINE__);
    }
    // dof for grid data
    if (not dof)
    {
      throw Exception("Degrees of freedom (dof) for grid data must be specified.",
                      __func__, __FILE__, __LINE__);
    }
    // periodicity specification
    if (not has_periodicity)
    {
      throw Exception("Periodicity for each axis is unspecified", __func__, __FILE__, __LINE__);
    }  
    // BC definition
    if (not has_bc)
    {
      throw Exception("BCs for ends of each axis for each dof are unspecified", __func__, __FILE__, __LINE__);
    }
  }
  catch (Exception& e)
  {
    e.getErr();
    return false;
  }
  return true;
}
