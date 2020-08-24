#include<fstream>
#include<iomanip>
#include<fftw3.h>
#include "Grid.h"
#include "exceptions.h"
#include "chebyshev.h"

Grid::Grid() : fG(0), fG_unwrap(0), xG(0), yG(0), zG(0), firstn(0), 
               nextn(0), number(0), Nx(0), Ny(0), Nz(0), Lx(0), 
               Ly(0), Lz(0), hx(0), hy(0), hz(0), Nxeff(0), 
               Nyeff(0), Nzeff(0), hxeff(0), hyeff(0), hzeff(0),
               has_locator(false), xdescend(false), ydescend(false),
               zdescend(false), dof(0), zG_wts(0), has_bc(false), unifZ(false)
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
    if (hx == 0) 
    {
      xdescend = (xG[0] > xG[1] ? true : false);
      this->configAxis(hxeff, Nxeff, xG, Nx, xdescend);
    }
    else {hxeff = hx; Nxeff = Nx;}
    if (hy == 0) 
    {
      ydescend = (yG[0] > yG[1] ? true : false);
      this->configAxis(hyeff, Nyeff, yG, Ny, ydescend);
    }
    else {hyeff = hy; Nyeff = Ny;}
    if (hz == 0) 
    {
      zdescend = (zG[0] > zG[1] ? true : false);
      this->configAxis(hzeff, Nzeff, zG, Nz, zdescend);
    }
    else {hzeff = hz; Nzeff = Nz; unifZ = true;}
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

void Grid::setBCs(const BC* _BCs)
{
  for (unsigned int i = 0; i < 6; ++i)
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
  for (unsigned int i = 0; i < 6; ++i) {this->BCs[i] = periodic;}
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
  for (unsigned int i = 0; i < 4; ++i) {this->BCs[i] = periodic;}
  this->BCs[4] = this->BCs[5] = no_slip_wall;
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

void Grid::configAxis(double& heff, unsigned int& Neff, const double* axis, 
                      const unsigned int N, const bool descend)
{
  heff = 0; double dist;
  if (descend)
  {
    for (unsigned int i = 0; i < N-1; ++i)
    {
      dist = axis[i] - axis[i + 1];
      if (heff < dist) {heff = dist;}  
    }
    Neff = (unsigned int) ((axis[0] - axis[N-1]) / heff);
    heff = (axis[0] - axis[N - 1]) / Neff;
  }
  else
  {
    for (unsigned int i = 0; i < N-1; ++i)
    {
      dist = axis[i + 1] - axis[i];
      if (heff < dist) {heff = dist;}  
    }
    Neff = (unsigned int) ((axis[N - 1] - axis[0]) / heff);
    heff = (axis[N-1] - axis[0]) / Neff;
  }
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
    // uniform x
    if (hx > 0 && hy == 0 && hz == 0 && (not (yG && zG)))
    {
      throw Exception("Detected uniform grid in x, but at least one of \
                       y and z grids is null.", __func__, __FILE__, __LINE__);
    }
    // non-uniform x y z 
    if (hx == 0 && hy == 0 && hz == 0 && (not (xG && yG && zG)))
    {
      throw Exception("Detected non-uniform grids in x,y and z, but at least\
                       one of the grids is null.", __func__, __FILE__, __LINE__);
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
    // BC definition
    if (not has_bc)
    {
      throw Exception("BCs for ends of each axis are unspecified", __func__, __FILE__, __LINE__);
    }  
    else
    {
      for (unsigned int i = 0; i < 6; i += 2)
      {
        if ((BCs[i] == periodic && BCs[i + 1] != periodic) || 
            (BCs[i + 1] == periodic && BCs[i] != periodic))
        {
          throw Exception("Periodic BC must applied to both ends of axis", 
                           __func__, __FILE__, __LINE__);
        }
      }
    }
  }
  catch (Exception& e)
  {
    e.getErr();
    return false;
  }
  return true;
}
