#ifndef SPREADINTERP_H 
#define SPREADINTERP_H
#include<math.h>
#include<iomanip>
#ifdef DEBUG
  #include<iostream> 
#endif

#ifndef MEM_ALIGN
  #define MEM_ALIGN 16 
#endif

// TODO: Fix DP interpolation

// forward declarations
class Grid;
class SpeciesList;


void spread(SpeciesList& species, Grid& grid); 
void interpolate(SpeciesList& species, Grid& grid);

extern "C"
{
  double* getGridSpread(Grid* grid);
  double* getSpeciesInterp(SpeciesList* species);
  double* Spread(SpeciesList* s, Grid* g) {spread(*s, *g); return getGridSpread(g);}
  double* Interpolate(SpeciesList* s, Grid* g) {interpolate(*s, *g); return getSpeciesInterp(s);}
}

void spreadTP(SpeciesList& species, Grid& grid);
void spreadDP(SpeciesList& species, Grid& grid);
inline void spreadSP(SpeciesList& species, Grid& grid){}
inline void spreadAP(SpeciesList& species, Grid& grid){}

void interpTP(SpeciesList& species, Grid& grid);
void interpDP(SpeciesList& species, Grid& grid);
inline void interpSP(SpeciesList& species, Grid& grid){}
inline void interpAP(SpeciesList& species, Grid& grid){}

#pragma omp declare simd
inline double const esKernel(const double x, const double beta, const double alpha)
{
  return exp(beta * (sqrt(1 - x * x / (alpha * alpha)) - 1));
}

#pragma omp declare simd
inline double const esKernel(const double x[3], const double beta, const double alpha)
{
  //std::cout << "in kernel x: " << sqrt(1 - x[0] * x[0] / (alpha * alpha)) << std::endl;
  //std::cout << "in kernel y: " << sqrt(1 - x[1] * x[1] / (alpha * alpha)) << std::endl;
  //std::cout << "in kernel z: " << sqrt(1 - x[2] * x[2] / (alpha * alpha)) << std::endl;
  return exp(beta * (sqrt(1 - x[0] * x[0] / (alpha * alpha)) - 1)) * \
         exp(beta * (sqrt(1 - x[1] * x[1] / (alpha * alpha)) - 1)) * \
         exp(beta * (sqrt(1 - x[2] * x[2] / (alpha * alpha)) - 1));
}

// flattened index into 3D array
inline unsigned int const at(unsigned int i, unsigned int j,unsigned int k,\
                             const unsigned int Nx, const unsigned int Ny)
{
  return i + Nx * (j + Ny * k);
}

// gather data from src at inds into trg
template<typename T> 
inline void gather(unsigned int N, T* trg, T const* src, 
                   const unsigned int* inds, const unsigned int dof)
{
  for (unsigned int i = 0; i < N; ++i) 
  {
    for (unsigned int j = 0; j < dof; ++j)
    {
      trg[j + dof * i] = src[j + dof * inds[i]];
    }
  }
}

// scatter data from trg into src at inds
template<typename T>
inline void scatter(unsigned int N, T const* trg, T* src, 
                    const unsigned int* inds, const unsigned int dof)
{
  for (unsigned int i = 0; i < N; ++i) 
  {
    for (unsigned int j = 0; j < dof; ++j)
    {
      src[j + dof * inds[i]] = trg[j + dof * i];
    }
  }
}

inline void delta_eval_col(double* delta, const double* betafPc,
                           const unsigned short* wfPc, const double* normfPc, 
                           const double* xunwrap, const double* yunwrap, 
                           const double* zunwrap, const double alphafP, const int npts, 
                           const unsigned short wx, const unsigned short wy, 
                           const unsigned short wz, const unsigned short wfxP_max,
                           const unsigned short wfyP_max, const unsigned short wfzP_max)
{
  alignas(MEM_ALIGN) double x[3];
  #pragma omp simd aligned(delta,betafPc,normfPc,xunwrap,yunwrap,zunwrap: MEM_ALIGN), collapse(3)
  for (unsigned int k = 0; k < wz; ++k)
  {
    for (unsigned int j = 0; j < wy; ++j)
    {
      for (unsigned int i = 0; i < wx; ++i)
      {
        unsigned int m = at(i, j, k, wx, wy);
        for (unsigned int ipt = 0; ipt < npts; ++ipt)
        {
          double norm = normfPc[ipt]; norm *= norm * norm;;
          x[0] = xunwrap[i + ipt * wfxP_max];
          x[1] = yunwrap[j + ipt * wfyP_max];
          x[2] = zunwrap[k + ipt * wfzP_max];
          delta[ipt + m * npts] = esKernel(x, betafPc[ipt] * wfPc[ipt], alphafP) / norm;
        }
      }
    }
  }
}

inline void delta_eval_col(double* delta, const double* betafPc,
                           const unsigned short* wfPc, const double* normfPc, 
                           const double* xunwrap, const double* yunwrap, 
                           const double* zunwrap, const double alphafP, const int npts, 
                           const unsigned short wx, const unsigned short wy, 
                           const unsigned short* wz, const unsigned short wfxP_max,
                           const unsigned short wfyP_max, const unsigned short wfzP_max)
{
  alignas(MEM_ALIGN) double x[3];
  for (unsigned int ipt = 0; ipt < npts; ++ipt)
  {
    for (unsigned int k = 0; k < wz[ipt]; ++k)
    {
      for (unsigned int j = 0; j < wy; ++j)
      {
        for (unsigned int i = 0; i < wx; ++i)
        {
          unsigned int m = at(i, j, k, wx, wy);
          double norm = normfPc[ipt]; norm *= norm * norm;;
          x[0] = xunwrap[i + ipt * wfxP_max];
          x[1] = yunwrap[j + ipt * wfyP_max];
          x[2] = zunwrap[k + ipt * wfzP_max];
          delta[ipt + m * npts] = esKernel(x, betafPc[ipt] * wfPc[ipt], alphafP) / norm;
        }
      }
    }
  }
}

inline void spread_col(double* Fec, const double* delta, const double* flc,
                       const unsigned int* zoffset, const int npts,
                       const int w3, const int dof)
{
  for (unsigned int ipt = 0; ipt < npts; ++ipt)
  {
    for (unsigned int i = 0; i < w3; ++i)
    {
      for (unsigned int j = 0; j < dof; ++j)
      {

        Fec[j + dof * (i + zoffset[ipt])] += delta[ipt + i * npts] * flc[j + dof * ipt];
      }
    }
  }
}

inline void spread_col(double* Fec, const double* delta, const double* flc,
                       const unsigned int* zoffset, const int npts,
                       const int w2, const unsigned short* wz, const int dof)
{
  for (unsigned int ipt = 0; ipt < npts; ++ipt)
  {
    for (unsigned int i = 0; i < w2 * wz[ipt]; ++i)
    {
      for (unsigned int j = 0; j < dof; ++j)
      {
        Fec[j + dof * (i + zoffset[ipt])] += delta[ipt + i * npts] * flc[j + dof * ipt];
      }
    }
  }
}

inline void interp_col(const double* Fec, const double* delta, double* flc, 
                       const unsigned int* zoffset, const int npts, 
                       const int w3, const int dof, const double weight)
{
  for (unsigned ipt = 0; ipt < npts; ++ipt)
  {
    for (unsigned int i = 0; i < w3; ++i)
    {
      for (unsigned int j = 0; j < dof; ++j)
      { 
        flc[j + dof * ipt] += Fec[j + dof * (i + zoffset[ipt])] * 
                                delta[ipt + i * npts] * weight; 
      }
    }
  }
}


// implements copy opertion to enforce periodicity of eulerian data before interpolation
inline void copyTP(double* Fe, const double* Fe_wrap, const unsigned short wx, 
                   const unsigned short wy, const unsigned short wz, 
                   const unsigned int Nx, const unsigned int Ny,
                   const unsigned int Nz, const unsigned int dof)
{
  unsigned int lend = wx, Nx_wrap = Nx - 2 * wx;
  unsigned int rbeg = Nx - lend;
  unsigned int bend = wy, Ny_wrap = Ny - 2 * wy;
  unsigned int tbeg = Ny - bend;
  unsigned int dend = wz, Nz_wrap = Nz - 2 * wz; 
  unsigned int ubeg = Nz - dend;
  #pragma omp parallel
  {
    // copy data on wrapped grid to extended periodic grid
    #pragma omp for collapse(3)
    for (unsigned int k = 0; k < Nz_wrap; ++k)
    {
      for (unsigned int j = 0; j < Ny_wrap; ++j)
      {
        for (unsigned int i = 0; i < Nx_wrap; ++i)
        {
          unsigned int ii = i + lend, jj = j + bend, kk = k + dend;
          for (unsigned int d = 0; d < dof; ++d)
          {
            Fe[d + dof * at(ii, jj, kk, Nx, Ny)] = Fe_wrap[d + dof * at(i, j, k, Nx_wrap, Ny_wrap)];
          }
        }
      }
    }
    // copy eulerian data in y-z plane in periodic region to ghost
    #pragma omp for collapse(3)
    for (unsigned int k = dend; k < ubeg; ++k)
    {
      for (unsigned int j = bend; j < tbeg; ++j)
      {
        // first copy right to left
        for (unsigned int i = 0; i < lend; ++i)
        {
          unsigned int ipb = i + Nx_wrap;
          #pragma omp simd aligned(Fe: MEM_ALIGN)
          for (unsigned int d = 0; d < dof; ++d)
          {
            Fe[d + dof * at(i, j, k, Nx, Ny)] = Fe[d + dof * at(ipb, j, k, Nx, Ny)]; 
          }
        }  
      }
    }
    #pragma omp for collapse(3)
    for (unsigned int k = dend; k < ubeg; ++k)
    {
      for (unsigned int j = bend; j < tbeg; ++j)
      {
        // now copy left to right
        for (unsigned int i = rbeg; i < Nx; ++i)
        {
          unsigned int ipb = i - Nx_wrap;
          #pragma omp simd aligned(Fe: MEM_ALIGN)
          for (unsigned int d = 0 ; d < dof; ++d)
          {
            Fe[d + dof * at(i, j, k, Nx, Ny)] = Fe[d + dof * at(ipb, j, k, Nx, Ny)]; 
          }
        }
      }
    }

    // copy eulerian data in x-z plane in periodic region to ghost
    #pragma omp for collapse(3)
    for (unsigned int k = dend; k < ubeg; ++k)
    {
      for (unsigned int j = tbeg; j < Ny; ++j)
      {
        // first copy bottom to top
        for (unsigned int i = 0; i < Nx; ++i)
        {
          unsigned int jpb = j - Ny_wrap;
          #pragma omp simd aligned(Fe: MEM_ALIGN)
          for (unsigned int d = 0; d < dof; ++d)
          {
            Fe[d + dof * at(i, j, k, Nx, Ny)] = Fe[d + dof * at(i, jpb, k, Nx, Ny)]; 
          }
        }
      }
    }
    #pragma omp for collapse(3)
    for (unsigned int k = dend; k < ubeg; ++k)
    {
      for (unsigned int j = 0; j < bend; ++j)
      {
        // now copy top to bottom
        for (unsigned int i = 0; i < Nx; ++i)
        {
          unsigned int jpb = j + Ny_wrap;
          #pragma omp simd aligned(Fe: MEM_ALIGN)
          for (unsigned int d = 0; d < dof; ++d)
          {
            Fe[d + dof * at(i, j, k, Nx, Ny)] = Fe[d + dof * at(i, jpb, k, Nx, Ny)]; 
          }
        }  
      }
    }
    // copy eulerian data in x-y plane in periodic region to ghost
    #pragma omp for collapse(3)
    for (unsigned int k = ubeg; k < Nz; ++k)
    {
      for (unsigned int j = 0; j < Ny; ++j)
      {
        // first copy down to up
        for (unsigned int i = 0; i < Nx; ++i)
        {
          unsigned int kpb = k - Nz_wrap;
          #pragma omp simd aligned(Fe: MEM_ALIGN)
          for (unsigned int d = 0; d < dof; ++d)
          {
            Fe[d + dof * at(i, j, k, Nx, Ny)] = Fe[d + dof * at(i, j, kpb, Nx, Ny)]; 
          }
        }
      }
    }
    #pragma omp for collapse(3)
    for (unsigned int k = 0; k < dend; ++k)
    {
      for (unsigned int j = 0; j < Ny; ++j)
      {
        // now copy up to down
        for (unsigned int i = 0; i < Nx; ++i)
        {
          unsigned int kpb = k + Nz_wrap;
          #pragma omp simd aligned(Fe: MEM_ALIGN)
          for (unsigned int d = 0; d < dof; ++d)
          {
            Fe[d + dof * at(i, j, k, Nx, Ny)] = Fe[d + dof * at(i, j, kpb, Nx, Ny)]; 
          }
        }
      }
    }
  }
}

// implements fold operation to de-ghostify spread data, i.e. enable periodic spread
inline void foldTP(double* Fe, double* Fe_wrap, const unsigned short wx, 
                   const unsigned short wy, const unsigned short wz, 
                   const unsigned int Nx, const unsigned int Ny,
                   const unsigned int Nz, const unsigned int dof)
{
  unsigned int lend = wx, Nx_wrap = Nx - 2 * wx;
  unsigned int rbeg = Nx - lend;
  unsigned int bend = wy, Ny_wrap = Ny - 2 * wy;
  unsigned int tbeg = Ny - bend;
  unsigned int dend = wz, Nz_wrap = Nz - 2 * wz; 
  unsigned int ubeg = Nz - dend;
  #pragma omp parallel
  {
    // fold eulerian data in y-z plane in ghost region to periodic index
    #pragma omp for collapse(3)
    for (unsigned int k = 0; k < Nz; ++k)
    {
      for (unsigned int j = 0; j < Ny; ++j)
      {
        // first do left
        for (unsigned int i = 0; i < lend; ++i)
        {
          unsigned int ipb = i + Nx_wrap;
          #pragma omp simd aligned(Fe: MEM_ALIGN)
          for (unsigned int d = 0; d < dof; ++d)
          { 
            Fe[d + dof * at(ipb, j, k, Nx, Ny)] += Fe[d + dof * at(i, j, k, Nx, Ny)]; 
          }
        }
      }
    }
    #pragma omp for collapse(3)
    for (unsigned int k = 0; k < Nz; ++k)
    {
      for (unsigned int j = 0; j < Ny; ++j)
      {
        // now do right
        for (unsigned int i = rbeg; i < Nx; ++i)
        {
          unsigned int ipb = i - Nx_wrap;
          #pragma omp simd aligned(Fe: MEM_ALIGN)
          for (unsigned int d = 0; d < dof; ++d)
          { 
            Fe[d + dof * at(ipb, j, k, Nx, Ny)] += Fe[d + dof * at(i, j, k, Nx, Ny)]; 
          }
        }  
      }
    }
    // fold eulerian data in x-z plane in ghost region to periodic index
    #pragma omp for collapse(3)
    for (unsigned int k = 0; k < Nz; ++k)
    {
      // first do bottom
      for (unsigned int j = 0; j < bend; ++j)
      {
        for (unsigned int i = 0; i < Nx; ++i)
        {
          unsigned int jpb = j + Ny_wrap;
          #pragma omp simd aligned(Fe: MEM_ALIGN)
          for (unsigned int d = 0; d < dof; ++d)
          { 
            Fe[d + dof * at(i, jpb, k, Nx, Ny)] += Fe[d + dof * at(i, j, k, Nx, Ny)]; 
          }
        }
      } 
    } 
    #pragma omp for collapse(3)
    for (unsigned int k = 0; k < Nz; ++k)
    {
      // now do top
      for (unsigned int j = tbeg; j < Ny; ++j)
      {
        for (unsigned int i = 0; i < Nx; ++i)
        {
          unsigned int jpb = j - Ny_wrap;
          #pragma omp simd aligned(Fe: MEM_ALIGN)
          for (unsigned int d = 0; d < dof; ++d)
          { 
            Fe[d + dof * at(i, jpb, k, Nx, Ny)] += Fe[d + dof * at(i, j, k, Nx, Ny)]; 
          }
        }
      }
    }
    // fold eulerian data in x-y plane in ghost region to periodic index
    // first do down
    #pragma omp for collapse(3)
    for (unsigned int k = 0; k < dend; ++k)
    {
      for (unsigned int j = 0; j < Ny; ++j)
      {
        for (unsigned int i = 0; i < Nx; ++i)
        {
          unsigned int kpb = k + Nz_wrap;
          #pragma omp simd aligned(Fe: MEM_ALIGN)
          for (unsigned int d = 0; d < dof; ++d)
          { 
            Fe[d + dof * at(i, j, kpb, Nx, Ny)] += Fe[d + dof * at(i, j, k, Nx, Ny)]; 
          }
        }
      } 
    } 
  
    // now do up
    #pragma omp for collapse(3)
    for (unsigned int k = ubeg; k < Nz; ++k)
    {
      for (unsigned int j = 0; j < Ny; ++j)
      {
        for (unsigned int i = 0; i < Nx; ++i)
        {
          unsigned int kpb = k - Nz_wrap;
          #pragma omp simd aligned(Fe: MEM_ALIGN)
          for (unsigned int d = 0; d < dof; ++d)
          { 
            Fe[d + dof * at(i, j, kpb, Nx, Ny)] += Fe[d + dof * at(i, j, k, Nx, Ny)]; 
          }
        }
      }
    }
    // copy data on extended periodic grid to wrapped grid
    #pragma omp for collapse(3)
    for (unsigned int k = dend; k < ubeg; ++k) 
    {
      for (unsigned int j = bend; j < tbeg; ++j)
      {
        for (unsigned int i = lend; i < rbeg; ++i)
        {
          unsigned int ii = i - lend, jj = j - bend, kk = k - dend;
          #pragma omp simd aligned(Fe, Fe_wrap: MEM_ALIGN)
          for (unsigned int d = 0; d < dof; ++d)
          { 
            Fe_wrap[d + dof * at(ii, jj, kk, Nx_wrap, Ny_wrap)] 
              = Fe[d + dof * at(i, j, k, Nx, Ny)];
          }
        }
      }
    }
  }
}

// implements fold operation to de-ghostify spread data, i.e. enable periodic spread
// for doubly-periodic geometry with z a-periodic
inline void foldDP(double* Fe, double* Fe_wrap, const unsigned short wx, 
                   const unsigned short wy, const unsigned int ext_up, 
                   const unsigned int ext_down, const unsigned int Nx, 
                   const unsigned int Ny, const unsigned int Nz, const unsigned int dof)
{
  unsigned int lend = wx, Nx_wrap = Nx - 2 * wx;
  unsigned int rbeg = Nx - lend;
  unsigned int bend = wy, Ny_wrap = Ny - 2 * wy;
  unsigned int tbeg = Ny - bend;
  unsigned int dend = ext_up, Nz_wrap = Nz - ext_up - ext_down; 
  unsigned int ubeg = Nz - ext_down;
  #pragma omp parallel
  {
    // fold eulerian data in y-z plane in ghost region to periodic index
    #pragma omp for collapse(3)
    for (unsigned int k = 0; k < Nz; ++k)
    {
      for (unsigned int j = 0; j < Ny; ++j)
      {
        // first do left
        for (unsigned int i = 0; i < lend; ++i)
        {
          unsigned int ipb = i + Nx_wrap;
          #pragma omp simd aligned(Fe: MEM_ALIGN)
          for (unsigned int d = 0; d < dof; ++d)
          { 
            Fe[d + dof * at(ipb, j, k, Nx, Ny)] += Fe[d + dof * at(i, j, k, Nx, Ny)]; 
          }
        }
      }
    }
    #pragma omp for collapse(3)
    for (unsigned int k = 0; k < Nz; ++k)
    {
      for (unsigned int j = 0; j < Ny; ++j)
      {
        // now do right
        for (unsigned int i = rbeg; i < Nx; ++i)
        {
          unsigned int ipb = i - Nx_wrap;
          #pragma omp simd aligned(Fe: MEM_ALIGN)
          for (unsigned int d = 0; d < dof; ++d)
          { 
            Fe[d + dof * at(ipb, j, k, Nx, Ny)] += Fe[d + dof * at(i, j, k, Nx, Ny)]; 
          }
        }  
      }
    }
    // fold eulerian data in x-z plane in ghost region to periodic index
    #pragma omp for collapse(3)
    for (unsigned int k = 0; k < Nz; ++k)
    {
      // first do bottom
      for (unsigned int j = 0; j < bend; ++j)
      {
        for (unsigned int i = 0; i < Nx; ++i)
        {
          unsigned int jpb = j + Ny_wrap;
          #pragma omp simd aligned(Fe: MEM_ALIGN)
          for (unsigned int d = 0; d < dof; ++d)
          { 
            Fe[d + dof * at(i, jpb, k, Nx, Ny)] += Fe[d + dof * at(i, j, k, Nx, Ny)]; 
          }
        }
      } 
    } 
    #pragma omp for collapse(3)
    for (unsigned int k = 0; k < Nz; ++k)
    {
      // now do top
      for (unsigned int j = tbeg; j < Ny; ++j)
      {
        for (unsigned int i = 0; i < Nx; ++i)
        {
          unsigned int jpb = j - Ny_wrap;
          #pragma omp simd aligned(Fe: MEM_ALIGN)
          for (unsigned int d = 0; d < dof; ++d)
          { 
            Fe[d + dof * at(i, jpb, k, Nx, Ny)] += Fe[d + dof * at(i, j, k, Nx, Ny)]; 
          }
        }
      }
    }
    // fold NEGATIVE of eulerian data in x-y plane in ghost region to adjacent interior region
    // first do down
    #pragma omp for collapse(3)
    for (unsigned int k = 0; k <= dend; ++k)
    {
      for (unsigned int j = 0; j < Ny; ++j)
      {
        for (unsigned int i = 0; i < Nx; ++i)
        {
          unsigned int kpb = 2 * dend - k;
          #pragma omp simd aligned(Fe: MEM_ALIGN)
          for (unsigned int d = 0; d < dof; ++d)
          { 
            Fe[d + dof * at(i, j, kpb, Nx, Ny)] -= Fe[d + dof * at(i, j, k, Nx, Ny)]; 
          }
        }
      } 
    }
  
    // now do up
    #pragma omp for collapse(3)
    for (unsigned int k = ubeg - 1; k < Nz; ++k)
    {
      for (unsigned int j = 0; j < Ny; ++j)
      {
        for (unsigned int i = 0; i < Nx; ++i)
        {
          unsigned int kpb = 2 * ubeg - k - 2;
          #pragma omp simd aligned(Fe: MEM_ALIGN)
          for (unsigned int d = 0; d < dof; ++d)
          { 
            Fe[d + dof * at(i, j, kpb, Nx, Ny)] -= Fe[d + dof * at(i, j, k, Nx, Ny)]; 
          }
        }
      }
    }
    // copy data on extended periodic grid to wrapped grid
    #pragma omp for collapse(3)
    for (unsigned int k = dend; k < ubeg; ++k) 
    {
      for (unsigned int j = bend; j < tbeg; ++j)
      {
        for (unsigned int i = lend; i < rbeg; ++i)
        {
          unsigned int ii = i - lend, jj = j - bend, kk = k - dend;
          #pragma omp simd aligned(Fe, Fe_wrap: MEM_ALIGN)
          for (unsigned int d = 0; d < dof; ++d)
          { 
            Fe_wrap[d + dof * at(ii, jj, kk, Nx_wrap, Ny_wrap)] 
              = Fe[d + dof * at(i, j, k, Nx, Ny)];
          }
        }
      }
    }
  }
}

// implements copy opertion to enforce periodicity of eulerian data before interpolation
inline void copyDP(double* Fe, double* Fe_wrap, const unsigned short wx, 
                   const unsigned short wy, const unsigned int ext_up, 
                   const unsigned int ext_down, const unsigned int Nx, 
                   const unsigned int Ny, const unsigned int Nz, const unsigned int dof)
{
  unsigned int lend = wx, Nx_wrap = Nx - 2 * wx;
  unsigned int rbeg = Nx - lend;
  unsigned int bend = wy, Ny_wrap = Ny - 2 * wy;
  unsigned int tbeg = Ny - bend;
  unsigned int dend = ext_up, Nz_wrap = Nz - ext_up - ext_down; 
  unsigned int ubeg = Nz - ext_down;
  #pragma omp parallel
  {
    // copy data on wrapped grid to extended periodic grid
    #pragma omp for collapse(3)
    for (unsigned int k = 0; k < Nz_wrap; ++k)
    {
      for (unsigned int j = 0; j < Ny_wrap; ++j)
      {
        for (unsigned int i = 0; i < Nx_wrap; ++i)
        {
          unsigned int ii = i + lend, jj = j + bend, kk = k + dend;
          for (unsigned int d = 0; d < dof; ++d)
          {
            Fe[d + dof * at(ii, jj, kk, Nx, Ny)] = Fe_wrap[d + dof * at(i, j, k, Nx_wrap, Ny_wrap)];
          }
        }
      }
    }
  
    // copy eulerian data in y-z plane in periodic region to ghost
    #pragma omp for collapse(3)
    for (unsigned int k = 0; k < Nz; ++k)
    {
      for (unsigned int j = 0; j < Ny; ++j)
      {
        // first copy left to right
        for (unsigned int i = rbeg; i < Nx; ++i)
        {
          unsigned int ipb = i - Nx_wrap;
          #pragma omp simd aligned(Fe: MEM_ALIGN)
          for (unsigned int d = 0 ; d < dof; ++d)
          {
            Fe[d + dof * at(i, j, k, Nx, Ny)] = Fe[d + dof * at(ipb, j, k, Nx, Ny)]; 
          }
        }
      }
    }
    #pragma omp for collapse(3)
    for (unsigned int k = 0; k < Nz; ++k)
    {
      for (unsigned int j = 0; j < Ny; ++j)
      {
        // now copy right to left
        for (unsigned int i = 0; i < lend; ++i)
        {
          unsigned int ipb = i + Nx_wrap;
          #pragma omp simd aligned(Fe: MEM_ALIGN)
          for (unsigned int d = 0 ; d < dof; ++d)
          {
            Fe[d + dof * at(i, j, k, Nx, Ny)] = Fe[d + dof * at(ipb, j, k, Nx, Ny)]; 
          }
        }  
      }
    }
    // copy eulerian data in x-z plane in periodic region to ghost
    #pragma omp for collapse(3)
    for (unsigned int k = 0; k < Nz; ++k)
    {
      for (unsigned int j = tbeg; j < Ny; ++j)
      {
        // first copy bottom to top
        for (unsigned int i = 0; i < Nx; ++i)
        {
          unsigned int jpb = j - Ny_wrap;
          #pragma omp simd aligned(Fe: MEM_ALIGN)
          for (unsigned int d = 0; d < dof; ++d)
          {
            Fe[d + dof * at(i, j, k, Nx, Ny)] = Fe[d + dof * at(i, jpb, k, Nx, Ny)]; 
          }
        }
      }
    }
    #pragma omp for collapse(3)
    for (unsigned int k = 0; k < Nz; ++k)
    {
      for (unsigned int j = 0; j < bend; ++j)
      {
        // now copy top to bottom
        for (unsigned int i = 0; i < Nx; ++i)
        {
          unsigned int jpb = j + Ny_wrap;
          #pragma omp simd aligned(Fe: MEM_ALIGN)
          for (unsigned int d = 0; d < dof; ++d)
          {
            Fe[d + dof * at(i, j, k, Nx, Ny)] = Fe[d + dof * at(i, jpb, k, Nx, Ny)]; 
          }
        }  
      }
    }

    // copy NEGATIVE of eulerian data in x-y plane in adjacent interior region to ghost
    #pragma omp for collapse(3)
    for (unsigned int k = ubeg - 1; k < Nz; ++k)
    {
      for (unsigned int j = 0; j < Ny; ++j)
      {
        for (unsigned int i = 0; i < Nx; ++i)
        {
          unsigned int kpb = 2 * ubeg - k - 2;
          #pragma omp simd aligned(Fe: MEM_ALIGN)
          for (unsigned int d = 0; d < dof; ++d)
          { 
            Fe[d + dof * at(i, j, kpb, Nx, Ny)] = -1.0 * Fe[d + dof * at(i, j, k, Nx, Ny)]; 
          }
        }
      }
    }
    #pragma omp for collapse(3)
    for (unsigned int k = 0; k <= dend; ++k)
    {
      for (unsigned int j = 0; j < Ny; ++j)
      {
        for (unsigned int i = 0; i < Nx; ++i)
        {
          unsigned int kpb = 2 * dend - k;
          #pragma omp simd aligned(Fe: MEM_ALIGN)
          for (unsigned int d = 0; d < dof; ++d)
          { 
            Fe[d + dof * at(i, j, kpb, Nx, Ny)] = -1.0 * Fe[d + dof * at(i, j, k, Nx, Ny)]; 
          }
        }
      } 
    }
  }
}



#endif
