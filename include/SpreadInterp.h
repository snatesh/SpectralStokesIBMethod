#ifndef SPREADINTERP_H 
#define SPREADINTERP_H
#include<math.h>
#include<iomanip>
#include<BoundaryConditions.h>
#ifdef DEBUG
  #include<iostream> 
#endif

#ifndef MEM_ALIGN
  #define MEM_ALIGN 16 
#endif

/* Main routines for spreading and interpolation, with
   the appropriate routines dispatched based on
   specified boundary conditions */


// forward declarations
struct Grid;
struct ParticleList;

// spread and interpolate
void spread(ParticleList& particles, Grid& grid); 
void interpolate(ParticleList& particles, Grid& grid);

// spread with z uniform or not
void spreadUnifZ(ParticleList& particles, Grid& grid);
void spreadNonUnifZ(ParticleList& particles, Grid& grid);
// interpolate with z uniform or not
void interpUnifZ(ParticleList& particles, Grid& grid);
void interpNonUnifZ(ParticleList& particles, Grid& grid);

// ES kernel definition (two versions for optimization testing)
#pragma omp declare simd
inline double const esKernel(const double x, const double beta, const double alpha)
{
  return exp(beta * (sqrt(1 - x * x / (alpha * alpha)) - 1));
}

#pragma omp declare simd
inline double const esKernel(const double x[3], const double beta, const double alpha)
{
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

// evaluate the delta function weights for the current column for UnifZ = True
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

// evaluate the delta function weights for the current column for UnifZ = false
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

// spread the delta functions weights for the column for UnifZ = true
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

// spread with forces and weights for the column for UnifZ = false
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

// interpolate with the forces and weights for the current column for UNIFORM Z
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


// interpolate with the forces and weights for the current column for NON-UNIFORM Z
inline void interp_col(const double* Fec, const double* delta, double* flc, 
                       const unsigned int* zoffset, const int npts, 
                       const unsigned short wx, const unsigned short wy, 
                       const unsigned short* wz, const unsigned short wfzP_max, 
                       const int dof, const double* weight)
{
  for (unsigned ipt = 0; ipt < npts; ++ipt)
  {
    for (unsigned int k = 0; k < wz[ipt]; ++k)
    {
      for (unsigned int j = 0; j < wy; ++j)
      {
        for (unsigned int i = 0; i < wx; ++i)
        {
          unsigned int m = at(i, j, k, wx, wy);
          for (unsigned int d = 0; d < dof; ++d)
          {
            flc[d + dof * ipt] += Fec[d + dof * (m + zoffset[ipt])] * 
                                    delta[ipt + m * npts] * weight[k + ipt * wfzP_max];
          } 
        }
      }
    }
  }
}

// implements fold operation to de-ghostify spread data according to BCs
inline void fold(double* Fe, double* Fe_wrap, const unsigned short wx, 
                 const unsigned short wy, const unsigned short ext_up, 
                 const unsigned short ext_down, const unsigned int Nx, 
                 const unsigned int Ny, const unsigned int Nz, 
                 const unsigned int dof, bool* periodic, const BC* BCs)
{
  unsigned int lend = wx, Nx_wrap = Nx - 2 * wx;
  unsigned int rbeg = Nx - lend;
  unsigned int bend = wy, Ny_wrap = Ny - 2 * wy;
  unsigned int tbeg = Ny - bend;
  unsigned int dend = ext_up, Nz_wrap = Nz - ext_up - ext_down; 
  unsigned int ubeg = Nz - ext_down;
  // periodic fold in x 
  if (periodic[0])
  {
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
    }
  }
  // handle BC for each end of x as specified
  else
  {
    // get bc for left end of x 
    const BC* bc_xl = &(BCs[0]);
    // multiplier to enforce bc
    double s;
    for (unsigned int d = 0; d < dof; ++d)
    {
      // we only do something if bc is not none
      if (bc_xl[d] != none)
      { 

        if (bc_xl[d] == mirror) s = 1.0;
        if (bc_xl[d] == mirror_inv) s = -1.0;
        // fold eulerian data in y-z plane in ghost region to adjacent interior region
        // at left end of x axis according to mirror or mirror_inv
        #pragma omp parallel for collapse(3)
        for (unsigned int k = 0; k < Nz; ++k)
        {
          for (unsigned int j = 0; j < Ny; ++j)
          {
            for (unsigned int i = 0; i <= lend; ++i)
            {
              unsigned int ipb = 2 * lend - i;
              Fe[d + dof * at(ipb, j, k, Nx, Ny)] += s * Fe[d + dof * at(i, j, k, Nx, Ny)]; 
            }
          }
        }
      }
    }
    // get bc for right end of x
    const BC* bc_xr = &(BCs[dof]); 
    // no slip wall on x right
    for (unsigned int d = 0; d < dof; ++d)
    {
      // we only do something if bc is not none
      if (bc_xr[d] != none)
      { 
        if (bc_xr[d] == mirror) s = 1.0;
        if (bc_xr[d] == mirror_inv) s = -1.0;
        // fold eulerian data in y-z plane in ghost region to adjacent interior region
        // at right end of x axis according to mirror or mirror_inv
        #pragma omp parallel for collapse(3)
        for (unsigned int k = 0; k < Nz; ++k)
        {
          for (unsigned int j = 0; j < Ny; ++j)
          {
            for (unsigned int i = rbeg - 1; i < Nx; ++i)
            {
              unsigned int ipb = 2 * rbeg - i - 2;
              Fe[d + dof * at(ipb, j, k, Nx, Ny)] -= Fe[d + dof * at(i, j, k, Nx, Ny)]; 
            }
          }
        }
      }
    }
  }
  // periodic fold in y
  if (periodic[1])
  {
    #pragma omp parallel
    {
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
    }
  }
  // handle BC for each end of y as specified
  else
  {
    // get bc for left end of y 
    const BC* bc_yl = &(BCs[2 * dof]);
    // multiplier to enforce bc
    double s;
    for (unsigned int d = 0; d < dof; ++d)
    {
      // we only do something if bc is not none
      if (bc_yl[d] != none)
      { 
        if (bc_yl[d] == mirror) s = 1.0;
        if (bc_yl[d] == mirror_inv) s = -1.0;
        // fold eulerian data in x-z plane in ghost region to adjacent interior region
        // at bottom end of y axis according to mirror or mirror_inv
        #pragma omp parallel for collapse(3)
        for (unsigned int k = 0; k < Nz; ++k)
        {
          for (unsigned int j = 0; j <= bend; ++j)
          {
            for (unsigned int i = 0; i < Nx; ++i)
            {
              unsigned int jpb = 2 * bend - j;
              Fe[d + dof * at(i, jpb, k, Nx, Ny)] += s * Fe[d + dof * at(i, j, k, Nx, Ny)]; 
            }
          }
        }
      }
    }
    // get bc for right end of y 
    const BC* bc_yr = &(BCs[3 * dof]);
    for (unsigned int d = 0; d < dof; ++d)
    {
      // we only do something if bc is not none
      if (bc_yr[d] != none)
      { 
        if (bc_yr[d] == mirror) s = 1.0;
        if (bc_yr[d] == mirror_inv) s = -1.0;
        // fold eulerian data in x-z plane in ghost region to adjacent interior region
        // at top end of y axis according to mirror or mirror_inv
        #pragma omp parallel for collapse(3)
        for (unsigned int k = 0; k < Nz; ++k)
        {
          for (unsigned int j = tbeg - 1; j < Ny; ++j)
          {
            for (unsigned int i = 0; i < Nx; ++i)
            {
              unsigned int jpb = 2 * tbeg - i - 2;
              Fe[d + dof * at(i, jpb, k, Nx, Ny)] += s * Fe[d + dof * at(i, j, k, Nx, Ny)]; 
            }
          }
        }
      }
    }
  }
  // periodic fold in z
  if (periodic[2])
  {
    #pragma omp parallel
    {
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
    }
  }
  // handle BC for each end of z as specified
  else
  {
    // get bc for left end of z 
    const BC* bc_zl = &(BCs[4 * dof]);
    // multiplier to enforce bc
    double s;
    for (unsigned int d = 0; d < dof; ++d)
    {
      // we only do something if bc is not none
      if (bc_zl[d] != none)
      { 
        if (bc_zl[d] == mirror) s = 1.0;
        if (bc_zl[d] == mirror_inv) s = -1.0;
        // fold eulerian data in x-y plane in ghost region to adjacent interior region
        // at lower end of z axis according to mirror or mirror_inv
        #pragma omp parallel for collapse(3)
        for (unsigned int k = 0; k <= dend; ++k)
        {
          for (unsigned int j = 0; j < Ny; ++j)
          {
            for (unsigned int i = 0; i < Nx; ++i)
            {
              unsigned int kpb = 2 * dend - k;
              Fe[d + dof * at(i, j, kpb, Nx, Ny)] += s * Fe[d + dof * at(i, j, k, Nx, Ny)]; 
            }
          } 
        }
      }
    }
    // get bc for right end of z
    const BC* bc_zr = &(BCs[5 * dof]);  
    for (unsigned int d = 0; d < dof; ++d)
    {
      // we only do something if bc is not none
      if (bc_zr[d] != none)
      { 
        if (bc_zr[d] == mirror) s = 1.0;
        if (bc_zr[d] == mirror_inv) s = -1.0;
        // fold eulerian data in x-y plane in ghost region to adjacent interior region
        // at upper end of z axis according to mirror or mirror_inv
        #pragma omp parallel for collapse(3)
        for (unsigned int k = ubeg - 1; k < Nz; ++k)
        {
          for (unsigned int j = 0; j < Ny; ++j)
          {
            for (unsigned int i = 0; i < Nx; ++i)
            {
              unsigned int kpb = 2 * ubeg - k - 2;
              Fe[d + dof * at(i, j, kpb, Nx, Ny)] += s * Fe[d + dof * at(i, j, k, Nx, Ny)]; 
            }
          }
        }
      }
    }
  }
  // copy data on extended grid to wrapped grid
  #pragma omp parallel for collapse(3)
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

// implements copy opertion to enforce periodicity of eulerian data before interpolation
inline void copy(double* Fe, const double* Fe_wrap, const unsigned short wx, 
                 const unsigned short wy, const unsigned short ext_up,
                 const unsigned short ext_down, const unsigned int Nx, 
                 const unsigned int Ny, const unsigned int Nz, 
                 const unsigned int dof, bool* periodic, const BC* BCs)
{
  unsigned int lend = wx, Nx_wrap = Nx - 2 * wx;
  unsigned int rbeg = Nx - lend;
  unsigned int bend = wy, Ny_wrap = Ny - 2 * wy;
  unsigned int tbeg = Ny - bend;
  unsigned int dend = ext_up, Nz_wrap = Nz - ext_up - ext_down; 
  unsigned int ubeg = Nz - ext_down;
  // copy data on wrapped grid to extended periodic grid
  #pragma omp parallel for collapse(3)
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
  // periodic copy in x
  if (periodic[0])
  {
    #pragma omp parallel
    {  
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
    }
  }
  else
  {
    // get bc for left end of x 
    const BC* bc_xl = &(BCs[0]);
    // multiplier to enforce bc
    double s;
    for (unsigned int d = 0; d < dof; ++d)
    {
      // we only do something if bc is not none
      if (bc_xl[d] != none)
      { 

        if (bc_xl[d] == mirror) s = 1.0;
        if (bc_xl[d] == mirror_inv) s = -1.0;
        // copy eulerian data in y-z plane in interior region to 
        // adjacent ghost region at left end of x axis according to mirror or mirror_inv
        #pragma omp parallel for collapse(3)
        for (unsigned int k = dend; k < ubeg; ++k)
        {
          for (unsigned int j = bend; j < tbeg; ++j)
          {
            for (unsigned int i = 0; i < lend; ++i)
            {
              unsigned int ipb = 2 * lend - i;
              Fe[d + dof * at(i, j, k, Nx, Ny)] = s * Fe[d + dof * at(ipb, j, k, Nx, Ny)]; 
            }
          }
        }
      }
    }
    // get bc for right end of x
    const BC* bc_xr = &(BCs[dof]); 
    // no slip wall on x right
    for (unsigned int d = 0; d < dof; ++d)
    {
      // we only do something if bc is not none
      if (bc_xr[d] != none)
      { 
        if (bc_xr[d] == mirror) s = 1.0;
        if (bc_xr[d] == mirror_inv) s = -1.0;
        // copy eulerian data in y-z plane in interior region to 
        // adjacent ghost region at right end of x axis according to mirror or mirror_inv
        #pragma omp parallel for collapse(3)
        for (unsigned int k = dend; k < ubeg; ++k)
        {
          for (unsigned int j = bend; j < tbeg; ++j)
          {
            for (unsigned int i = rbeg; i < Nx; ++i)
            {
              unsigned int ipb = Nx_wrap - 2 - i + rbeg;
              Fe[d + dof * at(i, j, k, Nx, Ny)] = s * Fe[d + dof * at(ipb, j, k, Nx, Ny)]; 
            }
          }
        }
      }
    }
  }
  // periodic copy in y
  if (periodic[1])
  {
    #pragma omp parallel
    { 
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
    }
  }
  else
  {
    // get bc for left end of y 
    const BC* bc_yl = &(BCs[2 * dof]);
    // multiplier to enforce bc
    double s;
    for (unsigned int d = 0; d < dof; ++d)
    {
      // we only do something if bc is not none
      if (bc_yl[d] != none)
      { 
        if (bc_yl[d] == mirror) s = 1.0;
        if (bc_yl[d] == mirror_inv) s = -1.0;
        // copy eulerian data in x-z plane in interior region to 
        // adjacent ghost region at bottom end of y axis according to mirror or mirror_inv
        #pragma omp parallel for collapse(3)
        for (unsigned int k = dend; k < ubeg; ++k)
        {
          for (unsigned int j = 0; j < bend; ++j)
          {
            for (unsigned int i = 0; i < Nx; ++i)
            {
              unsigned int jpb = 2 * bend - j;
              Fe[d + dof * at(i, j, k, Nx, Ny)] = s * Fe[d + dof * at(i, jpb, k, Nx, Ny)]; 
            }
          }
        }
      }
    }
    // get bc for right end of y 
    const BC* bc_yr = &(BCs[3 * dof]);
    for (unsigned int d = 0; d < dof; ++d)
    {
      // we only do something if bc is not none
      if (bc_yr[d] != none)
      { 
        if (bc_yr[d] == mirror) s = 1.0;
        if (bc_yr[d] == mirror_inv) s = -1.0;
        // copy of eulerian data in x-z plane in interior region to 
        // adjacent ghost region at top end of y axis according to mirror or mirror_inv
        #pragma omp parallel for collapse(3)
        for (unsigned int k = dend; k < ubeg; ++k)
        {
          for (unsigned int j = tbeg; j < Ny; ++j)
          {
            for (unsigned int i = 0; i < Nx; ++i)
            {
              unsigned int jpb = Ny_wrap - 2 - j + tbeg;
              Fe[d + dof * at(i, j, k, Nx, Ny)] = s * Fe[d + dof * at(i, jpb, k, Nx, Ny)]; 
            }
          }
        }
      }
    }
  }
  // periodic copy in z
  if (periodic[2])
  {
    #pragma omp parallel
    {
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
  else
  {
    // get bc for left end of z 
    const BC* bc_zl = &(BCs[4 * dof]);
    // multiplier to enforce bc
    double s;
    for (unsigned int d = 0; d < dof; ++d)
    {
      // we only do something if bc is not none
      if (bc_zl[d] != none)
      { 
        if (bc_zl[d] == mirror) s = 1.0;
        if (bc_zl[d] == mirror_inv) s = -1.0;
        // copy eulerian data in x-y plane in interior region to 
        // adjacent ghost region at down end of z axis according to mirror or mirror_inv
        #pragma omp parallel for collapse(3)
        for (unsigned int k = 0; k < dend; ++k)
        {
          for (unsigned int j = 0; j < Ny; ++j)
          {
            for (unsigned int i = 0; i < Nx; ++i)
            {
              unsigned int kpb = 2 * dend - k;
              Fe[d + dof * at(i, j, k, Nx, Ny)] = s * Fe[d + dof * at(i, j, kpb, Nx, Ny)]; 
            }
          }
        }
      }
    }
    // get bc for right end of z
    const BC* bc_zr = &(BCs[5 * dof]);  
    for (unsigned int d = 0; d < dof; ++d)
    {
      // we only do something if bc is not none
      if (bc_zr[d] != none)
      { 
        if (bc_zr[d] == mirror) s = 1.0;
        if (bc_zr[d] == mirror_inv) s = -1.0;
        // copy eulerian data in x-y plane in interior region to 
        // adjacent ghost region at up end of z axis according to mirror or mirror_inv
        #pragma omp parallel for collapse(3)
        for (unsigned int k = ubeg; k < Nz; ++k)
        {
          for (unsigned int j = 0; j < Ny; ++j)
          {
            for (unsigned int i = 0; i < Nx; ++i)
            {
              unsigned int kpb = 2 * ubeg - k - 2;
              Fe[d + dof * at(i, j, k, Nx, Ny)] = s * Fe[d + dof * at(i, j, kpb, Nx, Ny)]; 
            }
          }
        }
      }
    }
  }
}
#endif
