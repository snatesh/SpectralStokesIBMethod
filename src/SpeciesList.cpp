#include<unordered_set>
#include<algorithm>
#include<fstream>
#include<iomanip>
#include<random>
#include<omp.h>
#include<math.h>
#include<fftw3.h>
#include "SpeciesList.h"
#include "Grid.h"
#include "chebyshev.h"
#include "exceptions.h"


#ifndef MEM_ALIGN
  #define MEM_ALIGN 16 
#endif

#pragma omp declare simd
inline double const esKernel(const double x, const double beta, const double alpha)
{
  return exp(beta * (sqrt(1 - x * x / (alpha * alpha)) - 1));
}

// null initialization
SpeciesList::SpeciesList() : xP(0), fP(0), betafP(0), alphafP(0), 
                             radP(0), normfP(0), wfP(0), wfxP(0), wfyP(0),
                             wfzP(0), nP(0), normalized(false), dof(0), 
                             unique_monopoles(ESParticleSet(20,esparticle_hash)),
                             xclose(0), yclose(0), zclose(0), xunwrap(0),
                             yunwrap(0), zunwrap(0), zoffset(0)
{}


void SpeciesList::setup()
{
  if (this->validState())
  {
    this->findUniqueKernels();
    this->normalizeKernels();
  }
  else
  {
    exitErr("SpeciesList is invalid.");
  }
}

void SpeciesList::randInit(Grid& grid, const unsigned int _nP)
{
  if (grid.validState())
  {
    nP = _nP;
    dof = grid.dof;
    xP = (double*) fftw_malloc(nP * 3 * sizeof(double));
    fP = (double*) fftw_malloc(nP * dof * sizeof(double));
    betafP = (double*) fftw_malloc(nP * sizeof(double));
    radP = (double*) fftw_malloc(nP * sizeof(double));
    cwfP = (double*) fftw_malloc(nP * sizeof(double));
    alphafP = (double*) fftw_malloc(nP * sizeof(double));
    normfP = (double*) fftw_malloc(nP * sizeof(double));
    wfP = (unsigned short*) fftw_malloc(nP * sizeof(unsigned short));
    wfxP = (unsigned short*) fftw_malloc(nP * sizeof(unsigned short));
    wfyP = (unsigned short*) fftw_malloc(nP * sizeof(unsigned short));
    wfzP = (unsigned short*) fftw_malloc(nP * sizeof(unsigned short));
    //unsigned short ws[3] = {4, 5, 6};
    //unsigned short ws[3] = {6, 6, 6};
    unsigned short ws[3] = {5, 5, 5};
    //unsigned short ws[3] = {4, 4, 4};
    //double betas[3] = {1.785, 1.886, 1.714};
    //double betas[3] = {1.714, 1.714, 1.714};
    double betas[3] = {1.886, 1.886, 1.886};
    //double betas[3] = {1.785, 1.785, 1.785};
    //double Rhs[3] = {1.2047, 1.3437, 1.5539};
    //double Rhs[3] = {1.5539, 1.5539, 1.5539};
    double Rhs[3] = {1.3437, 1.3437, 1.3437};
    //double Rhs[3] = {1.2047, 1.2047, 1.2047};
    std::default_random_engine gen;
    std::uniform_int_distribution<int> unifInd(0,2);
    if (grid.periodicity == 3)
    {
      unsigned int randInd;
      for (unsigned int i = 0; i < nP; ++i) 
      {
        xP[3 * i] = drand48() * (grid.Lx - grid.hx); 
        xP[1 + 3 * i] = drand48() * (grid.Ly - grid.hy); 
        xP[2 + 3 * i] = drand48() * (grid.Lz - grid.hz); 
        for (unsigned int j = 0; j < dof; ++j)
        {
          fP[j + dof * i] = 10;//2 * drand48() - 1;
        }
        randInd = unifInd(gen);
        // usually, we multiply this by h 
        radP[i] = Rhs[randInd];
        // and this is Rh/h
        cwfP[i] = Rhs[randInd];
        wfP[i] = ws[randInd];
        betafP[i] = betas[randInd];
      }
    }
    else if (grid.periodicity == 2)
    {
      unsigned int randInd;
      for (unsigned int i = 0; i < nP; ++i) 
      {
        xP[3 * i] = drand48() * (grid.Lx - grid.hx); 
        xP[1 + 3 * i] = drand48() * (grid.Ly - grid.hy); 
        xP[2 + 3 * i] = drand48() * grid.Lz; 
        for (unsigned int j = 0; j < dof; ++j)
        {
          fP[j + dof * i] = 10;//2 * drand48() - 1;
        }
        randInd = unifInd(gen);
        // usually, we multiply this by h 
        radP[i] = Rhs[randInd];
        cwfP[i] = Rhs[randInd];
        wfP[i] = ws[randInd];
        betafP[i] = betas[randInd];
      }
    }
    this->setup(grid);
  }
  else
  {
    exitErr("Species could not be setup on grid because grid is invalid.");
  }
}


void SpeciesList::setup(Grid& grid)
{
  if (grid.validState())
  {
    if (dof != grid.dof)
    {
      exitErr("DOF of SpeciesList must match DOF of grid.");
    }
    this->setup();
    this->locateOnGrid(grid);
  }
  else
  {
    exitErr("Species could not be setup on grid because grid is invalid.");
  }
}

void SpeciesList::findUniqueKernels()
{
  if (this->unique_monopoles.size() == 0)
  {
    for (unsigned int i = 0; i < nP; ++i)
    {
      this->unique_monopoles.emplace(wfP[i], betafP[i], cwfP[i], radP[i]); 
    }
  }
}

/* normalize ES kernels using clenshaw-curtis quadrature*/
void SpeciesList::normalizeKernels()
{
  // proceed if we haven't already normalized
  if (not this->normalized)
  {
    // cheb grid size, pts, weights and f for kernel vals
    unsigned int N = 1000;
    double* cpts = (double*) fftw_malloc(N * sizeof(double));
    double* cwts = (double*) fftw_malloc(N * sizeof(double)); 
    double* f = (double*) fftw_malloc(N * sizeof(double));
    double alpha, beta, betaw, norm, cw, rad; unsigned short w;
    // iterate over unique tuples of (w, beta, c(w), Rh)
    for (const auto& tuple : unique_monopoles)
    {
      w = std::get<0>(tuple); beta = std::get<1>(tuple);
      betaw = beta * w;
      cw = std::get<2>(tuple);
      rad = std::get<3>(tuple); 
      alpha = w * rad / (2 * cw);
      this->unique_alphafP.emplace(alpha);
      clencurt(cpts, cwts, -1 * alpha, alpha, N);
      #pragma omp simd aligned(f, cpts: MEM_ALIGN)
      for (unsigned int j = 0; j < N; ++j)
      {
        f[j] = esKernel(cpts[j], betaw, alpha);
      }
      norm = 0;
      // get normalization for this particle type (unique tuple)
      #pragma omp simd aligned(f, cwts: MEM_ALIGN)//,reduction(+:norm)
      for (unsigned int j = 0; j < N; ++j) {norm += f[j] * cwts[j];} 
      // assign the normalization to particles with this type
      for (unsigned int i = 0; i < this->nP; ++i)
      {
        if (wfP[i] == w && betafP[i] == beta && cwfP[i] == cw && radP[i] == rad) 
        {
          normfP[i] = norm; 
          alphafP[i] = alpha;
        }
      }
    }
    fftw_free(cpts);
    fftw_free(cwts);
    fftw_free(f);
    this->normalized = true;
  }
}

void SpeciesList::locateOnGrid(Grid& grid)
{
  if (not grid.has_locator)
  {
    switch (grid.periodicity)
    {
      case 3 : this->locateOnGridTP(grid); break;
      case 2 : this->locateOnGridDP(grid); break;  
      default : exitErr("grid periodicity invalid.");
    }
    grid.has_locator = true; 
  }
}

void SpeciesList::locateOnGridTP(Grid& grid)
{
  // get widths on effective uniform grid
  #pragma omp parallel for
  for (unsigned int i = 0; i < nP; ++i)
  {
    wfxP[i] = std::round(2 * alphafP[i] / grid.hxeff);
    wfyP[i] = std::round(2 * alphafP[i] / grid.hyeff);
    wfzP[i] = std::round(2 * alphafP[i] / grid.hzeff);
  }
  wfxP_max = *std::max_element(wfxP, wfxP + nP); grid.Nxeff += 2 * wfxP_max;
  wfyP_max = *std::max_element(wfyP, wfyP + nP); grid.Nyeff += 2 * wfyP_max;
  wfzP_max = *std::max_element(wfzP, wfzP + nP); grid.Nzeff += 2 * wfzP_max;
 
  unsigned int N2 = grid.Nxeff * grid.Nyeff, N3 = N2 * grid.Nzeff;
  grid.fG_unwrap = (double*) fftw_malloc(N3 * grid.dof * sizeof(double)); 
  grid.firstn = (int*) fftw_malloc(N2 * sizeof(int));
  grid.number = (unsigned int*) fftw_malloc(N2 * sizeof(unsigned int));  
  grid.nextn = (int*) fftw_malloc(nP * sizeof(int));

  xunwrap = (double *) fftw_malloc(wfxP_max * nP * sizeof(double));
  yunwrap = (double *) fftw_malloc(wfyP_max * nP * sizeof(double));
  zunwrap = (double *) fftw_malloc(wfzP_max * nP * sizeof(double));
  
  xclose = (unsigned int *) fftw_malloc(nP * sizeof(unsigned int));
  yclose = (unsigned int *) fftw_malloc(nP * sizeof(unsigned int));
  zclose = (unsigned int *) fftw_malloc(nP * sizeof(unsigned int));
  
  zoffset = (unsigned int *) fftw_malloc(nP * sizeof(unsigned int));
 
  #pragma omp parallel
  { 
    #pragma omp for nowait
    for (unsigned int i = 0; i < nP; ++i)
    {
      const unsigned short wx = wfxP[i];
      const unsigned short wy = wfyP[i];
      const unsigned short wz = wfzP[i];
      const int evenx = -1 * (wx % 2) + 1, eveny = -1 * (wy % 2) + 1;
      const int evenz = -1 * (wz % 2) + 1;
      xclose[i] = (int) (xP[3 * i] / grid.hxeff);
      yclose[i] = (int) (xP[1 + 3 * i] / grid.hyeff);
      zclose[i] = (int) (xP[2 + 3 * i] / grid.hzeff);
      xclose[i] += ((wx % 2) && (xP[3 * i] / grid.hxeff - xclose[i] > 1.0 / 2.0) ? 1 : 0);
      yclose[i] += ((wy % 2) && (xP[1 + 3 * i] / grid.hyeff - yclose[i] > 1.0 / 2.0) ? 1 : 0);
      zclose[i] += ((wz % 2) && (xP[2 + 3 * i] / grid.hzeff - zclose[i] > 1.0 / 2.0) ? 1 : 0);
      for (unsigned int j = 0; j < wx; ++j)
      {
        // TODO : something wrong with weird grid spacing like 0.3 or 0.7
        //std::cout << wx / 2 << " " << (double) wx / 2 << " " << std::round((double) wx / 2) << std::endl;
        xunwrap[j + i * wfxP_max] = ((double) xclose[i] + j - wx / 2 + evenx) * grid.hxeff - xP[3 * i];
        // initialize buffer region if needed
        if (j == wx - 1 && wx < wfxP_max)
        {
          for (unsigned int k = wx; k < wfxP_max; ++k) {xunwrap[k + i * wfxP_max] = 0;}
        }
        if (abs(xunwrap[j + i * wfxP_max])/alphafP[i] > 1)
        {
          std::cout << "bad X: " << j << " " << wx << " " << xunwrap[j + i * wfxP_max] \
                    << " " << xP[3 * i] << " " << alphafP[i] << " " \
                    << xP[3 * i] / grid.hxeff << " " << xclose[i] << std::endl;
        }
      } 
      for (unsigned int j = 0; j < wy; ++j)
      {
        yunwrap[j + i * wfyP_max] = ((double) yclose[i] + j - wy / 2 + eveny) * grid.hyeff - xP[1 + 3 * i];
        // initialize buffer region if needed
        if (j == wy - 1 && wy < wfyP_max)
        {
          for (unsigned int k = wy; k < wfyP_max; ++k) {yunwrap[k + i * wfyP_max] = 0;}
        }
      } 
      for (unsigned int j = 0; j < wz; ++j)
      {
        zunwrap[j + i * wfzP_max] = ((double) zclose[i] + j - wz / 2 + evenz) * grid.hzeff - xP[2 + 3 * i];
        // initialize buffer region if needed
        if (j == wz - 1 && wz < wfzP_max)
        {
          for (unsigned int k = wz; k < wfzP_max; ++k) {zunwrap[k + i * wfzP_max] = 0;}
        }
        if (abs(zunwrap[j + i * wfzP_max])/alphafP[i] > 1)
        {
          std::cout << "bad Z: " << j << " " << wz << " " \
                    << zunwrap[j + i * wfzP_max] << " " << xP[2 + 3 * i] << " " \
                    << alphafP[i] << " " << xP[2 + 3 * i] / grid.hzeff << " " \
                    << zclose[i] << std::endl;
        }
      }
      zoffset[i] = wx * wy * (zclose[i] - wz / 2 + evenz + wfzP_max);    
      grid.nextn[i] = -1;
    }
    #pragma omp for nowait
    for (unsigned int i = 0; i < N2; ++i) {grid.firstn[i] = -1; grid.number[i] = 0;}
    #pragma omp for
    for (unsigned int i = 0; i < N3 * grid.dof; ++i) grid.fG_unwrap[i] = 0;
  }

  int ind, indn;
  for (unsigned int i = 0; i < nP; ++i) 
  {
    ind = (yclose[i] + wfyP_max) + (xclose[i] + wfxP_max) * grid.Nyeff;
    if (grid.firstn[ind] < 0) {grid.firstn[ind] = i;}
    else
    {
      indn = grid.firstn[ind];
      while (grid.nextn[indn] >= 0)
      {
        indn = grid.nextn[indn];
      }
      grid.nextn[indn] = i;
    }
    grid.number[ind] += 1;
  }
  if (xclose) {fftw_free(xclose); xclose = 0;}
  if (yclose) {fftw_free(yclose); yclose = 0;}
  if (zclose) {fftw_free(yclose); yclose = 0;}
}

void SpeciesList::locateOnGridDP(Grid& grid)
{
  // get widths on effective uniform grid
  #pragma omp parallel for
  for (unsigned int i = 0; i < nP; ++i)
  {
    wfxP[i] = std::round(2 * alphafP[i] / grid.hxeff);
    wfyP[i] = std::round(2 * alphafP[i] / grid.hyeff);
  }
  wfxP_max = *std::max_element(wfxP, wfxP + nP); grid.Nxeff += 2 * wfxP_max;
  wfyP_max = *std::max_element(wfyP, wfyP + nP); grid.Nyeff += 2 * wfyP_max;
  double alphafP_max = *std::max_element(alphafP, alphafP + nP);

  // define extended z grid
  ext_down = 0; ext_up = 0;
  double* zG_ext;
  unsigned short* indl = (unsigned short*) fftw_malloc(nP * sizeof(unsigned short));
  unsigned short* indr = (unsigned short*) fftw_malloc(nP * sizeof(unsigned short));
  if (grid.zdescend)
  {
    unsigned int i = 1;
    while (grid.zG[0] - grid.zG[i] <= alphafP_max) {ext_up += 1; i += 1;} 
    i = grid.Nz - 2;
    while (grid.zG[i] - grid.zG[grid.Nz - 1] <= alphafP_max) {ext_down += 1; i -= 1;}
    grid.Nzeff = grid.Nz + ext_up + ext_down;
    zG_ext = (double*) fftw_malloc(grid.Nzeff * sizeof(double));
    for (unsigned int i = ext_up; i < grid.Nzeff - ext_down; ++i)
    {
      zG_ext[i] = grid.zG[i - ext_up];
    } 
    unsigned int j = 0;
    for (unsigned int i = ext_up; i > 0; --i) 
    {
      zG_ext[j] = 2.0 * grid.zG[0] - grid.zG[i]; j += 1;
    }
    j = grid.Nzeff - ext_down;;
    for (unsigned int i = grid.Nz - 2; i > grid.Nz - 2 - ext_down; --i)
    { 
      zG_ext[j] = -1.0 * grid.zG[i]; j += 1; 
    } 
    // find wz for each particle
    for (unsigned int i = 0; i < nP; ++i)
    {
      
      // find index of z grid pt w/i alpha below 
      auto high = std::lower_bound(&zG_ext[0], &zG_ext[0] + grid.Nzeff, \
                                   xP[2 + 3 * i] - alphafP[i], std::greater<double>());
      auto low = std::lower_bound(&zG_ext[0], &zG_ext[0] + grid.Nzeff, \
                                  xP[2 + 3 * i] + alphafP[i], std::greater<double>());
      indl[i] = low - &zG_ext[0];  
      indr[i] = high - &zG_ext[0];
      if (indr[i] == grid.Nzeff) {indr[i] -= 1;}
      else if (xP[2 + 3 * i] - alphafP[i] > zG_ext[indr[i]]) {indr[i] -= 1;}
      wfzP[i] = indr[i] - indl[i] + 1; 
      //std::cout << zG_ext[indl[i]]- zG_ext[indr[i]] << " " << 2 * alphafP[i] << std::endl; 
    }
  }
  else
  {
    unsigned int i = 1;
    while (grid.zG[i] - grid.zG[0] <= alphafP_max) {ext_down += 1; i += 1;} 
    i = grid.Nz - 2;
    while (grid.zG[grid.Nz - 1] - grid.zG[i] <= alphafP_max) {ext_up += 1; i -= 1;}
    grid.Nzeff = grid.Nz + ext_up + ext_down;
    zG_ext = (double*) fftw_malloc(grid.Nzeff * sizeof(double));
    for (unsigned int i = ext_down; i < grid.Nzeff - ext_up; ++i)
    {
      zG_ext[i] = grid.zG[i - ext_down];
    }
    unsigned int j = 0;
    for (unsigned int i = ext_down; i > 0; --i) 
    {
      zG_ext[j] = -1.0 * grid.zG[i]; j += 1;
    }
    j = grid.Nzeff - ext_up;
    for (unsigned int i = grid.Nz - 2; i > grid.Nz - 2 - ext_up; --i)
    { 
      zG_ext[j] = 2.0 * grid.zG[grid.Nz-1] - grid.zG[i]; j += 1; 
    }
    // find wz for each particle
    for (unsigned int i = 0; i < nP; ++i)
    {
      // find index of z grid pt w/i alpha below and above 
      auto low = std::lower_bound(&zG_ext[0], &zG_ext[0] + grid.Nzeff, \
                                  xP[2 + 3 * i] - alphafP[i], std::less<double>());
      auto high = std::lower_bound(&zG_ext[0], &zG_ext[0] + grid.Nzeff, \
                                  xP[2 + 3 * i] + alphafP[i], std::less<double>());
      indl[i] = low - &zG_ext[0];  
      indr[i] = high - &zG_ext[0];
      wfzP[i] = indr[i] - indl[i] + 1; 
    }
  }
  std::cout << "ext_up: " << ext_up << ", ext_down: " << ext_down << std::endl; 
  wfzP_max = *std::max_element(wfzP, wfzP + nP); 
  unsigned int N2 = grid.Nxeff * grid.Nyeff, N3 = N2 * grid.Nzeff;
  grid.fG_unwrap = (double*) fftw_malloc(N3 * grid.dof * sizeof(double)); 
  grid.firstn = (int*) fftw_malloc(N2 * sizeof(int));
  grid.number = (unsigned int*) fftw_malloc(N2 * sizeof(unsigned int));  
  grid.nextn = (int*) fftw_malloc(nP * sizeof(int));

  xunwrap = (double*) fftw_malloc(wfxP_max * nP * sizeof(double));
  yunwrap = (double*) fftw_malloc(wfyP_max * nP * sizeof(double));
  zunwrap = (double*) fftw_malloc(wfzP_max * nP * sizeof(double));
  
  xclose = (unsigned int*) fftw_malloc(nP * sizeof(unsigned int));
  yclose = (unsigned int*) fftw_malloc(nP * sizeof(unsigned int));
  zoffset = (unsigned int *) fftw_malloc(nP * sizeof(unsigned int));

  #pragma omp parallel
  { 
    #pragma omp for nowait
    for (unsigned int i = 0; i < nP; ++i)
    {
      const unsigned short wx = wfxP[i];
      const unsigned short wy = wfyP[i];
      const unsigned short wz = wfzP[i];
      const int evenx = -1 * (wx % 2) + 1, eveny = -1 * (wy % 2) + 1;
      xclose[i] = (int) (xP[3 * i] / grid.hxeff);
      yclose[i] = (int) (xP[1 + 3 * i] / grid.hyeff);
      xclose[i] += ((wx % 2) && (xP[3 * i] / grid.hxeff - xclose[i] > 1.0 / 2.0) ? 1 : 0);
      yclose[i] += ((wy % 2) && (xP[1 + 3 * i] / grid.hyeff - yclose[i] > 1.0 / 2.0) ? 1 : 0);
      for (unsigned int j = 0; j < wx; ++j)
      {
        xunwrap[j + i * wfxP_max] = ((double) xclose[i] + j - wx / 2 + evenx) * grid.hxeff - xP[3 * i];
        // initialize buffer region if needed
        if (j == wx - 1 && wx < wfxP_max)
        {
          for (unsigned int k = wx; k < wfxP_max; ++k) {xunwrap[k + i * wfxP_max] = 0;}
        }
      } 
      for (unsigned int j = 0; j < wy; ++j)
      {
        yunwrap[j + i * wfyP_max] = ((double) yclose[i] + j - wy / 2 + eveny) * grid.hyeff - xP[1 + 3 * i];
        // initialize buffer region if needed
        if (j == wy - 1 && wy < wfyP_max)
        {
          for (unsigned int k = wy; k < wfyP_max; ++k) {yunwrap[k + i * wfyP_max] = 0;}
        }
      }
      unsigned int k = 0;
      for (unsigned int j = indl[i]; j <= indr[i]; ++j)
      {
        zunwrap[k + i * wfzP_max] = zG_ext[j] - xP[2 + 3 * i]; k += 1;
        // initialize buffer region if needed
        if (k == wz - 1 && wz < wfzP_max)
        {
          for (unsigned int l = wz; l < wfzP_max; ++l) {zunwrap[l + i * wfzP_max] = 0;}
        }   
      } 
      zoffset[i] = wx * wy * indl[i];   
      grid.nextn[i] = -1;
    }
    #pragma omp for nowait
    for (unsigned int i = 0; i < N2; ++i) {grid.firstn[i] = -1; grid.number[i] = 0;}
    #pragma omp for
    for (unsigned int i = 0; i < N3 * grid.dof; ++i) grid.fG_unwrap[i] = 0;
  }

  int ind, indn;
  for (unsigned int i = 0; i < nP; ++i) 
  {
    ind = (yclose[i] + wfyP_max) + (xclose[i] + wfxP_max) * grid.Nyeff;
    if (grid.firstn[ind] < 0) {grid.firstn[ind] = i;}
    else
    {
      indn = grid.firstn[ind];
      while (grid.nextn[indn] >= 0)
      {
        indn = grid.nextn[indn];
      }
      grid.nextn[indn] = i;
    }
    grid.number[ind] += 1;
  }

  if (zG_ext) {fftw_free(zG_ext); zG_ext = 0;}
  if (xclose) {fftw_free(xclose); xclose = 0;}
  if (yclose) {fftw_free(yclose); yclose = 0;}
  if (indl) {fftw_free(indl); indl = 0;}
  if (indr) {fftw_free(indr); indr = 0;}
}

/* write current state of SpeciesList to ostream */
void SpeciesList::writeSpecies(std::ostream& outputStream) const
{
  if (this->validState() && outputStream.good()) 
  { 
    for (unsigned int i = 0; i < nP; ++i)
    {
      for (unsigned int j = 0; j < 3; ++j)
      {
        outputStream << std::setprecision(16) << xP[j + i * 3] << " ";
      }
      for (unsigned int j = 0; j < this->dof; ++j)
      {
        outputStream << fP[j + i * dof] << " ";
      }
      outputStream << wfP[i] << " " << betafP[i] << " ";
      if (this->normalized) {outputStream << std::setprecision(16) << normfP[i] << " ";}
      outputStream << std::endl;
    }
  }
  else
  {
    exitErr("Unable to write species to output stream.");
  }
}

/* write current state of SpeciesList to file */
void SpeciesList::writeSpecies(const char* fname) const
{
  std::ofstream file; file.open(fname);
  writeSpecies(file); file.close();
}

bool SpeciesList::validState() const
{
  try
  {
    if (not (xP && fP && wfP && betafP && normfP && radP && 
             cwfP && alphafP && wfxP && wfyP && wfzP)) 
    {
      throw Exception("Pointer(s) is null", __func__, __FILE__,__LINE__ );
    } 
    if (not dof)
    {
      throw Exception("Degrees of freedom for data on species must be specified.",
                      __func__, __FILE__, __LINE__);
    }
  } 
  catch (Exception& e)
  {
    e.getErr();
    return false;
  }
  return true;
}

void SpeciesList::cleanup()
{
  if (this->validState())
  {
    if (xP) {fftw_free(xP); xP = 0;}
    if (fP) {fftw_free(fP); fP = 0;}
    if (betafP) {fftw_free(betafP); betafP = 0;}
    if (radP) {fftw_free(radP); radP = 0;}
    if (cwfP) {fftw_free(cwfP); cwfP = 0;}
    if (alphafP) {fftw_free(alphafP); alphafP = 0;}  
    if (normfP) {fftw_free(normfP); normfP = 0;}  
    if (wfP) {fftw_free(wfP); wfP = 0;} 
    if (wfxP) {fftw_free(wfxP); wfxP = 0;} 
    if (wfyP) {fftw_free(wfyP); wfyP = 0;} 
    if (wfzP) {fftw_free(wfzP); wfzP = 0;} 
    if (xunwrap) {fftw_free(xunwrap); xunwrap = 0;}
    if (yunwrap) {fftw_free(yunwrap); yunwrap = 0;}
    if (zunwrap) {fftw_free(zunwrap); zunwrap = 0;}
    if (zoffset) {fftw_free(zoffset); zoffset = 0;}
  }
}

