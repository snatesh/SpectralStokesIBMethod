#include<unordered_set>
#include<algorithm>
#include<fstream>
#include<iomanip>
#include<random>
#include<omp.h>
#include<math.h>
#include<fftw3.h>
#include"ParticleList.h"
#include"Grid.h"
#include"Quadrature.h"
#include"exceptions.h"


#ifndef MEM_ALIGN
  #define MEM_ALIGN 16 
#endif

//#pragma omp declare simd
//inline double const esKernel(const double x, const double beta, const double alpha)
//{
//  return exp(beta * (sqrt(1 - x * x / (alpha * alpha)) - 1));
//}

// null initialization
ParticleList::ParticleList() : xP(0), fP(0), betafP(0), alphafP(0), 
                             radP(0), normfP(0), wfP(0), wfxP(0), wfyP(0),
                             wfzP(0), nP(0), normalized(false), dof(0), 
                             unique_monopoles(ESParticleSet(20,esparticle_hash)),
                             xunwrap(0), yunwrap(0), zunwrap(0), zoffset(0), pt_wts(0)
{}

/* construct with external data by copy */
ParticleList::ParticleList(const double* _xP, const double* _fP, const double* _radP, 
                         const double* _betafP, const double* _cwfP, const unsigned short* _wfP, 
                         const unsigned int _nP, const unsigned int _dof) :
  nP(_nP), dof(_dof), alphafP(0), normfP(0), wfxP(0), wfyP(0), wfzP(0), normalized(false),
  unique_monopoles(ESParticleSet(20,esparticle_hash)), xunwrap(0), yunwrap(0), zunwrap(0),
  zoffset(0), pt_wts(0)
{
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
  #pragma omp parallel for
  for (unsigned int i = 0; i < nP; ++i)
  {
    xP[3 * i] = _xP[3 * i];
    xP[1 + 3 * i] = _xP[1 + 3 * i];
    xP[2 + 3 * i] = _xP[2 + 3 * i];
    radP[i] = _radP[i];
    betafP[i] = _betafP[i];
    cwfP[i] = _cwfP[i];
    wfP[i] = _wfP[i];
    for (unsigned int j = 0; j < dof; ++j)
    {
      fP[j + dof * i] = _fP[j + dof * i];
    }
  }
  this->setup();
}

void ParticleList::setForces(const double* _fP, unsigned int _dof)
{
  if (!dof) this->dof = _dof;
  else if (this->dof != _dof) exitErr("DOF does not match current.");
  if(!this->fP)
  {
    this->fP = (double*) fftw_malloc(nP * dof * sizeof(double));
  }
  #pragma omp parallel for
  for (unsigned int i = 0; i < dof * nP; ++i) this->fP[i] = _fP[i]; 
}

void ParticleList::zeroForces()
{
  if (this->fP)
  {
    #pragma omp parallel for
    for (unsigned int i = 0; i < dof * nP; ++i) this->fP[i] = 0;
  }
  else exitErr("Forces have not been allocated.");
}

void ParticleList::setup()
{
  if (this->validState())
  {
    this->findUniqueKernels();
    this->normalizeKernels();
  }
  else
  {
    exitErr("ParticleList is invalid.");
  }
}

void ParticleList::setup(Grid& grid)
{
  if (grid.validState())
  {
    if (dof != grid.dof)
    {
      exitErr("DOF of ParticleList must match DOF of grid.");
    }
    this->setup();
    this->locateOnGrid(grid);
  }
  else
  {
    exitErr("Particles could not be setup on grid because grid is invalid.");
  }
}

void ParticleList::findUniqueKernels()
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
void ParticleList::normalizeKernels()
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

void ParticleList::locateOnGrid(Grid& grid)
{
  if (not grid.has_locator)
  {
    if (grid.unifZ) {this->locateOnGridUnifZ(grid);}
    else {this->locateOnGridNonUnifZ(grid);}
    grid.has_locator = true; 
  }
}

void ParticleList::locateOnGridUnifZ(Grid& grid)
{
  // get widths on effective uniform grid
  #pragma omp parallel for
  for (unsigned int i = 0; i < nP; ++i)
  {
    wfxP[i] = std::round(2 * alphafP[i] / grid.hx);
    wfyP[i] = std::round(2 * alphafP[i] / grid.hy);
    wfzP[i] = std::round(2 * alphafP[i] / grid.hz);
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
  
  unsigned int* xclose = (unsigned int*) fftw_malloc(nP * sizeof(unsigned int));
  unsigned int* yclose = (unsigned int*) fftw_malloc(nP * sizeof(unsigned int));
  unsigned int* zclose = (unsigned int*) fftw_malloc(nP * sizeof(unsigned int));
  
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
      xclose[i] = (int) (xP[3 * i] / grid.hx);
      yclose[i] = (int) (xP[1 + 3 * i] / grid.hy);
      zclose[i] = (int) (xP[2 + 3 * i] / grid.hz);
      xclose[i] += ((wx % 2) && (xP[3 * i] / grid.hx - xclose[i] > 1.0 / 2.0) ? 1 : 0);
      yclose[i] += ((wy % 2) && (xP[1 + 3 * i] / grid.hy - yclose[i] > 1.0 / 2.0) ? 1 : 0);
      zclose[i] += ((wz % 2) && (xP[2 + 3 * i] / grid.hz - zclose[i] > 1.0 / 2.0) ? 1 : 0);
      for (unsigned int j = 0; j < wx; ++j)
      {
        xunwrap[j + i * wfxP_max] = ((double) xclose[i] + j - wx / 2 + evenx) * grid.hx - xP[3 * i];
        if (fabs(pow(xunwrap[j + i * wfxP_max],2) - pow(alphafP[i],2)) < 1e-14) 
        {
          xunwrap[j + i * wfxP_max] = alphafP[i];
        }
        // initialize buffer region if needed
        if (j == wx - 1 && wx < wfxP_max)
        {
          for (unsigned int k = wx; k < wfxP_max; ++k) {xunwrap[k + i * wfxP_max] = 0;}
        }
      } 
      for (unsigned int j = 0; j < wy; ++j)
      {
        yunwrap[j + i * wfyP_max] = ((double) yclose[i] + j - wy / 2 + eveny) * grid.hy - xP[1 + 3 * i];
        if (fabs(pow(yunwrap[j + i * wfyP_max],2) - pow(alphafP[i],2)) < 1e-14) 
        {
          yunwrap[j + i * wfyP_max] = alphafP[i];
        }
        // initialize buffer region if needed
        if (j == wy - 1 && wy < wfyP_max)
        {
          for (unsigned int k = wy; k < wfyP_max; ++k) {yunwrap[k + i * wfyP_max] = 0;}
        }
      } 
      for (unsigned int j = 0; j < wz; ++j)
      {
        zunwrap[j + i * wfzP_max] = ((double) zclose[i] + j - wz / 2 + evenz) * grid.hz - xP[2 + 3 * i];
        if (fabs(pow(zunwrap[j + i * wfzP_max],2) - pow(alphafP[i],2)) < 1e-14) 
        {
          zunwrap[j + i * wfzP_max] = alphafP[i];
        }
        // initialize buffer region if needed
        if (j == wz - 1 && wz < wfzP_max)
        {
          for (unsigned int k = wz; k < wfzP_max; ++k) {zunwrap[k + i * wfzP_max] = 0;}
        }
      }
      zoffset[i] = wx * wy * (zclose[i] - wz / 2 + evenz + wfzP_max);    
      grid.nextn[i] = -1;
    }
    #pragma omp for 
    for (unsigned int i = 0; i < N2; ++i) {grid.firstn[i] = -1; grid.number[i] = 0;}
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
  if (zclose) {fftw_free(zclose); zclose = 0;}
}

void ParticleList::locateOnGridNonUnifZ(Grid& grid)
{
  // get widths on effective uniform grid
  #pragma omp parallel for
  for (unsigned int i = 0; i < nP; ++i)
  {
    wfxP[i] = std::round(2 * alphafP[i] / grid.hx);
    wfyP[i] = std::round(2 * alphafP[i] / grid.hy);
  }
  wfxP_max = *std::max_element(wfxP, wfxP + nP); grid.Nxeff += 2 * wfxP_max;
  wfyP_max = *std::max_element(wfyP, wfyP + nP); grid.Nyeff += 2 * wfyP_max;
  double alphafP_max = *std::max_element(alphafP, alphafP + nP);

  // define extended z grid
  ext_down = 0; ext_up = 0;
  double *zG_ext, *zG_ext_wts;
  unsigned short* indl = (unsigned short*) fftw_malloc(nP * sizeof(unsigned short));
  unsigned short* indr = (unsigned short*) fftw_malloc(nP * sizeof(unsigned short));
 
  unsigned int i = 1;
  while (grid.zG[0] - grid.zG[i] <= alphafP_max) {ext_up += 1; i += 1;} 
  i = grid.Nz - 2;
  while (grid.zG[i] - grid.zG[grid.Nz - 1] <= alphafP_max) {ext_down += 1; i -= 1;}
  grid.Nzeff += ext_up + ext_down;
  zG_ext = (double*) fftw_malloc(grid.Nzeff * sizeof(double));
  zG_ext_wts = (double*) fftw_malloc(grid.Nzeff * sizeof(double));
  for (unsigned int i = ext_up; i < grid.Nzeff - ext_down; ++i)
  {
    zG_ext[i] = grid.zG[i - ext_up];
    zG_ext_wts[i] = grid.zG_wts[i - ext_up];
  } 
  unsigned int j = 0;
  for (unsigned int i = ext_up; i > 0; --i) 
  {
    zG_ext[j] = 2.0 * grid.zG[0] - grid.zG[i]; 
    zG_ext_wts[j] = grid.zG_wts[i];
    j += 1;
  }
  j = grid.Nzeff - ext_down;;
  for (unsigned int i = grid.Nz - 2; i > grid.Nz - 2 - ext_down; --i)
  { 
    zG_ext[j] = -1.0 * grid.zG[i]; 
    zG_ext_wts[j] = grid.zG_wts[i]; 
    j += 1; 
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
  }
  
  wfzP_max = *std::max_element(wfzP, wfzP + nP); 
  unsigned int N2 = grid.Nxeff * grid.Nyeff, N3 = N2 * grid.Nzeff;
  grid.fG_unwrap = (double*) fftw_malloc(N3 * grid.dof * sizeof(double)); 
  grid.firstn = (int*) fftw_malloc(N2 * sizeof(int));
  grid.number = (unsigned int*) fftw_malloc(N2 * sizeof(unsigned int));  
  grid.nextn = (int*) fftw_malloc(nP * sizeof(int));

  xunwrap = (double*) fftw_malloc(wfxP_max * nP * sizeof(double));
  yunwrap = (double*) fftw_malloc(wfyP_max * nP * sizeof(double));
  zunwrap = (double*) fftw_malloc(wfzP_max * nP * sizeof(double));
  pt_wts = (double*) fftw_malloc(wfzP_max * nP * sizeof(double));  
  unsigned int* xclose = (unsigned int*) fftw_malloc(nP * sizeof(unsigned int));
  unsigned int* yclose = (unsigned int*) fftw_malloc(nP * sizeof(unsigned int));
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
      xclose[i] = (int) (xP[3 * i] / grid.hx);
      yclose[i] = (int) (xP[1 + 3 * i] / grid.hy);
      xclose[i] += ((wx % 2) && (xP[3 * i] / grid.hx - xclose[i] > 1.0 / 2.0) ? 1 : 0);
      yclose[i] += ((wy % 2) && (xP[1 + 3 * i] / grid.hy - yclose[i] > 1.0 / 2.0) ? 1 : 0);
      for (unsigned int j = 0; j < wx; ++j)
      {
        xunwrap[j + i * wfxP_max] = ((double) xclose[i] + j - wx / 2 + evenx) * grid.hx - xP[3 * i];
        if (fabs(pow(xunwrap[j + i * wfxP_max],2) - pow(alphafP[i],2)) < 1e-14) 
        {
          xunwrap[j + i * wfxP_max] = alphafP[i] - 1e-14;
        }
        // initialize buffer region if needed
        if (j == wx - 1 && wx < wfxP_max)
        {
          for (unsigned int k = wx; k < wfxP_max; ++k) {xunwrap[k + i * wfxP_max] = 0;}
        }
      } 
      for (unsigned int j = 0; j < wy; ++j)
      {
        yunwrap[j + i * wfyP_max] = ((double) yclose[i] + j - wy / 2 + eveny) * grid.hy - xP[1 + 3 * i];
        if (fabs(pow(yunwrap[j + i * wfyP_max],2) - pow(alphafP[i],2)) < 1e-14) 
        {
          yunwrap[j + i * wfyP_max] = alphafP[i] - 1e-14;
        }
        // initialize buffer region if needed
        if (j == wy - 1 && wy < wfyP_max)
        {
          for (unsigned int k = wy; k < wfyP_max; ++k) {yunwrap[k + i * wfyP_max] = 0;}
        }
      }
      unsigned int k = 0;
      for (unsigned int j = indl[i]; j <= indr[i]; ++j)
      {
        zunwrap[k + i * wfzP_max] = zG_ext[j] - xP[2 + 3 * i]; 
        if (fabs(pow(zunwrap[k + i * wfzP_max],2) - pow(alphafP[i],2)) < 1e-14) 
        {
          zunwrap[k + i * wfzP_max] = alphafP[i] - 1e-14;
        }
        pt_wts[k + i * wfzP_max] = grid.hx * grid.hy * zG_ext_wts[j];
        k += 1;
        // initialize buffer region if needed
        if (k == wz - 1 && wz < wfzP_max)
        {
          for (unsigned int l = wz; l < wfzP_max; ++l) 
          {
            zunwrap[l + i * wfzP_max] = 0;
            pt_wts[l + i * wfzP_max] = 0;
          }
        }
      }
      zoffset[i] = wx * wy * indl[i];   
      grid.nextn[i] = -1;
    }
    #pragma omp for
    for (unsigned int i = 0; i < N2; ++i) {grid.firstn[i] = -1; grid.number[i] = 0;}
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
  if (zG_ext_wts) {fftw_free(zG_ext_wts); zG_ext_wts = 0;}
  if (indl) {fftw_free(indl); indl = 0;}
  if (indr) {fftw_free(indr); indr = 0;}
  if (xclose) {fftw_free(xclose); xclose = 0;}
  if (yclose) {fftw_free(yclose); yclose = 0;}
}

void ParticleList::update(const double* xP_new, Grid& grid)
{
  // loop over unique alphas
  for (const double& alphaf : unique_alphafP)
  {
    const unsigned short wx = std::round(2 * alphaf / grid.hx);
    const unsigned short wy = std::round(2 * alphaf / grid.hy);
    const int evenx = -1 * (wx % 2) + 1, eveny = -1 * (wy % 2) + 1;
    // loop over w^2 groups of columns
    for (unsigned int izero = 0; izero < wx; ++izero)
    {
      for (unsigned int jzero = 0; jzero < wy; ++jzero)
      {
        // Uncomment pragma if enforcing that particles
        // can move at most to a neighboring column per step

        //#pragma omp parallel for collapse(2)
        for (unsigned int ii = izero; ii < grid.Nxeff; ii += wx)
        {
          for (unsigned int jj = jzero; jj < grid.Nyeff; jj += wy)
          { 
            // column index
            int col = jj + ii * grid.Nyeff;
            // trailing ptr, index of first particle in col
            int nprev = -1, n = grid.firstn[col];
            // if there is a particle
            while (n > -1)
            {
              // if it has matching alpha
              if (alphafP[n] == alphaf) 
              {
                // check if particle n has moved out of the column
                int xclose = (int) (xP_new[3 * n] / grid.hx);
                int yclose = (int) (xP_new[1 + 3 * n] / grid.hy);
                xclose += ((wx % 2) && (xP_new[3 * n] / grid.hx - xclose > 1.0 / 2.0) ? 1 : 0);
                yclose += ((wy % 2) && (xP_new[1 + 3 * n] / grid.hy - yclose > 1.0 / 2.0) ? 1 : 0);
                int col_new = (yclose + wfyP_max) + (xclose + wfxP_max) * grid.Nyeff;
                // if the particle has moved
                if (col_new != col)
                {
                  // index of next pt in column
                  int nnext = grid.nextn[n];
                  // update add particle n to col_new of search struct
                  grid.nextn[n] = grid.firstn[col_new];
                  grid.firstn[col_new] = n;
                  grid.number[col_new] += 1;
                  // delete particle n from col of search struct
                  if (nprev == -1){grid.firstn[col] = nnext;}
                  else {grid.nextn[nprev] = nnext;}
                  grid.number[col] -= 1;
                  n = nnext;
                }
                // if the particle hasn't moved
                else
                {
                  // update trailing pointer and set n to next particle in col
                  nprev = n; n = grid.nextn[n];
                }
              }
              // if the particle does not have matching alpha
              else
              {
                // update trailing pointer and set n to next particle in col
                nprev = n; n = grid.nextn[n]; 
              }
            }
          }
        }
      }
    }
  }
}

/* write current state of ParticleList to ostream */
void ParticleList::writeParticles(std::ostream& outputStream) const
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
    exitErr("Unable to write particles to output stream.");
  }
}

/* write current state of ParticleList to file */
void ParticleList::writeParticles(const char* fname) const
{
  std::ofstream file; file.open(fname);
  writeParticles(file); file.close();
}

bool ParticleList::validState() const
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
      throw Exception("Degrees of freedom for data on particles must be specified.",
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

void ParticleList::cleanup()
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

    if (pt_wts) {fftw_free(pt_wts); pt_wts = 0;}
  }
}

void ParticleList::randInit(Grid& grid, const unsigned int _nP)
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
    unsigned short ws[3] = {4, 5, 6};
    //unsigned short ws[3] = {6, 6, 6};
    //unsigned short ws[3] = {5, 5, 5};
    //unsigned short ws[3] = {4, 4, 4};
    double betas[3] = {1.785, 1.886, 1.714};
    //double betas[3] = {1.714, 1.714, 1.714};
    //double betas[3] = {1.886, 1.886, 1.886};
    //double betas[3] = {1.785, 1.785, 1.785};
    double Rhs[3] = {1.2047, 1.3437, 1.5539};
    //double Rhs[3] = {1.5539, 1.5539, 1.5539};
    //double Rhs[3] = {1.3437, 1.3437, 1.3437};
    //double Rhs[3] = {1.2047, 1.2047, 1.2047};
    std::default_random_engine gen;
    std::uniform_int_distribution<int> unifInd(0,2);
    if (grid.unifZ)
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
    else
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
    exitErr("Particles could not be setup on grid because grid is invalid.");
  }
}
