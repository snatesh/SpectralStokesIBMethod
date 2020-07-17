#include"SpreadInterp.h"
#include"Grid.h"
#include"SpeciesList.h"
#include"exceptions.h"
#include<omp.h>
#include<fftw3.h>
#include<algorithm>
/* before calling anything, need to run
  Grid grid; SpeciesList species; 
  grid.Setup(); species.Setup();
  species.NormalizeKernels(grid.maxh); 
  
*/
// TODO: for some reason, zunwrap is giving values out of the support, 
//       which causes the kernel evaluation to return nan as 1-(x[2]/alpha)^2 < 0
//       this happens for even and odd widths after a certain number of particles
//       are requested (????)

void spread(SpeciesList& species, Grid& grid)
{
  switch (grid.periodicity)
  {
    case 3 : spreadTP(species, grid); break;
    case 2 : spreadDP(species, grid); break;
    case 1 : spreadSP(species, grid); break;
    case 0 : spreadAP(species, grid); break;
    default : exitErr("grid periodicity invalid.");
  } 
}

void interpolate(SpeciesList& species, Grid& grid)
{
  switch (grid.periodicity)
  {
    case 3 : interpTP(species, grid); break;
    case 2 : interpDP(species, grid); break;
    case 1 : interpSP(species, grid); break;
    case 0 : interpAP(species, grid); break;
    default : exitErr("grid periodicity invalid.");
  } 
}

void spreadTP(SpeciesList& species, Grid& grid)
{
  // loop over unique alphas
  for (const double& alphaf : species.unique_alphafP)
  {
    const unsigned short wx = std::round(2 * alphaf / grid.hxeff);
    const unsigned short wy = std::round(2 * alphaf / grid.hyeff);
    const unsigned short wz = std::round(2 * alphaf / grid.hzeff);
    std::cout << std::setprecision(16) << wx << " " << wy  << " " << wz << std::endl;
    const unsigned short w2 = wx * wy;
    const unsigned int kersz = w2 * wz; 
    const unsigned int subsz = w2 * grid.Nzeff;
    const int evenx = -1 * (wx % 2) + 1, eveny = -1 * (wy % 2) + 1;
    // loop over w^2 groups of columns
    for (unsigned int izero = 0; izero < wx; ++izero)
    {
      for (unsigned int jzero = 0; jzero < wy; ++jzero)
      {
        // parallelize over the N^2/w^2 columns in a group
        #pragma omp parallel for collapse(2)
        for (unsigned int ii = izero; ii < grid.Nxeff; ii += wx)
        {
          for (unsigned int jj = jzero; jj < grid.Nyeff; jj += wy)
          { 
            // number of pts in this column
            unsigned int npts = grid.number[jj + ii * grid.Nyeff];
            // find first particle in column(ii,jj) with matching alpha 
            int l = grid.firstn[jj + ii * grid.Nyeff];
            while (l >= 0 && species.alphafP[l] != alphaf) 
            {
              l = grid.nextn[l];
              npts -= 1;
            }
            // continue if it's there
            if (l >= 0 && species.alphafP[l] == alphaf)
            {
              // global indices of wx x wy x Nz subarray influenced by column(i,j)
//              alignas(MEM_ALIGN) unsigned int indc3D[subsz];
              unsigned int* indc3D = (unsigned int*) fftw_malloc(subsz * sizeof(unsigned int));
              for (int k3D = 0; k3D < grid.Nzeff; ++k3D)
              {
                for (int j = 0; j < wy; ++j)
                {
                  int j3D = jj + j - wy / 2 + eveny;
                  for (int i = 0; i < wx; ++i) 
                  {
                    int i3D = ii + i - wx / 2 + evenx;
                    indc3D[at(i,j,k3D,wx,wy)] = at(i3D, j3D, k3D, grid.Nxeff, grid.Nyeff);
                  }
                }
              }
              // gather forces from grid subarray
              //alignas(MEM_ALIGN) double fGc[subsz * grid.dof];  
              double* fGc = (double*) fftw_malloc(subsz * grid.dof * sizeof(double));
              gather(subsz, fGc, grid.fG_unwrap, indc3D, grid.dof);
              // particle indices
              unsigned int npts_match = 1, count  = 1; int ltmp = l;
              // get other particles in col with this alphaf
              for (unsigned int ipt = 1; ipt < npts; ++ipt) 
              {
                ltmp = grid.nextn[ltmp];
                if (species.alphafP[ltmp] == alphaf) {npts_match += 1;}
              }
              // TODO: don't need to allocate this cause it'll never be very large
              //alignas(MEM_ALIGN) unsigned int indx[npts_match]; indx[0] = l;
              unsigned int* indx = (unsigned int*) fftw_malloc(npts_match * sizeof(unsigned int));
              indx[0] = l;
              for (unsigned int ipt = 1; ipt < npts; ++ipt)
              {
                l = grid.nextn[l];
                if (species.alphafP[l] == alphaf) {indx[count] = l; count += 1;}
              }

              // gather particle pts, betas, forces etc. for this column
              double *fPc, *betafPc, *normfPc, *xunwrap, *yunwrap, *zunwrap;
              unsigned int* zoffset;
              unsigned short* wfPc;
              fPc = (double*) fftw_malloc(npts_match * species.dof * sizeof(double));
              betafPc = (double*) fftw_malloc(npts_match * sizeof(double));
              wfPc = (unsigned short*) fftw_malloc(npts_match * sizeof(unsigned short));
              normfPc = (double*) fftw_malloc(npts_match * sizeof(double)); 
              xunwrap = (double*) fftw_malloc(species.wfxP_max * npts_match * sizeof(double)); 
              yunwrap = (double*) fftw_malloc(species.wfyP_max * npts_match * sizeof(double)); 
              zunwrap = (double*) fftw_malloc(species.wfzP_max * npts_match * sizeof(double));
              zoffset = (unsigned int*) fftw_malloc(npts_match * sizeof(unsigned int));

              gather(npts_match, betafPc, species.betafP, indx, 1);
              gather(npts_match, fPc, species.fP, indx, species.dof);
              gather(npts_match, normfPc, species.normfP, indx, 1);
              gather(npts_match, wfPc, species.wfP, indx, 1);
              gather(npts_match, xunwrap, species.xunwrap, indx, species.wfxP_max);
              gather(npts_match, yunwrap, species.yunwrap, indx, species.wfyP_max);
              gather(npts_match, zunwrap, species.zunwrap, indx, species.wfzP_max);
              gather(npts_match, zoffset, species.zoffset, indx, 1);

              // get the kernel w x w x w kernel weights for each particle in col 
              //alignas(MEM_ALIGN) double delta[kersz * npts_match];
              double* delta = (double*) fftw_malloc(kersz * npts_match * sizeof(double));
              delta_eval_col(delta, betafPc, wfPc, normfPc, xunwrap, yunwrap, 
                             zunwrap, alphaf, npts_match, wx, wy, wz, species.wfxP_max,
                             species.wfyP_max, species.wfzP_max);

              // spread the particle forces with the kernel weights
              spread_col(fGc, delta, fPc, zoffset, npts_match, kersz, grid.dof);

              // scatter back to global eulerian grid
              scatter(subsz, fGc, grid.fG_unwrap, indc3D, grid.dof);

              fftw_free(fPc); fPc = 0; fftw_free(betafPc); 
              betafPc = 0; fftw_free(wfPc); wfPc = 0; fftw_free(normfPc); normfPc = 0; 
              fftw_free(xunwrap); xunwrap = 0; fftw_free(yunwrap); yunwrap = 0;
              fftw_free(zunwrap); zunwrap = 0; fftw_free(zoffset); zoffset = 0; 
              fftw_free(delta); delta = 0; fftw_free(indc3D); indc3D = 0; fftw_free(fGc); 
              fGc = 0; fftw_free(indx); indx = 0;
            } // finished with column
          } 
        } // finished with group of columns
      }
    } // finished with all groups
  } // finished with this alphaf

  // fold periodic spread data from ghost region into interior
  foldTP(grid.fG_unwrap, grid.fG, species.wfxP_max, species.wfyP_max, 
           species.wfzP_max, grid.Nxeff, grid.Nyeff, grid.Nzeff, grid.dof);
}

void interpTP(SpeciesList& species, Grid& grid)
{
  // reinitialize force for interp
  #pragma omp parallel
  {
    #pragma omp for
    for (unsigned int i = 0; i < species.nP * species.dof; ++i) species.fP[i] = 0;
    #pragma omp for
    for (unsigned int i = 0; i < grid.Nxeff * grid.Nyeff * grid.Nzeff * grid.dof; ++i) {grid.fG_unwrap[i] = 0;}
  }
  // ensure periodicity of eulerian data for interpolation
  copyTP(grid.fG_unwrap, grid.fG, species.wfxP_max, species.wfyP_max, 
         species.wfzP_max, grid.Nxeff, grid.Nyeff, grid.Nzeff, grid.dof);
  // loop over unique alphas
  for (const double& alphaf : species.unique_alphafP)
  {
    const unsigned short wx = std::round(2 * alphaf / grid.hxeff);
    const unsigned short wy = std::round(2 * alphaf / grid.hyeff);
    const unsigned short wz = std::round(2 * alphaf / grid.hzeff);
    const unsigned short w2 = wx * wy;
    const unsigned int kersz = w2 * wz; 
    const unsigned int subsz = w2 * grid.Nzeff;
    const int evenx = -1 * (wx % 2) + 1, eveny = -1 * (wy % 2) + 1;
    const double weight = grid.hxeff * grid.hyeff * grid.hzeff;
    std::cout << std::setprecision(16) << wx << " " << wy  << " " << wz << " " << weight << std::endl;
    // loop over w^2 groups of columns
    for (unsigned int izero = 0; izero < wx; ++izero)
    {
      for (unsigned int jzero = 0; jzero < wy; ++jzero)
      {
        // parallelize over the N^2/w^2 columns in a group
        #pragma omp parallel for collapse(2)
        for (unsigned int ii = izero; ii < grid.Nxeff; ii += wx)
        {
          for (unsigned int jj = jzero; jj < grid.Nyeff; jj += wy)
          { 
            // number of pts in this column
            unsigned int npts = grid.number[jj + ii * grid.Nyeff];
            // find first particle in column(ii,jj) with matching alpha 
            int l = grid.firstn[jj + ii * grid.Nyeff];
            while (l >= 0 && species.alphafP[l] != alphaf) 
            {
              l = grid.nextn[l];
              npts -= 1;
            }
            // continue if it's there
            if (l >= 0 && species.alphafP[l] == alphaf)
            {
              // global indices of wx x wy x Nz subarray influenced by column(i,j)
//              alignas(MEM_ALIGN) unsigned int indc3D[subsz];
              unsigned int* indc3D = (unsigned int*) fftw_malloc(subsz * sizeof(unsigned int));
              for (int k3D = 0; k3D < grid.Nzeff; ++k3D)
              {
                for (int j = 0; j < wy; ++j)
                {
                  int j3D = jj + j - wy / 2 + eveny;
                  for (int i = 0; i < wx; ++i) 
                  {
                    int i3D = ii + i - wx / 2 + evenx;
                    indc3D[at(i,j,k3D,wx,wy)] = at(i3D, j3D, k3D, grid.Nxeff, grid.Nyeff);
                  }
                }
              }
              // gather forces from grid subarray
              //alignas(MEM_ALIGN) double fGc[subsz * grid.dof];  
              double* fGc = (double*) fftw_malloc(subsz * grid.dof * sizeof(double));
              gather(subsz, fGc, grid.fG_unwrap, indc3D, grid.dof);
              // particle indices
              unsigned int npts_match = 1, count  = 1; int ltmp = l;
              // get other particles in col with this alphaf
              for (unsigned int ipt = 1; ipt < npts; ++ipt) 
              {
                ltmp = grid.nextn[ltmp];
                if (species.alphafP[ltmp] == alphaf) {npts_match += 1;}
              }
              // TODO: don't need to allocate this cause it'll never be very large
              //alignas(MEM_ALIGN) unsigned int indx[npts_match]; indx[0] = l;
              unsigned int* indx = (unsigned int*) fftw_malloc(npts_match * sizeof(unsigned int));
              indx[0] = l;
              for (unsigned int ipt = 1; ipt < npts; ++ipt)
              {
                l = grid.nextn[l];
                if (species.alphafP[l] == alphaf) {indx[count] = l; count += 1;}
              }

              // gather particle pts, betas, forces etc. for this column
              double *fPc, *betafPc, *normfPc, *xunwrap, *yunwrap, *zunwrap;
              unsigned int* zoffset; unsigned short* wfPc;
              fPc = (double*) fftw_malloc(npts_match * species.dof * sizeof(double));
              betafPc = (double*) fftw_malloc(npts_match * sizeof(double));
              wfPc = (unsigned short*) fftw_malloc(npts_match * sizeof(unsigned short));
              normfPc = (double*) fftw_malloc(npts_match * sizeof(double)); 
              xunwrap = (double*) fftw_malloc(species.wfxP_max * npts_match * sizeof(double)); 
              yunwrap = (double*) fftw_malloc(species.wfyP_max * npts_match * sizeof(double)); 
              zunwrap = (double*) fftw_malloc(species.wfzP_max * npts_match * sizeof(double));
              zoffset = (unsigned int*) fftw_malloc(npts_match * sizeof(unsigned int));

              gather(npts_match, betafPc, species.betafP, indx, 1);
              gather(npts_match, fPc, species.fP, indx, species.dof);
              gather(npts_match, normfPc, species.normfP, indx, 1);
              gather(npts_match, wfPc, species.wfP, indx, 1);
              gather(npts_match, xunwrap, species.xunwrap, indx, species.wfxP_max);
              gather(npts_match, yunwrap, species.yunwrap, indx, species.wfyP_max);
              gather(npts_match, zunwrap, species.zunwrap, indx, species.wfzP_max);
              gather(npts_match, zoffset, species.zoffset, indx, 1);

              // get the kernel w x w x w kernel weights for each particle in col 
              //alignas(MEM_ALIGN) double delta[kersz * npts_match];
              double* delta = (double*) fftw_malloc(kersz * npts_match * sizeof(double));
              delta_eval_col(delta, betafPc, wfPc, normfPc, xunwrap, yunwrap, 
                             zunwrap, alphaf, npts_match, wx, wy, wz, species.wfxP_max,
                             species.wfyP_max, species.wfzP_max);

              // spread the particle forces with the kernel weights
              interp_col(fGc, delta, fPc, zoffset, npts_match, kersz, grid.dof, weight);

              // scatter back to global lagrangian grid
              scatter(npts_match, fPc, species.fP, indx, species.dof);

              fftw_free(fPc); fPc = 0; fftw_free(betafPc); 
              betafPc = 0; fftw_free(wfPc); wfPc = 0; fftw_free(normfPc); normfPc = 0; 
              fftw_free(xunwrap); xunwrap = 0; fftw_free(yunwrap); yunwrap = 0;
              fftw_free(zunwrap); zunwrap = 0; fftw_free(zoffset); zoffset = 0; 
              fftw_free(delta); delta = 0; fftw_free(indc3D); indc3D = 0; fftw_free(fGc); 
              fGc = 0; fftw_free(indx); indx = 0;
            } // finished with column
          } 
        } // finished with group of columns
      }
    } // finished with all groups
  } // finished with this alphaf
}

void spreadDP(SpeciesList& species, Grid& grid)
{
  // loop over unique alphas
  for (const double& alphaf : species.unique_alphafP)
  {
    const unsigned short wx = std::round(2 * alphaf / grid.hxeff);
    const unsigned short wy = std::round(2 * alphaf / grid.hyeff);
    std::cout << std::setprecision(16) << wx << " " << wy << std::endl;
    const unsigned short w2 = wx * wy;
    const unsigned int subsz = w2 * grid.Nzeff;
    const int evenx = -1 * (wx % 2) + 1, eveny = -1 * (wy % 2) + 1;
    // loop over w^2 groups of columns
    for (unsigned int izero = 0; izero < wx; ++izero)
    {
      for (unsigned int jzero = 0; jzero < wy; ++jzero)
      {
        // parallelize over the N^2/w^2 columns in a group
        #pragma omp parallel for collapse(2)
        for (unsigned int ii = izero; ii < grid.Nxeff; ii += wx)
        {
          for (unsigned int jj = jzero; jj < grid.Nyeff; jj += wy)
          {
            // number of pts in this column
            unsigned int npts = grid.number[jj + ii * grid.Nyeff];
            // find first particle in column(ii,jj) with matching alpha 
            int l = grid.firstn[jj + ii * grid.Nyeff];
            while (l >= 0 && species.alphafP[l] != alphaf) 
            {
              l = grid.nextn[l];
              npts -= 1;
            }
            // continue if it's there
            if (l >= 0 && species.alphafP[l] == alphaf)
            {
              // global indices of wx x wy x Nz subarray influenced by column(i,j)
              unsigned int* indc3D = (unsigned int*) fftw_malloc(subsz * sizeof(unsigned int));
              for (int k3D = 0; k3D < grid.Nzeff; ++k3D)
              {
                for (int j = 0; j < wy; ++j)
                {
                  int j3D = jj + j - wy / 2 + eveny;
                  for (int i = 0; i < wx; ++i) 
                  {
                    int i3D = ii + i - wx / 2 + evenx;
                    indc3D[at(i,j,k3D,wx,wy)] = at(i3D, j3D, k3D, grid.Nxeff, grid.Nyeff);
                  }
                }
              }
              // gather forces from grid subarray
              double* fGc = (double*) fftw_malloc(subsz * grid.dof * sizeof(double));
              gather(subsz, fGc, grid.fG_unwrap, indc3D, grid.dof);
              // particle indices
              unsigned int npts_match = 1, count  = 1; int ltmp = l;
              // get other particles in col with this alphaf
              for (unsigned int ipt = 1; ipt < npts; ++ipt) 
              {
                ltmp = grid.nextn[ltmp];
                if (species.alphafP[ltmp] == alphaf) {npts_match += 1;}
              }
              unsigned int* indx = (unsigned int*) fftw_malloc(npts_match * sizeof(unsigned int));
              indx[0] = l;
              for (unsigned int ipt = 1; ipt < npts; ++ipt)
              {
                l = grid.nextn[l];
                if (species.alphafP[l] == alphaf) {indx[count] = l; count += 1;}
              }

              // gather particle pts, betas, forces etc. for this column
              double *fPc, *betafPc, *normfPc, *xunwrap, *yunwrap, *zunwrap;
              unsigned int* zoffset; unsigned short *wfPc, *wz;
              fPc = (double*) fftw_malloc(npts_match * species.dof * sizeof(double));
              betafPc = (double*) fftw_malloc(npts_match * sizeof(double));
              wfPc = (unsigned short*) fftw_malloc(npts_match * sizeof(unsigned short));
              wz = (unsigned short*) fftw_malloc(npts_match * sizeof(unsigned short));
              normfPc = (double*) fftw_malloc(npts_match * sizeof(double)); 
              xunwrap = (double*) fftw_malloc(species.wfxP_max * npts_match * sizeof(double)); 
              yunwrap = (double*) fftw_malloc(species.wfyP_max * npts_match * sizeof(double)); 
              zunwrap = (double*) fftw_malloc(species.wfzP_max * npts_match * sizeof(double));
              zoffset = (unsigned int*) fftw_malloc(npts_match * sizeof(unsigned int));

              gather(npts_match, betafPc, species.betafP, indx, 1);
              gather(npts_match, fPc, species.fP, indx, species.dof);
              gather(npts_match, normfPc, species.normfP, indx, 1);
              gather(npts_match, wfPc, species.wfP, indx, 1);
              gather(npts_match, wz, species.wfzP, indx, 1);
              gather(npts_match, xunwrap, species.xunwrap, indx, species.wfxP_max);
              gather(npts_match, yunwrap, species.yunwrap, indx, species.wfyP_max);
              gather(npts_match, zunwrap, species.zunwrap, indx, species.wfzP_max);
              gather(npts_match, zoffset, species.zoffset, indx, 1);

              
              const unsigned int kersz = w2 * (*std::max_element(wz, wz + npts_match));

              // get the kernel w x w x w kernel weights for each particle in col 
              //alignas(MEM_ALIGN) double delta[kersz * npts_match];
              double* delta = (double*) fftw_malloc(kersz * npts_match * sizeof(double));
              delta_eval_col(delta, betafPc, wfPc, normfPc, xunwrap, yunwrap, 
                             zunwrap, alphaf, npts_match, wx, wy, wz, species.wfxP_max,
                             species.wfyP_max, species.wfzP_max);

              // spread the particle forces with the kernel weights
              spread_col(fGc, delta, fPc, zoffset, npts_match, w2, wz, grid.dof);

              // scatter back to global eulerian grid
              scatter(subsz, fGc, grid.fG_unwrap, indc3D, grid.dof);

              fftw_free(fPc); fPc = 0; fftw_free(betafPc); fftw_free(wz); wz = 0; 
              betafPc = 0; fftw_free(wfPc); wfPc = 0; fftw_free(normfPc); normfPc = 0; 
              fftw_free(xunwrap); xunwrap = 0; fftw_free(yunwrap); yunwrap = 0;
              fftw_free(zunwrap); zunwrap = 0; fftw_free(zoffset); zoffset = 0; 
              fftw_free(delta); delta = 0; fftw_free(indc3D); indc3D = 0; fftw_free(fGc); 
              fGc = 0; fftw_free(indx); indx = 0;
            } // finished with column
          } 
        } // finished with group of columns
      }
    } // finished with all groups
  } // finished with this alphaf

  // fold periodic spread data from ghost region into interior
  foldDP(grid.fG_unwrap, grid.fG, species.wfxP_max, species.wfyP_max, 
         species.ext_up, species.ext_down, grid.Nxeff, grid.Nyeff, grid.Nzeff, grid.dof);
}

void interpDP(SpeciesList& species, Grid& grid)
{
  // reinitialize force for interp
  #pragma omp parallel
  {
    #pragma omp for
    for (unsigned int i = 0; i < species.nP * species.dof; ++i) species.fP[i] = 0;
    #pragma omp for
    for (unsigned int i = 0; i < grid.Nxeff * grid.Nyeff * grid.Nzeff * grid.dof; ++i) {grid.fG_unwrap[i] = 0;}
  }
  // fold periodic spread data from ghost region into interior
  copyDP(grid.fG_unwrap, grid.fG, species.wfxP_max, species.wfyP_max, 
         species.ext_up, species.ext_down, grid.Nxeff, grid.Nyeff, grid.Nzeff, grid.dof);
  // loop over unique alphas
  for (const double& alphaf : species.unique_alphafP)
  {
    const unsigned short wx = std::round(2 * alphaf / grid.hxeff);
    const unsigned short wy = std::round(2 * alphaf / grid.hyeff);
    std::cout << std::setprecision(16) << wx << " " << wy << std::endl;
    const unsigned short w2 = wx * wy;
    const unsigned int subsz = w2 * grid.Nzeff;
    const int evenx = -1 * (wx % 2) + 1, eveny = -1 * (wy % 2) + 1;
    double weight;
    // loop over w^2 groups of columns
    for (unsigned int izero = 0; izero < wx; ++izero)
    {
      for (unsigned int jzero = 0; jzero < wy; ++jzero)
      {
        // parallelize over the N^2/w^2 columns in a group
        #pragma omp parallel for collapse(2)
        for (unsigned int ii = izero; ii < grid.Nxeff; ii += wx)
        {
          for (unsigned int jj = jzero; jj < grid.Nyeff; jj += wy)
          {
            // number of pts in this column
            unsigned int npts = grid.number[jj + ii * grid.Nyeff];
            // find first particle in column(ii,jj) with matching alpha 
            int l = grid.firstn[jj + ii * grid.Nyeff];
            while (l >= 0 && species.alphafP[l] != alphaf) 
            {
              l = grid.nextn[l];
              npts -= 1;
            }
            // continue if it's there
            if (l >= 0 && species.alphafP[l] == alphaf)
            {
              // global indices of wx x wy x Nz subarray influenced by column(i,j)
              unsigned int* indc3D = (unsigned int*) fftw_malloc(subsz * sizeof(unsigned int));
              for (int k3D = 0; k3D < grid.Nzeff; ++k3D)
              {
                for (int j = 0; j < wy; ++j)
                {
                  int j3D = jj + j - wy / 2 + eveny;
                  for (int i = 0; i < wx; ++i) 
                  {
                    int i3D = ii + i - wx / 2 + evenx;
                    indc3D[at(i,j,k3D,wx,wy)] = at(i3D, j3D, k3D, grid.Nxeff, grid.Nyeff);
                  }
                }
              }
              // gather forces from grid subarray
              double* fGc = (double*) fftw_malloc(subsz * grid.dof * sizeof(double));
              gather(subsz, fGc, grid.fG_unwrap, indc3D, grid.dof);
              // particle indices
              unsigned int npts_match = 1, count  = 1; int ltmp = l;
              // get other particles in col with this alphaf
              for (unsigned int ipt = 1; ipt < npts; ++ipt) 
              {
                ltmp = grid.nextn[ltmp];
                if (species.alphafP[ltmp] == alphaf) {npts_match += 1;}
              }
              unsigned int* indx = (unsigned int*) fftw_malloc(npts_match * sizeof(unsigned int));
              indx[0] = l;
              for (unsigned int ipt = 1; ipt < npts; ++ipt)
              {
                l = grid.nextn[l];
                if (species.alphafP[l] == alphaf) {indx[count] = l; count += 1;}
              }

              // gather particle pts, betas, forces etc. for this column
              double *fPc, *betafPc, *normfPc, *xunwrap, *yunwrap, *zunwrap;
              unsigned int* zoffset; unsigned short *wfPc, *wz;
              fPc = (double*) fftw_malloc(npts_match * species.dof * sizeof(double));
              betafPc = (double*) fftw_malloc(npts_match * sizeof(double));
              wfPc = (unsigned short*) fftw_malloc(npts_match * sizeof(unsigned short));
              wz = (unsigned short*) fftw_malloc(npts_match * sizeof(unsigned short));
              normfPc = (double*) fftw_malloc(npts_match * sizeof(double)); 
              xunwrap = (double*) fftw_malloc(species.wfxP_max * npts_match * sizeof(double)); 
              yunwrap = (double*) fftw_malloc(species.wfyP_max * npts_match * sizeof(double)); 
              zunwrap = (double*) fftw_malloc(species.wfzP_max * npts_match * sizeof(double));
              zoffset = (unsigned int*) fftw_malloc(npts_match * sizeof(unsigned int));

              gather(npts_match, betafPc, species.betafP, indx, 1);
              gather(npts_match, fPc, species.fP, indx, species.dof);
              gather(npts_match, normfPc, species.normfP, indx, 1);
              gather(npts_match, wfPc, species.wfP, indx, 1);
              gather(npts_match, wz, species.wfzP, indx, 1);
              gather(npts_match, xunwrap, species.xunwrap, indx, species.wfxP_max);
              gather(npts_match, yunwrap, species.yunwrap, indx, species.wfyP_max);
              gather(npts_match, zunwrap, species.zunwrap, indx, species.wfzP_max);
              gather(npts_match, zoffset, species.zoffset, indx, 1);

              
              const unsigned int kersz = w2 * (*std::max_element(wz, wz + npts_match));

              // get the kernel w x w x w kernel weights for each particle in col 
              //alignas(MEM_ALIGN) double delta[kersz * npts_match];
              double* delta = (double*) fftw_malloc(kersz * npts_match * sizeof(double));
              delta_eval_col(delta, betafPc, wfPc, normfPc, xunwrap, yunwrap, 
                             zunwrap, alphaf, npts_match, wx, wy, wz, species.wfxP_max,
                             species.wfyP_max, species.wfzP_max);

              // spread the particle forces with the kernel weights
              interp_col(fGc, delta, fPc, zoffset, npts_match, kersz, grid.dof, weight);

              // scatter back to global lagrangian grid
              scatter(npts_match, fPc, species.fP, indx, species.dof);

              fftw_free(fPc); fPc = 0; fftw_free(betafPc); fftw_free(wz); wz = 0; 
              betafPc = 0; fftw_free(wfPc); wfPc = 0; fftw_free(normfPc); normfPc = 0; 
              fftw_free(xunwrap); xunwrap = 0; fftw_free(yunwrap); yunwrap = 0;
              fftw_free(zunwrap); zunwrap = 0; fftw_free(zoffset); zoffset = 0; 
              fftw_free(delta); delta = 0; fftw_free(indc3D); indc3D = 0; fftw_free(fGc); 
              fGc = 0; fftw_free(indx); indx = 0;
            } // finished with column
          } 
        } // finished with group of columns
      }
    } // finished with all groups
  } // finished with this alphaf
}
