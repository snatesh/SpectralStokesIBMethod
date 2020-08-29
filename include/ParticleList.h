#ifndef PARTICLE_LIST_H
#define PARTICLE_LIST_H
#include<ostream>
#include<unordered_set>
#include<tuple>
#include<functional>

/*
 *  ParticleList is an SoA describing the particle set.
 *  
 *  xP - particle positions (x1,y1,z1,x2,y2,z2,...)
 *  fP - force on each particle (can be overwritten with interpolation)
 *  betafP - beta values for monopole kernels
 *  wfP - widths of monopole kernels
 *  normfP - normalizations for monopole kernels
 *  radP - radius of each particle
 *  cwfP - dimensionless radius given the kernel 
 *  alphafP - support of kernels, alpha = w * rad / (2 * c(w,beta))
 *  nP - number of particles
 *  wfxP, wfyP, wfzP - actual width we use for each direction
 *  unique_monopoles - unique ES kernels, automatically freed when ParticleList exits scope
 *  zoffset - offset index in the z direction for each particle
 *  pt_wts - the kernel weights for each particle (only populated if grid.unifZ = false)
*/

/* first  define some types to minimize work during initialization. eg. for es, we need to compute
   the normalization for each unique kernel, not each particle. */
typedef std::tuple<unsigned short, double, double, double> ESParticle;

// generalized hashing for range elements (from boost)
template <class T>
inline void hash_combine(std::size_t & seed, const T & v)
{
  std::hash<T> hasher;
  seed ^= hasher(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

// custom hash function to uniquely identify tuple of (w, beta, c(w), Rh)
auto esparticle_hash = [](const ESParticle& v)
{ 
  size_t seed = 0;
  hash_combine<unsigned short>(seed, std::get<0>(v));
  hash_combine<double>(seed, std::get<1>(v));
  hash_combine<double>(seed, std::get<2>(v));
  hash_combine<double>(seed, std::get<3>(v));
  return seed;
};


// extending unordered_set type to support the 4-tuples for the ES kernel
typedef std::unordered_set<ESParticle, decltype(esparticle_hash)> ESParticleSet;

// forward declare Grid
struct Grid;
struct ParticleList
{
  double *xP, *fP;
  double *xunwrap, *yunwrap, *zunwrap, *pt_wts;
  unsigned int *zoffset;
  double *radP, *betafP, *normfP, *alphafP, *cwfP; 
  unsigned short *wfP, *wfxP, *wfyP, *wfzP;
  unsigned short wfxP_max, wfyP_max, wfzP_max;
  unsigned int nP, dof, ext_down, ext_up;
  ESParticleSet unique_monopoles;
  std::unordered_set<double> unique_alphafP;
  bool normalized;
  
  /* empty/null ctor */
  ParticleList();
  /* construct with external data by copy */
  ParticleList(const double* xP, const double* fP, const double* radP, 
              const double* betafP, const double* cwfP, const unsigned short* wfP, 
              const unsigned int nP, const unsigned int dof);
  /* setup ParticleList based on what caller has provided */
  void setup(Grid& grid);
  /* finds unique kernels and computes normalization using clenshaw-curtis quadrature */
  void setup();
  /* clean memory */ 
  void cleanup();
  /* normalize ES kernels using clenshaw-curtis quadrature*/
  void normalizeKernels();
  /* find unique ES kernels */
  void findUniqueKernels();
  /* Locate the particles in terms of the columns of the grid,
     dispatching either locateOnGridUnifZ or locateOnGridNonUnivZ
     based on grid.unifZ boolean

     Specifically, this function computes
      - wf(x,y,z)P the widths given the grids on each axis
      - wf(x,y,z)P_max the max widths, used for extending the grid
      - grid.N(x,y,z)eff - the number of points on extended grid axes
      - allocates grid.fG_unwrap based on above extended size
      - (x,y,z)unwrap - the grid points in x,y,z for each particle
                        given the width of their kernels, in unwrapped
                        coordinates.
      - zoffset, the offset in the z direction for each particle
      - for NonUnifZ, the weights (pt_wts) for each particle given its width
  
      - MOST IMPORTANTLY, grid.firstn and grid.nextn are computed. These
        partition the particles on the grid into columns. 
        That is, 
          for column ind, grid.firstn[ind] = i1 is the index of the first particle in the column
                          grid.nextn[i1] = i2 is the index of the next particle in the column, and so on.
  */
  void locateOnGrid(Grid& grid);
  void locateOnGridUnifZ(Grid& grid);
  void locateOnGridNonUnifZ(Grid& grid);
  /* write current state of ParticleList to ostream */
  void writeParticles(std::ostream& outputStream) const; 
  void writeParticles(const char* fname) const;
  /* check validity of current state */
  bool validState() const;
  /* randomly initialize particles on grid, used for testing */
  void randInit(Grid& grid, const unsigned int _nP);
};


/* C wrapper for calling from Python. Any functions
   defined here should also have their prototypes 
   and wrappers defined in ParticleList.py */
extern "C"
{
  /* make a ParticleList from external data */
  ParticleList* MakeParticles(const double* xP, const double* fP, const double* radP, 
                           const double* betafP, const double* cwfP, const unsigned short* wfP, 
                           const unsigned int nP, const unsigned int dof)
  {
    ParticleList* particles = new ParticleList(xP, fP, radP, betafP, cwfP, wfP, nP, dof);
    return particles;
  }

  /* setup the particles on the grid (builds grid locators)*/
  void Setup(ParticleList* particles, Grid* grid)
  {
    particles->setup(*grid);  
  }

  /* create random configuration given the grid and number of particles 
     NOTE: this calls Setup() internaly */
  ParticleList* RandomConfig(Grid* grid, const unsigned int nP)
  {
    ParticleList* particles = new ParticleList();
    particles->randInit(*grid, nP);
    return particles;
  } 
 
  double* getPoints(ParticleList* particles) {return particles->xP;}
  double* getForces(ParticleList* particles) {return particles->fP;}
  double* getParticlesInterp(ParticleList* particles) {return particles->fP;}
  double* getRadii(ParticleList* particles) {return particles->radP;}
  double* getBetaf(ParticleList* particles) {return particles->betafP;}
  double* getNormf(ParticleList* particles) {return particles->normfP;}
  unsigned short* getWf(ParticleList* particles) {return particles->wfP;}
  unsigned short* getWfx(ParticleList* particles) {return particles->wfxP;}
  unsigned short* getWfy(ParticleList* particles) {return particles->wfyP;}
  unsigned short* getWfz(ParticleList* particles) {return particles->wfzP;}

  void CleanParticles(ParticleList* s) {s->cleanup();}
  void DeleteParticles(ParticleList* s) {if(s) {delete s; s = 0;}}
  void WriteParticles(ParticleList* s, const char* fname) {s->writeParticles(fname);}
}

#endif
