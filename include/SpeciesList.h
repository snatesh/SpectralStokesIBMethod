#ifndef SPECIES_LIST_H
#define SPECIES_LIST_H
#include<ostream>
#include<unordered_set>
#include<tuple>
#include<functional>

/*
 *  SpeciesList is an SoA describing the particle set.
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
 *  unique_monopoles - unique ES kernels, automatically freed when SpeciesList exits scope
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
struct SpeciesList
{
  double *xP, *fP;
  double *xunwrap, *yunwrap, *zunwrap;
  unsigned int *zoffset;
  double *radP, *betafP, *normfP, *alphafP, *cwfP; 
  unsigned short *wfP, *wfxP, *wfyP, *wfzP;
  unsigned short wfxP_max, wfyP_max, wfzP_max;
  unsigned int nP, dof, ext_down, ext_up;
  ESParticleSet unique_monopoles;
  std::unordered_set<double> unique_alphafP;
  bool normalized;
  
  /* empty/null ctor */
  SpeciesList();
  /* setup SpeciesList based on what caller has provided */
  void setup(Grid& grid);
  void setup(); 
  void cleanup();
  void randInit(Grid& grid, const unsigned int _nP);
  /* normalize ES kernels using clenshaw-curtis quadrature*/
  void normalizeKernels();
  /* find unique ES kernels */
  void findUniqueKernels();
  /* locate the particles in terms of the columns of the grid */
  void locateOnGrid(Grid& grid);
  void locateOnGridTP(Grid& grid);
  void locateOnGridDP(Grid& grid);
  /* write current state of SpeciesList to ostream */
  void writeSpecies(std::ostream& outputStream) const; 
  void writeSpecies(const char* fname) const;
  /* check validity of current state */
  bool validState() const;
};


// C wrapper for calling from Python
extern "C"
{
  SpeciesList* RandomConfig(Grid* grid, const unsigned int nP)
  {
    SpeciesList* species = new SpeciesList();
    species->randInit(*grid, nP);
    return species;
  } 
  
  double* getPoints(SpeciesList* species) {return species->xP;}
  double* getForces(SpeciesList* species) {return species->fP;}
  double* getSpeciesInterp(SpeciesList* species) {return species->fP;}
  double* getRadii(SpeciesList* species) {return species->radP;}
  double* getBetaf(SpeciesList* species) {return species->betafP;}
  double* getNormf(SpeciesList* species) {return species->normfP;}
  unsigned short* getWf(SpeciesList* species) {return species->wfP;}
  unsigned short* getWfx(SpeciesList* species) {return species->wfxP;}
  unsigned short* getWfy(SpeciesList* species) {return species->wfyP;}
  unsigned short* getWfz(SpeciesList* species) {return species->wfzP;}

  void CleanSpecies(SpeciesList* s) {s->cleanup();}
  void DeleteSpecies(SpeciesList* s) {if(s) {delete s; s = 0;}}
  void WriteSpecies(SpeciesList* s, const char* fname) {s->writeSpecies(fname);}
}

#endif
