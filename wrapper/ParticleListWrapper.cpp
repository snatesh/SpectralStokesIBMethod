#include "ParticleList.h"

/* C wrapper for calling ParticleList methods from Python. Any functions
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
  
  /* set or get data on the particles */
  void SetForces(ParticleList* particles, const double* _fP, unsigned int dof)
  {
    particles->setForces(_fP, dof);
  }
  
  /* zero the data on particles */
  void ZeroForces(ParticleList* particles) {particles->zeroForces();}
  double* GetForces(ParticleList* particles) {return particles->fP;}
  
  /* create random configuration given the grid and number of particles 
     NOTE: this calls Setup() internaly */
  ParticleList* RandomConfig(Grid* grid, const unsigned int nP)
  {
    ParticleList* particles = new ParticleList();
    particles->randInit(*grid, nP);
    return particles;
  } 
  
 
  double* getPoints(ParticleList* particles) {return particles->xP;}
  double* getRadii(ParticleList* particles) {return particles->radP;}
  double* getBetaf(ParticleList* particles) {return particles->betafP;}
  unsigned short* getWf(ParticleList* particles) {return particles->wfP;}
  unsigned short* getWfx(ParticleList* particles) {return particles->wfxP;}
  unsigned short* getWfy(ParticleList* particles) {return particles->wfyP;}
  unsigned short* getWfz(ParticleList* particles) {return particles->wfzP;}
  double* getNormf(ParticleList* particles) 
  {
    if (not particles->normalized) particles->normalizeKernels();
    return particles->normfP;
  }

  void CleanParticles(ParticleList* s) {s->cleanup();}
  void DeleteParticles(ParticleList* s) {if(s) {delete s; s = 0;}}
  void WriteParticles(ParticleList* s, const char* fname) {s->writeParticles(fname);}
}
