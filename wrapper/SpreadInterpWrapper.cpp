#include "SpreadInterp.h"


/* C wrapper for calling from Python. Any functions
   defined here should also have their prototypes 
   and wrappers defined in SpreadInterp.py */
extern "C"
{
  // forward decls
  double* GetSpread(Grid* grid);
  double* GetForces(ParticleList* particles);
  // Spread/Interp and get pointer to data
  double* Spread(ParticleList* s, Grid* g) {spread(*s, *g); return GetSpread(g);}
  double* Interpolate(ParticleList* s, Grid* g) {interpolate(*s, *g); return GetForces(s);}
}
