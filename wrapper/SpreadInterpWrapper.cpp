#include"SpreadInterp.h"
#include"Grid.h" 
#include"ParticleList.h"

/* C wrapper for calling from Python. Any functions
   defined here should also have their prototypes 
   and wrappers defined in SpreadInterp.py */
extern "C"
{
  // forward decls
  double* GetSpread(Grid* grid);
  double* GetForces(ParticleList* particles);
  // Spread and get pointer to data
  void Spread(ParticleList* s, Grid* g) 
  {
    //// grid * data size
    //unsigned int N = g->Nxeff * g->Nyeff * g->Nzeff * g->dof;
    //// zero the extended grid
    //#pragma omp parallel for
    //for (unsigned int i = 0; i < N; ++i) g->fG_unwrap[i] = 0; 
    // spread from particles onto grid
    spread(*s, *g);   
    // return a pointer to spread data
    //return GetSpread(g);
  }
  
  // Spread and get pointer to data
  void Interpolate(ParticleList* s, Grid* g) 
  {
    // grid * data size
    //unsigned int N = g->Nxeff * g->Nyeff * g->Nzeff * g->dof;
    //// particles * data size
    //unsigned int M = s->nP * s->dof;
    //// reinitialize force for interp
    //#pragma omp parallel
    //{
    //  #pragma omp for
    //  for (unsigned int i = 0; i < M; ++i) s->fP[i] = 0;
    //  #pragma omp for
    //  for (unsigned int i = 0; i < N; ++i) {g->fG_unwrap[i] = 0;}
    //}
    // interpolate data from grid onto particles
    interpolate(*s, *g); 
    // return pointer to interpolated data
    //return GetForces(s);
  }
}
