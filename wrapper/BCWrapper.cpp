#include"BoundaryConditions.h"
#include"Grid.h"
#include"ParticleList.h"

/* C wrapper for calling BoundaryConditions methods from Python. Any functions
   defined here should also have their prototypes 
   and wrappers defined in ParticleList.py */
extern "C"
{
  
  // fold spread data from ghost region of extended grid into the interior
  // according to periodicity or boundary condition for each data component
  void DeGhostify(Grid* grid, ParticleList* particles)
  {
    if (grid->unifZ)
    {
      fold(grid->fG_unwrap, grid->fG, particles->wfxP_max, particles->wfyP_max, 
           particles->wfzP_max, particles->wfzP_max, grid->Nxeff, grid->Nyeff, 
           grid->Nzeff, grid->dof, grid->isperiodic, grid->BCs);
    }
    else
    {
      fold(grid->fG_unwrap, grid->fG, particles->wfxP_max, particles->wfyP_max, 
           particles->ext_up, particles->ext_down, grid->Nxeff, grid->Nyeff, 
           grid->Nzeff, grid->dof, grid->isperiodic, grid->BCs);
    }
  }

  // copy spread data from interior grid to ghost region of extended grid
  // according to periodicity or boundary condition for each data component
  void Ghostify(Grid* grid, ParticleList* particles)
  {
    if (grid->unifZ)
    {
      copy(grid->fG_unwrap, grid->fG, particles->wfxP_max, particles->wfyP_max, 
           particles->wfzP_max, particles->wfzP_max, grid->Nxeff, grid->Nyeff, 
           grid->Nzeff, grid->dof, grid->isperiodic, grid->BCs);
    }
    else
    {
      copy(grid->fG_unwrap, grid->fG, particles->wfxP_max, particles->wfyP_max, 
           particles->ext_up, particles->ext_down, grid->Nxeff, grid->Nyeff, 
           grid->Nzeff, grid->dof, grid->isperiodic, grid->BCs);
    
    }
  }
}
