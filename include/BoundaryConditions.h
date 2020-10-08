#ifndef _BOUNDARY_CONDITION_H
#define _BOUNDARY_CONDITION_H

/* Enumeration for boundary condition types 
   These must be specified at the ends of
   each axis. Note, if periodic is applied
   at the end of one axis, it must be applied
   at the other end as well. */
enum BC {mirror, mirror_inv, none};

#endif
