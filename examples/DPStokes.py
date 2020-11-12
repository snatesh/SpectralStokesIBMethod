# python native stuff
import sys
import random
sys.path.append('../python')
from Grid import *
from Particles import *
from SpreadInterp import *
from Transform import *
from Chebyshev import *
from Solvers import DoublyPeriodicStokes_init, DoublyPeriodicStokes_no_wall
from Ghost import *
import timeit


# grid info 
Nx = 128; Ny = 128; Nz = 29; dof = 3 
hx = 1.113649526996589; hy = 1.113649526996589;
Lx = 2 * 71.273569727781705; Ly = 2 * 71.273569727781705; Lz = 16.890351204817986; H = Lz / 2
# precompute wave nums, fourier deriv ops, cheb integral mats and linops+bcs for each k
Kx, Ky, K, Dx, Dy, FIMat, SIMat, pints, uvints, BCs_k0, \
  BCs_k, LU, Ainv_B, C, PIV, C_k0, Ginv, Ginv_k0, _, _, _, _, \
    = DoublyPeriodicStokes_init(Nx, Ny, Nz, Lx, Ly, H)

# number of particles
nP = 100000
# viscocity
eta = 1/4/np.sqrt(np.pi)
# boundary conditions specified for ends of each axis
# 0 - mirror wall
# 1 - inverse mirror wall
# 2 - none 
BCs = 2 * np.ones(dof * 6, dtype = np.uintc)
# grid periodicity
periodic_x = periodic_y = True; periodic_z = False;
# chebyshev grid and weights for z
zpts, zwts = clencurt(Nz, 0, Lz)
# flag for whether to write output or not
write = False

# particle positions
xP = np.zeros(3 * nP, dtype = np.double)
# particle forces
fP = np.zeros(dof * nP, dtype = np.double)
# beta for ES kernel for each particle (from table)
betafP = np.zeros(nP, dtype = np.double)
# dimensionless radii given ES kernel for each particle (from table)
cwfP = np.zeros(nP, dtype = np.double)
# width of ES kernel given dimensionless radii (from table)
wfP = np.zeros(nP, dtype = np.ushort)
# actual radii of the particles
radP = np.zeros(nP, dtype = np.double)
# define random configuration of particles
# in terms of kernel, force and position
#wf_choices = np.array([4,5,6])
wf_choices = np.array([6,6,6])
#cwf_choices = np.array([1.2047, 1.3437, 1.5539])
cwf_choices = np.array([1.5539, 1.5539, 1.5539])
#betaf_choices = np.array([1.785, 1.886, 1.714])
betaf_choices = np.array([1.714, 1.714, 1.714])
xP_tmp = np.loadtxt('pts.txt', delimiter=',')
for iP in np.arange(0,nP):
  # random index
  randInd = random.randrange(np.size(wf_choices)) 
  cwfP[iP] = cwf_choices[randInd]
  wfP[iP] = wf_choices[randInd]
  betafP[iP] = betaf_choices[randInd]
  # set actual radius to same as dimensionless
  # so if h < 1, more than w points will be under the kernel
  radP[iP] = hx * cwfP[iP] 
  xP[3 * iP] = xP_tmp[iP,0]#random.random() * (Lx - hx)
  xP[1 + 3 * iP] = xP_tmp[iP,1]#random.random() * (Ly - hy)
  xP[2 + 3 * iP] = xP_tmp[iP,2]#random.random() * Lz 
  #xP[3 * iP] = random.random() * (Lx - hx)
  #xP[1 + 3 * iP] = random.random() * (Ly - hy)
  #xP[2 + 3 * iP] = random.random() * Lz 
  for j in np.arange(0,dof):
    fP[j + dof * iP] = 1.0
xP[0::3] += Lx / 2.0
xP[1::3] += Lx / 2.0
# if no wall, choose whether to 0 the k=0 mode of the RHS
# k0 = 0 - the k=0 mode of the RHS for pressure and velocity will be 0
# k0 = 1 - the k=0 mode of the RHS for pressure and velocity will not be 0
#        - there will be a correction to the k=0 mode after each solve
k0 = 0;
  
# instantiate the python grid wrapper
gridGen = GridGen(Lx, Ly, Lz, hx, hy, 0, Nx, Ny, Nz, dof, periodic_x, periodic_y, periodic_z, BCs, zpts, zwts)
# instantiate and define the grid with C lib call
# this sets the GridGen.grid member to a pointer to a C++ Grid struct
gridGen.Make()
# instantiate the python particles wrapper
particlesGen = ParticlesGen(nP, dof, xP, fP, radP, wfP, cwfP, betafP)
# instantiate and define the particles with C lib call
# this sets the ParticlesGen.particles member to a pointer to a C++ ParticlesList struct
particlesGen.Make()
# setup the particles on the grid with C lib call
# this builds the particles-grid locator and defines other
# interal data used to spread and interpolate
particlesGen.Setup(gridGen.grid)

time = 0.0
nits = 10
for j in range(0,nits):
  t0 = timeit.default_timer()
  # initialize the extended grid data 
  gridGen.ZeroExtGrid()
  # spread forces on the particles (C lib)
  Spread(particlesGen.particles, gridGen.grid, gridGen.Ntotal)
  # enforce DP boundary conditions on spread data
  DeGhostify(gridGen.grid, particlesGen.particles)
  fG = gridGen.GetSpread()
  if write:
    # write the grid with spread to file (C lib)
    gridGen.WriteGrid('spread.txt')
    gridGen.WriteCoords('coords.txt')  
  
  # instantiate forward transform wrapper with spread forces (C lib)
  fTransformer = Transformer(fG, None, Nx, Ny, Nz, dof)
  fTransformer.Ftransform_cheb()
  # get the Fourier coefficients
  fG_hat_r = fTransformer.out_real
  fG_hat_i = fTransformer.out_complex
  # solve Stokes eq 
  U_hat_r, U_hat_i, _, _ = DoublyPeriodicStokes_no_wall(fG_hat_r, fG_hat_i, eta, Nx, Ny, Nz, H, \
                                                        Kx, Ky, K, Dx, Dy, FIMat, SIMat, pints, \
                                                        uvints, BCs_k0, BCs_k, LU, Ainv_B, C, \
                                                        PIV, C_k0, Ginv, Ginv_k0, k0)
  # instantiate back transform wrapper with velocities on grid (C lib)
  bTransformer = Transformer(U_hat_r, U_hat_i, Nx, Ny, Nz, dof)
  bTransformer.Btransform_cheb()
  # get real part of back transform
  uG_r = bTransformer.out_real
  uG_i = bTransformer.out_complex
  # set velocity as new grid spread (C lib)
  gridGen.SetSpread(uG_r)
  # reinitialize forces on particles for interpolation
  particlesGen.ZeroForces()
  # copy data to ghost cells to enforce DP boundary conditions before interp
  Ghostify(gridGen.grid, particlesGen.particles)
  # interpolate velocities on the particles (C lib)
  Interpolate(particlesGen.particles, gridGen.grid, nP * dof)
  vP = particlesGen.GetForces()
  if write:
    # write particles with interpolated vel to file (C lib)
    particlesGen.WriteParticles('particles.txt')
    # write grid velocities
    gridGen.WriteGrid('velocities.txt')
  # reset the forces on the particles
  particlesGen.SetForces(fP)
  time += timeit.default_timer() - t0

# free memory persisting b/w C and python (C lib)
fTransformer.Clean()
bTransformer.Clean()
gridGen.Clean()
particlesGen.Clean()
print(time / nits)
