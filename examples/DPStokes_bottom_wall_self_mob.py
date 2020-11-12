# python native stuff
import sys
import random
# import Python modules wrapping C libraries (and also numpy)
sys.path.append('../python')
from Grid import *
from Particles import *
from SpreadInterp import *
from Transform import *
from Chebyshev import *
from Solvers import DoublyPeriodicStokes_init, DoublyPeriodicStokes_bottom_wall
from Ghost import *
import matplotlib.pyplot as plt

# grid info 
#Nx = 100; Ny = 100; 
#hx = 0.4; hy = 0.4;
#Lx = hx * Nx; Ly = hy * Ny;
#Lz = Lx / 4; H = Lz / 2; 
#Nz = np.ceil(np.pi/(np.arccos(-hx/H) - np.pi/2)).astype(np.int);
Nx = 128; Ny = 128; Nz = 65; dof = 3 
hx = 6.435421e-01; hy = 6.435421e-01;
Lx = 2 * 41.1866915502928066; Ly = 2 * 41.1866915502928066; Lz = 22.0959392496299643; 
dof = 3; H = Lz / 2 
# number of particles
nP = 1
# viscocity
eta = 1/4/np.sqrt(np.pi)
# boundary conditions specified for ends of each axis
# 0 - mirror wall
# 1 - inverse mirror wall
# 2 - none 
BCs = 2 * np.ones(dof * 6, dtype = np.uintc)
# apply mirror-inv on bottom wall only for each solution component
BCs[5 * dof] = BCs[5 * dof + 1] = BCs[5 * dof + 2] = 1
# grid periodicity
periodic_x = periodic_y = True; periodic_z = False;
# chebyshev grid and weights for z
zpts, zwts = clencurt(Nz, 0, Lz)
# flag for whether to write output or not
write = False
heights = np.linspace(0,10,21)
mobx = np.zeros((heights.size,1))
Kx, Ky, K, Dx, Dy, FIMat, SIMat, pints, uvints, BCs_k0,\
  BCs_k, LU, Ainv_B, C, PIV, C_k0, Ginv, Ginv_k0, BCR1, BCL1, BCR2, BCL2 \
    = DoublyPeriodicStokes_init(Nx, Ny, Nz, Lx, Ly, H)

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
for iHeight in range(0,heights.size):
  for iP in np.arange(0,nP):
    # random index
    randInd = random.randrange(np.size(wf_choices)) 
    cwfP[iP] = cwf_choices[randInd]
    wfP[iP] = wf_choices[randInd]
    betafP[iP] = betaf_choices[randInd]
    radP[iP] = 1;#hx * cwfP[iP] 
    xP[3 * iP] = Lx / 2
    xP[1 + 3 * iP] = Ly / 2
    xP[2 + 3 * iP] = heights[iHeight] 
    fP[2 + dof * iP] = 1.0
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
  # solve DP Stokes eq with the DP + Correction method 
  U_hat_r, U_hat_i, P_hat_r, P_hat_i = \
    DoublyPeriodicStokes_bottom_wall(fG_hat_r, fG_hat_i, zpts, eta, Nx, Ny, Nz, H,\
                                     Kx, Ky, K, Dx, Dy, FIMat, SIMat, pints, \
                                     uvints, BCs_k0, BCs_k, LU, Ainv_B, C, \
                                     PIV, C_k0, Ginv, Ginv_k0, BCR1, BCL2)

  
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
  mobx[iHeight] = vP[2]
  print(mobx)
  if write:
    # write particles with interpolated vel to file (C lib)
    particlesGen.WriteParticles('particles.txt')
    # write grid velocities
    gridGen.WriteGrid('velocities.txt')
  # reset the forces on the particles
  particlesGen.SetForces(fP)
  
  # free memory persisting b/w C and python (C lib)
  fTransformer.Clean()
  bTransformer.Clean()
  gridGen.Clean()
  particlesGen.Clean()

print(mobx)
fig, ax = plt.subplots(1,1)
ax.plot(heights / radP[0], mobx/(1 / (6 * np.pi * eta * radP[0])),'ro-')
plt.show()
