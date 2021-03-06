# python native stuff
import sys
import random
# import Python modules wrapping C libraries (and also numpy)
sys.path.append('../python')
from Grid import *
from Particles import *
from SpreadInterp import *
from Transform import *
from Ghost import *
from Solvers import TriplyPeriodicStokes

# grid info 
Nx = 64; Ny = 64; Nz = 64; dof = 3 
hx = 0.5; hy = 0.25; hz = 1; 
Lx = Nx * hx; Ly = Ny * hy; Lz = Nz * hz; 
# number of particles
nP = 100
# viscocity
eta = 1/4/np.sqrt(np.pi)
# grid periodicity
periodic_x = periodic_y = periodic_z = True
# boundary conditions specified for ends of each axis
# 2 - no bc should be used for periodic grid
BCs = 2 * np.ones(dof * 6, dtype = np.uintc)

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
wf_choices = np.array([4,5,6])
cwf_choices = np.array([1.2047, 1.3437, 1.5539])
betaf_choices = np.array([1.785, 1.886, 1.714])
for iP in np.arange(0,nP):
  # random index
  randInd = random.randrange(np.size(wf_choices)) 
  cwfP[iP] = cwf_choices[randInd]
  wfP[iP] = wf_choices[randInd]
  betafP[iP] = betaf_choices[randInd]
  # set actual radius to same as dimensionless
  # so if h < 1, more than w points will be under the kernel
  radP[iP] = cwfP[iP] 
  xP[3 * iP] = random.random() * (Lx - hx)
  xP[1 + 3 * iP] = random.random() * (Ly - hy)
  xP[2 + 3 * iP] = random.random() * (Lz - hz)
  for j in np.arange(0,dof):
    fP[j + dof * iP] = 10


# instantiate the python grid wrapper
gridGen = GridGen(Lx, Ly, Lz, hx, hy, hz, Nx, Ny, Nz, dof, periodic_x, periodic_y, periodic_z, BCs)
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
# initialize extended grid
gridGen.ZeroExtGrid()
# spread forces on the particles (C lib)
Spread(particlesGen.particles, gridGen.grid, gridGen.Ntotal)
# handle triply periodic BCs
DeGhostify(gridGen.grid, particlesGen.particles)
# get spread data
fG = gridGen.GetSpread()
# write the grid with spread to file (C lib)
gridGen.WriteGrid('spread.txt')
gridGen.WriteCoords('coords.txt')  

# instantiate transform wrapper with spread forces (C lib)
fTransformer = Transformer(fG, None, Nx, Ny, Nz, dof)
fTransformer.Ftransform()
# get the Fourier coefficients
fG_hat_r = fTransformer.out_real
fG_hat_i = fTransformer.out_complex

# solve Stokes eq 
U_hat_r, U_hat_i = TriplyPeriodicStokes(fG_hat_r, fG_hat_i, eta, Lx, Ly, Lz, Nx, Ny, Nz)

# instantiate back transform wrapper with velocities on grid (C lib)
bTransformer = Transformer(U_hat_r, U_hat_i, Nx, Ny, Nz, dof)
bTransformer.Btransform()
# get real part of back transform
uG_r = bTransformer.out_real

# set velocity as new grid spread (C lib)
gridGen.SetSpread(uG_r)
# reinitialize forces on particles before interp
particlesGen.ZeroForces()
# populate ghost points according to triply periodic BCs
Ghostify(gridGen.grid, particlesGen.particles)
# interpolate velocities on the particles (C lib)
Interpolate(particlesGen.particles, gridGen.grid, nP * dof)
vP = particlesGen.GetForces()
print(vP)
# write particles with interpolated vel to file (C lib)
particlesGen.WriteParticles('particles.txt')
# write grid velocities
gridGen.WriteGrid('velocities.txt')

# free memory persisting b/w C and python (C lib)
fTransformer.Clean()
bTransformer.Clean()
gridGen.Clean()
particlesGen.Clean()
