# python native stuff
import sys
import random
# import Python modules wrapping C libraries (and also numpy)
sys.path.append('../python')
from Grid import *
from Particles import *
from SpreadInterp import *
from Transform import *
from Solvers import TriplyPeriodicStokes

nTrials = 50
Ls = np.linspace(60.,200.,5)
mobx = np.zeros((Ls.size,nTrials), dtype = np.double)
# degrees of freedom
dof = 3
# grid periodicity
periodic_x = periodic_y = periodic_z = True
# boundary conditions specified for ends of each axis
# 2 - no bc should be used for periodic grid
BCs = 2 * np.ones(dof * 6, dtype = np.uintc)


for iL in np.arange(0,Ls.size):
  for iTrial in np.arange(0,nTrials):
    # grid info 
    Nx = Ny = Nz = int(Ls[iL]) 
    hx = hy = hz = 0.5 
    Lx = Ly = Lz = hx * Nx 
    # number of particles
    nP = 1
    # viscocity
    eta = 1/4/np.sqrt(np.pi)
    
    # particle positions
    xP = np.zeros(3 * nP, dtype = np.double)
    xP[0] = random.random() * (Lx - hx)
    xP[1] = random.random() * (Ly - hy)
    xP[2] = random.random() * (Lz - hz)
    # particle forces
    fP = np.zeros(dof * nP, dtype = np.double)
    fP[0] = 1; fP[1] = 0; fP[2] = 0
    # beta for ES kernel for each particle (from table)
    betafP = np.array([1.714])
    # dimensionless radii given ES kernel for each particle (from table)
    cwfP = np.array([1.5539])
    # width of ES kernel given dimensionless radii (from table)
    wfP = np.array([6], dtype = np.ushort)
    # actual radii of the particles
    radP = hx * np.array([cwfP[0]])
    
    # instantiate the python grid wrapper
    gridGen = GridGen(Lx, Ly, Lz, hx, hy, hz, Nx, Ny, Nz, dof, periodic_x, periodic_y, periodic_z, BCs)
    # instantiate and define the grid with C lib call
    # this sets the GridGen.grid member to a pointer to a C++ Grid struct
    gridGen.Make()
    # instantiate the python particles wrapper
    particlesGen = ParticlesGen(nP, dof, xP, fP, radP, wfP, cwfP, betafP)
    # instantiate and define the particles with C lib call
    # this sets the ParticlesGen.particles member to a pointer to a C++ ParticlesList struct
    particles = particlesGen.Make()
    # setup the particles on the grid with C lib call
    # this builds the particles-grid locator and defines other
    # interal data used to spread and interpolate
    particlesGen.Setup(gridGen.grid)
    
    # spread forces on the particles (C lib)
    fG = Spread(particlesGen.particles, gridGen.grid, gridGen.Ntotal)

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
    
    # interpolate velocities on the particles (C lib)
    vP = Interpolate(particlesGen.particles, gridGen.grid, nP * dof)
    
    # save x mobility
    mobx[iL,iTrial] = vP[0] 
 
    # free memory persisting b/w C and python (C lib)
    fTransformer.Clean()
    bTransformer.Clean()
    gridGen.Clean()
    particlesGen.Clean()
  print(iL)

np.savetxt('x_mobility_nonUnit.txt', mobx)


