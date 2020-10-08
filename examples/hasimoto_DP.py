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
from Solvers import DoublyPeriodicStokes_no_wall
from Ghost import *
nTrials = 50
Ls = np.linspace(60.,200.,5)
mobx = np.zeros((Ls.size,nTrials), dtype = np.double)
dof = 3
# boundary conditions specified for ends of each axis
# 0 - mirror wall
# 1 - inverse mirror wall
# 2 - none 
BCs = 2 * np.ones(dof * 6, dtype = np.uintc)
# grid periodicity
periodic_x = periodic_y = True; periodic_z = False;
# grid spacing in x,y
hx = hy = 0.5
# chebyshev grid and weights for z
Lz = 20.0; H = Lz / 2.0; 
Nz = int(np.ceil(1.25 * (np.pi / (np.arccos(-hx / H) - np.pi / 2))))
zpts, zwts = clencurt(Nz, 0, Lz)

# if no wall, choose whether to 0 the k=0 mode of the RHS
# k0 = 0 - the k=0 mode of the RHS for pressure and velocity will be 0
# k0 = 1 - the k=0 mode of the RHS for pressure and velocity will not be 0
#        - there will be a correction to the k=0 mode after each solve
k0 = 0;

for iL in range(0,Ls.size):
  for iTrial in range(0,nTrials):
    # grid info 
    Nx = Ny = int(Ls[iL]) 
    Lx = Ly = hx * Nx 
    # number of particles
    nP = 1
    # viscocity
    eta = 1/4/np.sqrt(np.pi)
    # particle positions
    xP = np.zeros(3 * nP, dtype = np.double)
    xP[0] = Lx / 2
    xP[1] = Lx / 2
    xP[2] = Lz / 2.0#random.random() * Lz
    print(xP)
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
    gridGen = GridGen(Lx, Ly, Lz, hx, hy, 0, Nx, Ny, Nz, dof, periodic_x, periodic_y, periodic_z, BCs, zpts, zwts)
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
    # interal data used to spread and interpolate, including ghost points
    particlesGen.Setup(gridGen.grid)
    # initialize the extended grid data 
    gridGen.ZeroExtGrid()
    # spread forces on the particles (C lib)
    Spread(particlesGen.particles, gridGen.grid, gridGen.Ntotal)
    # enforce DP boundary conditions on spread data
    DeGhostify(gridGen.grid, particlesGen.particles)
    # get spread data
    fG = gridGen.GetSpread()
    # instantiate transform wrapper with spread forces (C lib)
    fTransformer = Transformer(fG, None, Nx, Ny, Nz, dof)
    fTransformer.Ftransform_cheb()
    # get the Fourier coefficients
    fG_hat_r = fTransformer.out_real
    fG_hat_i = fTransformer.out_complex
    
    # solve Stokes eq 
    U_hat_r, U_hat_i, _, _ = DoublyPeriodicStokes_no_wall(fG_hat_r, fG_hat_i, eta, Lx, Ly, Lz, Nx, Ny, Nz, k0)
    
    # instantiate back transform wrapper with velocities on grid (C lib)
    bTransformer = Transformer(U_hat_r, U_hat_i, Nx, Ny, Nz, dof)
    bTransformer.Btransform_cheb()
    # get real part of back transform
    uG_r = bTransformer.out_real
    
    # set velocity as new grid spread (C lib)
    gridGen.SetSpread(uG_r)
    # reinitialize forces for interp
    particlesGen.ZeroForces()
    # copy data to ghost cells to enforce DP boundary conditions before interp
    Ghostify(gridGen.grid, particlesGen.particles)
    # interpolate velocities on the particles (C lib)
    Interpolate(particlesGen.particles, gridGen.grid, nP * dof)
    # get interp data
    vP = particlesGen.GetForces()
    print(vP)
    
    # save x mobility
    mobx[iL,iTrial] = vP[0] 
 
    # free memory persisting b/w C and python (C lib)
    fTransformer.Clean()
    bTransformer.Clean()
    gridGen.Clean()
    particlesGen.Clean()

np.savetxt('x_mobility_nonUnit_DP.txt', mobx)


