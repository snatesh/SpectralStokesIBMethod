# python native stuff
import sys
import random
# import Python modules wrapping C libraries (and also numpy)
sys.path.append('../python')
from Grid import *
from Species import *
from SpreadInterp import *
from Transform import *
from Solvers import TriplyPeriodicStokes

nTrials = 100
Ls = np.linspace(60.,200.,5)
mobx = np.zeros((Ls.size,nTrials), dtype = np.double)

for iL in np.arange(0,Ls.size):
  for iTrial in np.arange(0,nTrials):
    # grid info 
    Nx = Ny = Nz = int(Ls[iL]); dof = 3 
    hx = hy = hz = 1.0 
    Lx = Ly = Lz = Ls[iL] 
    # number of particles
    nP = 1
    # viscocity
    eta = 1/4/np.sqrt(np.pi)
    periodicity = 3
    
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
    radP = np.array([cwfP[0]])
    
    # instantiate the python grid wrapper
    gridGen = GridGen(Lx, Ly, Lz, hx, hy, hz, Nx, Ny, Nz, dof, periodicity)
    # instantiate and define the grid with C lib call
    # this is a pointer to a C++ struct
    grid = gridGen.Make()
    # instantiate the python species wrapper
    speciesGen = SpeciesGen(nP, dof, xP, fP, radP, wfP, cwfP, betafP)
    # instantiate and define the species with C lib call
    # this is a pointer to a C++ struct
    species = speciesGen.Make()
    # setup the species on the grid with C lib call
    # this builds the species-grid locator and defines other
    # interal data used to spread and interpolate
    speciesGen.Setup(species, grid)
    
    # spread forces on the particles (C lib)
    fG = Spread(species, grid, gridGen.Ntotal)
    
    # instantiate transform wrapper with spread forces (C lib)
    fTranformer = Transformer(fG, None, Nx, Ny, Nz, dof)
    # compute forward transform (C lib)
    forward = fTranformer.Ftransform()
    # get the Fourier coefficients (C lib)
    fG_hat_r = fTranformer.GetRealOut(forward)
    fG_hat_i = fTranformer.GetComplexOut(forward)
    
    # solve Stokes eq 
    U_hat_r, U_hat_i = TriplyPeriodicStokes(fG_hat_r, fG_hat_i, eta, Lx, Ly, Lz, Nx, Ny, Nz)
    
    # instantiate back transform wrapper with velocities on grid (C lib)
    bTransformer = Transformer(U_hat_r, U_hat_i, Nx, Ny, Nz, dof)
    backward = bTransformer.Btransform()
    # get real part of back transform and normalize (C lib)
    uG_r = bTransformer.GetRealOut(backward) / bTransformer.N
    
    # set velocity as new grid spread (C lib)
    gridGen.SetGridSpread(grid, uG_r)
    
    # interpolate velocities on the particles (C lib)
    vP = Interpolate(species, grid, nP * dof)
    
    # save x mobility
    mobx[iL,iTrial] = vP[0] 
 
    # free memory persisting b/w C and python (C lib)
    fTranformer.Clean(forward)
    fTranformer.Delete(forward)
    bTransformer.Clean(backward)
    bTransformer.Delete(backward)
    gridGen.Clean(grid)
    gridGen.Delete(grid)
    speciesGen.Clean(species)
    speciesGen.Delete(species)
  print(iL)

np.savetxt('x_mobility.txt', mobx)


