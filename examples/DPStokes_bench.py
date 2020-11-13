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
from timeit import default_timer as timer
import matplotlib.pyplot as plt

# grid info 
Lx = 2 * 71.273569727781705; Ly = 2 * 71.273569727781705; Lz = 16.890351204817986; H = Lz / 2
dof = 3; 
# viscocity
eta = 1/4/np.sqrt(np.pi)
# boundary conditions specified for ends of each axis
# 0 - mirror wall
# 1 - inverse mirror wall
# 2 - none 
BCs = 2 * np.ones(dof * 6, dtype = np.uintc)
# grid periodicity
periodic_x = periodic_y = True; periodic_z = False;
# kernel width, dimensionless rad, ES beta
wf = 6; cwf = 1.5539; betaf = 1.714;
#cwf_choices = np.array([1.2047, 1.3437, 1.5539])
#betaf_choices = np.array([1.785, 1.886, 1.714])

# read 100k particle positionsfrom file
xP_tmp = np.loadtxt('pts.txt', delimiter=',')

Ns = np.linspace(32,128,25)
maxit = 30
#maxit = 1
timeInit = np.zeros(Ns.size)
timeSpread = np.zeros(Ns.size)
timeTransform = np.zeros(Ns.size)
timeSolve = np.zeros(Ns.size)
timeInterp = np.zeros(Ns.size)
for iN in range(Ns.size):
  Nx = Ny = int(Ns[iN]); 
  hx = hy = Lx / Nx
  Nz = int(np.ceil(np.pi/(np.arccos(-hx/H) - np.pi/2)).astype(np.int))
  # chebyshev grid and weights for z
  zpts, zwts = clencurt(Nz, 0, Lz)
  # number of particles to use
  phi_max = 0.1
  nP = int(np.ceil(phi_max * (Lx * Ly * Lz) / ((4 / 3) * np.pi * np.max(hx * cwf) ** 2)))
  print(nP)
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
  for iP in np.arange(0,nP):
    cwfP[iP] = cwf
    wfP[iP] = wf
    betafP[iP] = betaf
    # actual particle radius
    radP[iP] = hx * cwf 
    xP[3 * iP] = xP_tmp[iP,0]
    xP[1 + 3 * iP] = xP_tmp[iP,1]
    xP[2 + 3 * iP] = xP_tmp[iP,2] 
    for j in np.arange(0,dof):
      fP[j + dof * iP] = 1.0
  xP[0::3] += Lx / 2.0
  xP[1::3] += Lx / 2.0

  for nIt in range(0,maxit): 
    t0 = timer()
    # precompute wave nums, fourier deriv ops, cheb integral mats and linops+bcs for each k
    Kx, Ky, K, Dx, Dy, FIMat, SIMat, pints, uvints, BCs_k0, \
      BCs_k, LU, Ainv_B, C, PIV, C_k0, Ginv, Ginv_k0, _, _, _, _, \
        = DoublyPeriodicStokes_init(Nx, Ny, Nz, Lx, Ly, H)
    # instantiate the python grid wrapper
    gridGen = GridGen(Lx, Ly, Lz, hx, hy, 0, Nx, Ny, Nz, dof, periodic_x, periodic_y, periodic_z, BCs, zpts, zwts)
    # instantiate and define the grid with C lib call
    # this sets the GridGen.grid member to a pointer to a C++ Grid struct
    gridGen.Make()
    # if no wall, choose whether to 0 the k=0 mode of the RHS
    # k0 = 0 - the k=0 mode of the RHS for pressure and velocity will be 0
    # k0 = 1 - the k=0 mode of the RHS for pressure and velocity will not be 0
    #        - there will be a correction to the k=0 mode after each solve
    k0 = 0;
    # instantiate the python particles wrapper
    particlesGen = ParticlesGen(nP, dof, xP, fP, radP, wfP, cwfP, betafP)
    # instantiate and define the particles with C lib call
    # this sets the ParticlesGen.particles member to a pointer to a C++ ParticlesList struct
    particlesGen.Make()
    # setup the particles on the grid with C lib call
    # this builds the particles-grid locator and defines other
    # interal data used to spread and interpolate
    particlesGen.Setup(gridGen.grid)
    timeInit[iN] += timer() - t0
 
    t0 = timer() 
    # initialize the extended grid data 
    gridGen.ZeroExtGrid()
    # spread forces on the particles (C lib)
    Spread(particlesGen.particles, gridGen.grid, gridGen.Ntotal)
    # enforce DP boundary conditions on spread data
    DeGhostify(gridGen.grid, particlesGen.particles)
    fG = gridGen.GetSpread()
    timeSpread[iN] += timer() - t0

    tFtransform = timer()
    # instantiate forward transform wrapper with spread forces (C lib)
    fTransformer = Transformer(fG, None, Nx, Ny, Nz, dof)
    fTransformer.Ftransform_cheb()
    # get the Fourier coefficients
    fG_hat_r = fTransformer.out_real
    fG_hat_i = fTransformer.out_complex
    tFtransform = timer() - tFtransform

    t0 = timer()
    # solve Stokes eq 
    U_hat_r, U_hat_i, _, _ = DoublyPeriodicStokes_no_wall(fG_hat_r, fG_hat_i, eta, Nx, Ny, Nz, H, \
                                                          Kx, Ky, K, Dx, Dy, FIMat, SIMat, pints, \
                                                          uvints, BCs_k0, BCs_k, LU, Ainv_B, C, \
                                                          PIV, C_k0, Ginv, Ginv_k0, k0)
    timeSolve[iN] += timer() - t0
    
    tBtransform = timer()
    # instantiate back transform wrapper with velocities on grid (C lib)
    bTransformer = Transformer(U_hat_r, U_hat_i, Nx, Ny, Nz, dof)
    bTransformer.Btransform_cheb()
    # get real part of back transform
    uG_r = bTransformer.out_real
    uG_i = bTransformer.out_complex
    tBtransform = timer() - tBtransform
    timeTransform[iN] += tFtransform + tBtransform 

    t0 = timer()
    # set velocity as new grid spread (C lib)
    gridGen.SetSpread(uG_r)
    # reinitialize forces on particles for interpolation
    particlesGen.ZeroForces()
    # copy data to ghost cells to enforce DP boundary conditions before interp
    Ghostify(gridGen.grid, particlesGen.particles)
    # interpolate velocities on the particles (C lib)
    Interpolate(particlesGen.particles, gridGen.grid, nP * dof)
    vP = particlesGen.GetForces()
    timeInterp[iN] += timer() - t0

    # reset the forces on the particles
    particlesGen.SetForces(fP)
    # free memory persisting b/w C and python (C lib)
    fTransformer.Clean()
    bTransformer.Clean()
    gridGen.Clean()
    particlesGen.Clean()
    #print(timeInterp)

timeInit /= maxit; timeSpread /= maxit; timeTransform /= maxit; timeSolve /= maxit; timeInterp /= maxit
fig, ax = plt.subplots()
labels = []; width = 0.7
for iN in range(Ns.size):
  labels.append(str(int(Ns[iN])))
r = np.arange(0,Ns.size)
ax.bar(r, timeInit, width, label = 'Initialize')
ax.bar(r, timeSpread, bottom=timeInit, width=width, label = 'Spread')
ax.bar(r, timeTransform, bottom=(timeInit + timeSpread).tolist(), width=width, label = 'Transform')
ax.bar(r, timeSolve, bottom=(timeInit+timeSpread+timeTransform).tolist(), width=width, label = 'Solve')
ax.bar(r, timeInterp, bottom=(timeInit+timeSpread+timeTransform+timeSolve).tolist(), width=width, label = 'Interpolate')
plt.xticks(r, labels)
plt.xlabel('$N_{xy}$', fontsize=17)
plt.ylabel('Time (s)', fontsize=17)
plt.title('Timing DP solver stages for $L_{xy} = 140$, $L_z = 17$, $N_z =$ match $h_{xy}$, $\phi = 0.1$', fontsize=20)
ax.axes.get_xaxis().set_visible(True)
ax.axes.get_yaxis().set_visible(True)
ax.legend()
plt.show()
