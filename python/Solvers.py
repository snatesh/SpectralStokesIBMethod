import numpy as np
from Chebyshev import * 
import ctypes

# get LAPACK solvers called in threaded context and create wrappers
libLinSolve = ctypes.CDLL('../lib/liblinSolve.so')
# get doubly periodic tools, parallelized with openMP and using blas/lapack
libDPTools = ctypes.CDLL('../lib/libdpTools.so')
# see end of file for lib function signatures

###################################################################################
########################## Main solver routines ###################################
###################################################################################

def TriplyPeriodicStokes(fG_hat_r, fG_hat_i, eta, Lx, Ly, Lz, Nx, Ny, Nz):
  """
  Solve triply periodic Stokes eq in Fourier domain given the Fourier
  coefficients of the forcing.
  
  Parameters:
    fG_hat_r, fG_hat_i - real and complex part of Fourier coefficients
                         of spread forces on the grid. These are both
                         arrays of doubles (not complex).
  
    eta - viscocity
    Lx, Ly, Lz - length of unit cell in x,y,z
    Nx, Ny, Nz - number of points in x, y and z
  
  Returns:
    U_hat_r, U_hat_i - real and complex part of Fourier coefficients of
                       fluid velocity on the grid. 
  
  Note: We assume the net force on the unit cell is 0 by *ignoring* 
        the k = 0 mode. That is, the k=0 mode of the output solution
        will be 0.
  """
  Ntotal = Nx * Ny * Nz * 3
  # separate x,y,z components
  f_hat = (fG_hat_r[0::3] + 1j * fG_hat_i[0::3])
  g_hat = (fG_hat_r[1::3] + 1j * fG_hat_i[1::3])
  h_hat = (fG_hat_r[2::3] + 1j * fG_hat_i[2::3])
  # wave numbers
  kvec_x = 2*np.pi*np.concatenate((np.arange(0,np.floor(Nx/2)),\
                                   np.arange(-1*np.ceil(Nx/2),0)), axis=None) / Lx
  
  kvec_y = 2*np.pi*np.concatenate((np.arange(0,np.floor(Ny/2)),\
                                   np.arange(-1*np.ceil(Ny/2),0)), axis=None) / Ly
  
  kvec_z = 2*np.pi*np.concatenate((np.arange(0,np.floor(Nz/2)),\
                                   np.arange(-1*np.ceil(Nz/2),0)), axis=None) / Lz
  Kz, Ky, Kx = [a.flatten() for a in np.meshgrid(kvec_z, kvec_y, kvec_x, indexing='ij')]
  Ksq = Kx**2 + Ky**2 + Kz**2
  # precompute parts of RHS
  rhs = np.divide((1j * Kx * f_hat + 1j * Ky * g_hat + 1j * Kz * h_hat), Ksq, \
                  out = np.zeros_like(f_hat), where = Ksq != 0, dtype = np.complex) 
  I2 = np.divide(1, eta * Ksq, out = np.zeros_like(Ksq), where = Ksq != 0, dtype = np.double)
  # solve for Fourier coeffs of velocity
  u_hat = I2 * (f_hat + (1j * Kx * rhs))
  v_hat = I2 * (g_hat + (1j * Ky * rhs))
  w_hat = I2 * (h_hat + (1j * Kz * rhs))
  # ignore k = 0
  u_hat[0] = 0
  v_hat[0] = 0
  w_hat[0] = 0
  # interleave solution components and split
  # real/imaginary parts for passing back to c
  U_hat_r = np.zeros((Ntotal,), dtype = np.double)
  U_hat_r[0::3] = np.real(u_hat)
  U_hat_r[1::3] = np.real(v_hat)
  U_hat_r[2::3] = np.real(w_hat)
  U_hat_i = np.zeros((Ntotal,), dtype = np.double)
  U_hat_i[0::3] = np.imag(u_hat)
  U_hat_i[1::3] = np.imag(v_hat)
  U_hat_i[2::3] = np.imag(w_hat)
  return U_hat_r, U_hat_i

# Precomputations for all DP solvers
def DoublyPeriodicStokes_init(Nx, Ny, Nz, Lx, Ly, H):
  """
  Precompute the linear operators and boundary conditions
  for the doubly periodic no wall problem. The return
  values of this function are used in the no_wall,
  bottom_wall and slit_channel solvers

  This function is useful when several solves are done
  on the same grid, as it only needs to be called
  before the first solve eg) at t=0 in a time stepping context.

  Parameters:
    Nx, Ny, Nz - number of points in x,y,z
    Lx, Ly - extent of x and y grids
    H - half extent of z grid (Lz / 2)
  
  Side Effects: None
  Returns:
    Kx, Ky, K - meshgrid of wavenumbers and K = sqrt(Kx^2 + Ky^2)
    Dx, Dy - Fourier derivative operator iKx (with 0 middle mode if N is even)
    SIMat - second Chebyshev integral matrix (Nz x Nz + 2)
    FIMat - first Chebyshev integral matrix (Nz x Nz + 2)
    pints - precomputed integrals of Chebyshev polynomials for pressure correction
          - this is only used in DP no_wall if the k0 flag is 1
    uvints - precomputed integrals of Chebyshev polynomials for velocity correction
           - this is only used in DP no_wall if the k0 flag is 1
    BCs_k0 - Boundary conditions for k = 0 in DP no-wall
    BCs_k - Boundary conditions for each k != 0 in DP no-wall
    LU, Ainv_B, C, PIV, Ginv - Components of matrices in BVP for each k != 0 
                       - See precomputeBandedLinOps and its side effects for details
    C_k0, Ginv_k0 - analog of C, Ginv above for k = 0
    BCR1, BCL1, BCR2, BCL2 - See DoublyPeriodicNoWallBCs for details
                           - this is only used in the bottom_wall and 
                           - slit_channel routines
  """
  # wave numbers
  kvec_x = 2*np.pi*np.concatenate((np.arange(0,np.floor(Nx/2)),\
                                   np.arange(-1*np.ceil(Nx/2),0)), axis=None) / Lx
  
  kvec_y = 2*np.pi*np.concatenate((np.arange(0,np.floor(Ny/2)),\
                                   np.arange(-1*np.ceil(Ny/2),0)), axis=None) / Ly
  Ky, Kx = [K.reshape((Ny * Nx,)).copy() for K in np.meshgrid(kvec_y, kvec_x, indexing='ij')]
  Ksq = Kx**2 + Ky**2; K = np.sqrt(Ksq)
  Dx = 1j * Kx; Dy = 1j * Ky; 
  # zero unpaired mode
  if Nx % 2 == 0: 
    Dx.reshape((Ny,Nx))[:,int(Nx/2)] = 0
  if Ny % 2 == 0:
    Dy.reshape((Ny,Nx))[int(Ny/2),:] = 0
  # get Chebyshev integration matrices
  FIMat = firstIntegralMatrix(Nz, H)
  SIMat = secondIntegralMatrix(Nz, H)
  # precompute integrals of cheb polys for pressure/vel correction
  pints, uvints = precomputeInts(Nz, H)
  # get boundary conditions rows
  BCR1, BCR2, BCL1, BCL2 = DoublyPeriodic_no_wall_BCs(Nz)
  # assemble appropriate BCs for k = 0 and k != 0
  # and permute to ijk with fortran storage
  BCs_k0 = np.asfortranarray(np.stack((BCR2, -BCL2), axis = 0))
  BCs_k = np.asfortranarray(
            np.transpose(
              BCs_k0 + H * np.stack((BCR1, BCL1), axis = 0) + \
              np.einsum('ij, k->kij', np.stack((BCR2, BCL2), axis = 0), H**2 * K) + \
              np.stack((-BCR2, BCL2), axis = 0), (1,2,0)))
  # precompute system matrices for each k and permute to ijk with fortran storage
  A = np.asfortranarray(
        np.transpose(
          tobanded(np.eye(Nz), 2, 2, np.double) -\
          np.einsum('ij, k->kij', tobanded(SIMat[:,:Nz], 2, 2, np.double), Ksq),\
          (1,2,0)))
  B = np.asfortranarray(
        np.transpose(
          np.einsum('ij, k->kij', SIMat[:,Nz::], -Ksq),(1,2,0)))
  C = np.asfortranarray(BCs_k[:,0:Nz,:]); C_k0 = BCs_k0[:,0:Nz]; 
  D = np.asfortranarray(BCs_k[:,Nz::,:]); D_k0 = -BCs_k0[:,Nz::]; 
  # precompute LU decompositions and precomputable linear solves for each k
  Ainv_B = np.zeros((Nz, 2, Ny * Nx))
  PIV = np.asfortranarray(np.zeros((Nz, Ny * Nx), dtype = np.int32))
  Ginv = np.asfortranarray(np.zeros((2, 2, Ny * Nx), dtype = np.double))
  G = np.asfortranarray(np.zeros((2, 2, Ny * Nx), dtype = np.double))
  # get LU decomposition of A for each k and Ainv * B
  # A is overwritten with LU and B is overwritten with Ainv * B
  # Ginv is the inverse of the 2x2 schur complement
  precomputeBandedLinOps(A, B, C, D, G, Ginv, PIV, 2, 2, Ny * Nx, Nz)
  # rename overwritten vars
  LU = A; Ainv_B = B
  # get Ginv for k=0
  Ginv_k0 = 1.0 / (D_k0[0,0] * D_k0[1,1] - D_k0[0,1] * D_k0[1,0]) * \
                   np.array([[D_k0[1,1], -D_k0[0,1]], [-D_k0[1,0],D_k0[0,0]]])
  return Kx, Ky, K, Dx, Dy, FIMat, SIMat, pints, uvints,\
         BCs_k0, BCs_k, LU, Ainv_B, C, PIV, C_k0, Ginv,\
         Ginv_k0, BCR1, BCL1, BCR2, BCL2 

# DP no wall solver
def DoublyPeriodicStokes_no_wall(fG_hat_r, fG_hat_i, eta, Nx, Ny, Nz, H,\
                                 Kx, Ky, K, Dx, Dy, FIMat, SIMat, pints, \
                                 uvints, BCs_k0, BCs_k, LU, Ainv_B, C, \
                                 PIV, C_k0, Ginv, Ginv_k0, k0):
  """
  Solve doubly periodic Stokes eq in Fourier-Chebyshev domain given the Fourier-Chebyshev
  coefficients of the forcing.
  
  Parameters:
    fG_hat_r, fG_hat_i - real and complex part of Fourier-Chebyshev coefficients
                         of spread forces on the grid. These are both
                         arrays of doubles (not complex).
  
    eta - viscocity
    Lx, Ly, Lz - length of unit cell in x,y,z
    Nx, Ny, Nz - number of points in x, y and z
    H - Lz / 2
    Kx, Ky, K - meshgrid of wavenumbers and K = sqrt(Kx^2 + Ky^2)
    Dx, Dy - Fourier derivative operator iKx (with 0 middle mode if N is even)
    SIMat - second Chebyshev integral matrix (Nz x Nz + 2)
    FIMat - first Chebyshev integral matrix (Nz x Nz + 2)
    pints - precomputed integrals of Chebyshev polynomials for pressure correction
          - this is only used in DP no_wall if the k0 flag is 1
    uvints - precomputed integrals of Chebyshev polynomials for velocity correction
           - this is only used in DP no_wall if the k0 flag is 1
    BCs_k0 - Boundary conditions for k = 0 in DP no-wall
    BCs_k - Boundary conditions for each k != 0 in DP no-wall
    LU, Ainv_B, C, PIV, Ginv - Components of matrices in BVP for each k != 0 
                       - See precomputeBandedLinOps and its side effects for details
    C_k0, Ginv_k0 - analog of C, Ginv above for k = 0
    k0 - switch determining how to handle k = 0 mode of pressure and velocity
       - if k0 = 0, k = 0 mode of RHS is assumed to be 0, and sol will be 0
       - if k0 = 1, the k = 0 mode of RHS is assumed to be non-zero. There
         will be a correction to the non-zero solution.
  
  Returns:
    U_hat_r, U_hat_i, P_hat_r, P_hat_i - real and complex part of 
                                       - Fourier-Chebyshev coefficients of
                                         fluid velocity and pressure on the grid. 
  """
  dof = 3;
  # separate x,y,z components
  Cf = np.asfortranarray((fG_hat_r[:,:,:,0] + 1j * fG_hat_i[:,:,:,0]).reshape((Nz, Ny * Nx)))
  Cg = np.asfortranarray((fG_hat_r[:,:,:,1] + 1j * fG_hat_i[:,:,:,1]).reshape((Nz, Ny * Nx)))
  Ch = np.asfortranarray((fG_hat_r[:,:,:,2] + 1j * fG_hat_i[:,:,:,2]).reshape((Nz, Ny * Nx)))
  Dh = np.asfortranarray(chebCoeffDiff(Ch, Nx, Ny, Nz, 1, H).reshape((Nz, Ny * Nx)))
 
  # compute RHS of pressure poisson eq
  p_RHS = Dx * Cf + Dy * Cg + Dh
    
  # solve for pressure, handling k=0 separately
  Cp, Dp = DoublyPeriodicStokes_no_wall_solvePressureBVP_k(p_RHS, LU, PIV, C, Ainv_B, Ginv, SIMat, FIMat)

  # handle k = 0 for pressure
  Cp[:,0], Dp[:,0] = DoublyPeriodicStokes_no_wall_solvePressureBVP_k0(\
                      p_RHS[:,0], C_k0, Ginv_k0, SIMat, FIMat, \
                      Ch[:,0], pints, k0)
  
  # compute RHS for velocity solve
  u_RHS = (Dx * Cp - Cf) / eta
  v_RHS = (Dy * Cp - Cg) / eta
  w_RHS = (Dp - Ch) / eta
  # precompute last two rows of RHS in velocity solve
  fac1 = 2 * eta; fac2 = fac1 * K; indpow = np.power(-1.0, np.arange(0,Nz))
  Cp_sum = np.sum(Cp, axis = 0); Cp_alt_sum = np.sum(Cp * indpow.reshape((Nz,1)), axis = 0) 

  u_bc_RHS = np.divide(np.stack((-Dx * Cp_sum, Dx * Cp_alt_sum), axis = 0), fac2, where = fac2 != 0)
  v_bc_RHS = np.divide(np.stack((-Dy * Cp_sum, Dy * Cp_alt_sum), axis = 0), fac2, where = fac2 != 0)
  w_bc_RHS = np.stack((Cp_sum, Cp_alt_sum), axis = 0) / fac1

  # solve for velocity, handling k = 0 separately
  Cu, Cv, Cw = DoublyPeriodicStokes_no_wall_solveVelocityBVP_k(\
                u_RHS, v_RHS, w_RHS, LU, PIV, C, \
                Ainv_B, Ginv, SIMat, u_bc_RHS, \
                v_bc_RHS, w_bc_RHS)


  # handle k = 0 for velocity
  Cu[:,0], Cv[:,0], Cw[:,0] = DoublyPeriodic_no_wall_solveVelocityBVP_k0(\
                                u_RHS[:,0], v_RHS[:,0], w_RHS[:,0], C_k0,\
                                Ginv_k0, SIMat, Cf[:,0], Cg[:,0], uvints, eta, k0)
  # interleave solution components and split
  # real/imaginary parts for passing back to c
  U_hat_r = np.zeros((Nz * Ny * Nx * dof,), dtype = np.double)
  U_hat_r[0::3] = np.real(Cu).reshape((Nz * Ny * Nx,))
  U_hat_r[1::3] = np.real(Cv).reshape((Nz * Ny * Nx,))
  U_hat_r[2::3] = np.real(Cw).reshape((Nz * Ny * Nx,))
  U_hat_i = np.zeros((Nz * Ny * Nx * dof,), dtype = np.double)
  U_hat_i[0::3] = np.imag(Cu).reshape((Nz * Ny * Nx,))
  U_hat_i[1::3] = np.imag(Cv).reshape((Nz * Ny * Nx,))
  U_hat_i[2::3] = np.imag(Cw).reshape((Nz * Ny * Nx,))
  P_hat_r = np.real(Cp).reshape((Nz * Ny * Nx,))
  P_hat_i = np.imag(Cp).reshape((Nz * Ny * Nx,))
  return U_hat_r, U_hat_i, P_hat_r, P_hat_i

# DP bottom wall solver
def DoublyPeriodicStokes_bottom_wall(fG_hat_r, fG_hat_i, zpts, eta, Nx, Ny, Nz, H,\
                                     Kx, Ky, K, Dx, Dy, FIMat, SIMat, pints, \
                                     uvints, BCs_k0, BCs_k, LU, Ainv_B, C, \
                                     PIV, C_k0, Ginv, Ginv_k0, BCR1, BCL2):
  """
  Solve Stokes eq in doubly periodic bottom wall in the Fourier-Chebyshev domain 
  given the Fourier-Chebyshev coefficients of the forcing. We first solve a DP
  subproblem (no walls), and then compute a correction to the DP solution to enforce
  the no-slip BCs.
  
  Parameters:
    fG_hat_r, fG_hat_i - real and complex part of Fourier-Chebyshev coefficients
                         of spread forces on the grid. These are both
                         arrays of doubles (not complex).
  
    eta - viscocity
    Lx, Ly, Lz - length of unit cell in x,y,z
    Nx, Ny, Nz - number of points in x, y and z
    H - Lz / 2
    Kx, Ky, K - meshgrid of wavenumbers and K = sqrt(Kx^2 + Ky^2)
    Dx, Dy - Fourier derivative operator iKx (with 0 middle mode if N is even)
    SIMat - second Chebyshev integral matrix (Nz x Nz + 2)
    FIMat - first Chebyshev integral matrix (Nz x Nz + 2)
    pints - precomputed integrals of Chebyshev polynomials for pressure correction
          - this is only used in DP no_wall if the k0 flag is 1
    uvints - precomputed integrals of Chebyshev polynomials for velocity correction
           - this is only used in DP no_wall if the k0 flag is 1
    BCs_k0 - Boundary conditions for k = 0 in DP no-wall
    BCs_k - Boundary conditions for each k != 0 in DP no-wall
    LU, Ainv_B, C, PIV, Ginv - Components of matrices in BVP for each k != 0 
                       - See precomputeBandedLinOps and its side effects for details
    C_k0, Ginv_k0 - analog of C, Ginv above for k = 0
    BCR1, BCL2 - See DoublyPeriodicNoWallBCs for details
  
  Returns:
    U_hat_r, U_hat_i, P_hat_r, P_hat_i - real and complex part of 
                                       - Fourier-Chebyshev coefficients of
                                         fluid velocity and pressure on the grid. 
  """
  dof = 3; Nyx = Ny * Nx;
  # solve the doubly periodic problem, ignoring k=0
  U_hat_r, U_hat_i, P_hat_r, P_hat_i = DoublyPeriodicStokes_no_wall(fG_hat_r, fG_hat_i, eta, Nx, Ny, Nz, H, \
                                                                    Kx, Ky, K, Dx, Dy, FIMat, SIMat, pints, \
                                                                    uvints, BCs_k0, BCs_k, LU, Ainv_B, C, \
                                                                    PIV, C_k0, Ginv, Ginv_k0, 0)
  
  # Cheb coeffs of forcing for k = 0
  Cf_k0 = fG_hat_r[:,0,0,0] + 1j * fG_hat_i[:,0,0,0]
  Cg_k0 = fG_hat_r[:,0,0,1] + 1j * fG_hat_i[:,0,0,1]
  Ch_k0 = fG_hat_r[:,0,0,2] + 1j * fG_hat_i[:,0,0,2]
  Dh_k0 = chebCoeffDiff(Ch_k0, 1, 1, Nz, 1, H).reshape((Nz,))
  # First cheb coeff of pressure for k = 0
  Cp_k0 = P_hat_r[0] + 1j * P_hat_i[0]
  Dp_k0 = chebCoeffDiff(P_hat_r[0::Ny * Nx] + 1j * P_hat_i[0::Ny * Nx], 1, 1, Nz, 1, H).reshape((Nz,)) 
  # RHS for k = 0 correction solve for pressure and vel
  p_RHS_k0 = Dh_k0
  u_RHS_k0 = -Cf_k0 / eta
  v_RHS_k0 = -Cg_k0 / eta
  w_RHS_k0 = (Dp_k0 - Ch_k0) / eta

  # get negative of velocities at bottom wall for BCs of correction problem
  Cubw_r = -1.0 * evalTheta(U_hat_r, np.pi, Nyx, Nz, dof)
  Cubw_i = -1.0 * evalTheta(U_hat_i, np.pi, Nyx, Nz, dof)
  # compute the correction field for k != 0 
  Cpcorr, Cucorr, Cvcorr, Cwcorr = \
    evalCorrectionSol_bottomWall(Cubw_r, Cubw_i, zpts, Kx, Ky, eta, Nx, Ny, Nz, dof)
  # correct pressure for k = 0
  Cpcorr[:,0] = DoublyPeriodicStokes_wall_solvePressureBVP_k0(\
                   p_RHS_k0, C_k0, Ginv_k0, SIMat, Ch_k0, pints, Cp_k0)
  # correct velocities for k = 0
  BCs_k0_bw = np.asfortranarray(np.stack((BCR1, BCL2), axis = 0))
  C_k0_bw = BCs_k0_bw[:,0:Nz]
  D_k0_bw = -BCs_k0_bw[:,Nz::]
 
  Ginv_k0_bw = 1.0 / (D_k0_bw[0,0] * D_k0_bw[1,1] - D_k0_bw[0,1] * D_k0_bw[1,0]) * \
                   np.array([[D_k0_bw[1,1], -D_k0_bw[0,1]], [-D_k0_bw[1,0],D_k0_bw[0,0]]])
  
  Cucorr[:,0], Cvcorr[:,0] = DoublyPeriodic_wall_solveVelocityBVP_k0(\
                               u_RHS_k0, v_RHS_k0, w_RHS_k0, C_k0_bw,\
                               Ginv_k0_bw, SIMat, Cf_k0, Cg_k0)
  # add the solutions to the two subproblems
  P_hat_r += np.real(Cpcorr.reshape((Nz * Nyx,))); P_hat_i += np.imag(Cpcorr.reshape((Nz * Nyx)))
  U_hat_r[0::3] += np.real(Cucorr.reshape((Nz * Nyx,))); U_hat_i[0::3] += np.imag(Cucorr.reshape((Nz * Nyx,)))
  U_hat_r[1::3] += np.real(Cvcorr.reshape((Nz * Nyx,))); U_hat_i[1::3] += np.imag(Cvcorr.reshape((Nz * Nyx,)))
  U_hat_r[2::3] += np.real(Cwcorr.reshape((Nz * Nyx,))); U_hat_i[2::3] += np.imag(Cwcorr.reshape((Nz * Nyx,)))
  return U_hat_r, U_hat_i, P_hat_r, P_hat_i  

# DP slit channel solver
def DoublyPeriodicStokes_slit_channel(fG_hat_r, fG_hat_i, zpts, eta, Nx, Ny, Nz, H,\
                                      Kx, Ky, K, Dx, Dy, FIMat, SIMat, pints, \
                                      uvints, BCs_k0, BCs_k, LU, Ainv_B, C, \
                                      PIV, C_k0, Ginv, Ginv_k0, BCR2, BCL2):
  """
  Solve Stokes eq in doubly periodic slit channel in the Fourier-Chebyshev domain 
  given the Fourier-Chebyshev coefficients of the forcing. We first solve a DP
  subproblem (no walls), and then compute a correction to the DP solution to enforce
  the no-slip BCs.
  
  Parameters:
    fG_hat_r, fG_hat_i - real and complex part of Fourier-Chebyshev coefficients
                         of spread forces on the grid. These are both
                         arrays of doubles (not complex).
  
    eta - viscocity
    Lx, Ly, Lz - length of unit cell in x,y,z
    Nx, Ny, Nz - number of points in x, y and z
    H - Lz / 2
    Kx, Ky, K - meshgrid of wavenumbers and K = sqrt(Kx^2 + Ky^2)
    Dx, Dy - Fourier derivative operator iKx (with 0 middle mode if N is even)
    SIMat - second Chebyshev integral matrix (Nz x Nz + 2)
    FIMat - first Chebyshev integral matrix (Nz x Nz + 2)
    pints - precomputed integrals of Chebyshev polynomials for pressure correction
          - this is only used in DP no_wall if the k0 flag is 1
    uvints - precomputed integrals of Chebyshev polynomials for velocity correction
           - this is only used in DP no_wall if the k0 flag is 1
    BCs_k0 - Boundary conditions for k = 0 in DP no-wall
    BCs_k - Boundary conditions for each k != 0 in DP no-wall
    LU, Ainv_B, C, PIV, Ginv - Components of matrices in BVP for each k != 0 
                       - See precomputeBandedLinOps and its side effects for details
    C_k0, Ginv_k0 - analog of C, Ginv above for k = 0
    BCR2, BCL2 - See DoublyPeriodicNoWallBCs for details
  
  Returns:
    U_hat_r, U_hat_i, P_hat_r, P_hat_i - real and complex part of 
                                       - Fourier-Chebyshev coefficients of
                                         fluid velocity and pressure on the grid. 
  """
  dof = 3; Nyx = Ny * Nx; Lz = 2 * H
  # solve the doubly periodic problem, ignoring k=0
  U_hat_r, U_hat_i, P_hat_r, P_hat_i = DoublyPeriodicStokes_no_wall(fG_hat_r, fG_hat_i, eta, Nx, Ny, Nz, H, \
                                                                    Kx, Ky, K, Dx, Dy, FIMat, SIMat, pints, \
                                                                    uvints, BCs_k0, BCs_k, LU, Ainv_B, C, \
                                                                    PIV, C_k0, Ginv, Ginv_k0, 0)

  # Cheb coeffs of forcing for k = 0
  Cf_k0 = fG_hat_r[:,0,0,0] + 1j * fG_hat_i[:,0,0,0]
  Cg_k0 = fG_hat_r[:,0,0,1] + 1j * fG_hat_i[:,0,0,1]
  Ch_k0 = fG_hat_r[:,0,0,2] + 1j * fG_hat_i[:,0,0,2]
  Dh_k0 = chebCoeffDiff(Ch_k0, 1, 1, Nz, 1, H).reshape((Nz,))
  # First cheb coeff of pressure for k = 0
  Cp_k0 = P_hat_r[0] + 1j * P_hat_i[0]
  Dp_k0 = chebCoeffDiff(P_hat_r[0::Ny * Nx] + 1j * P_hat_i[0::Ny * Nx], 1, 1, Nz, 1, H).reshape((Nz,)) 
  # RHS for k = 0 correction solve for pressure and vel
  p_RHS_k0 = Dh_k0
  u_RHS_k0 = -Cf_k0 / eta
  v_RHS_k0 = -Cg_k0 / eta
  w_RHS_k0 = (Dp_k0 - Ch_k0) / eta
  # get negative of velocities at walls for BCs of correction problem
  Cubw_r = -1.0 * evalTheta(U_hat_r, np.pi, Nyx, Nz, dof)
  Cubw_i = -1.0 * evalTheta(U_hat_i, np.pi, Nyx, Nz, dof)
  Cutw_r = -1.0 * evalTheta(U_hat_r, 0, Nyx, Nz, dof)
  Cutw_i = -1.0 * evalTheta(U_hat_i, 0, Nyx, Nz, dof)
  # compute the correction field for k != 0 
  Cpcorr, Cucorr, Cvcorr, Cwcorr = \
    evalCorrectionSol_slitChannel(Cubw_r, Cubw_i, Cutw_r, Cutw_i, zpts, Kx, Ky, eta, Lz, Nx, Ny, Nz, dof)

  # correct pressure for k = 0
  Cpcorr[:,0] = DoublyPeriodicStokes_wall_solvePressureBVP_k0(\
                   p_RHS_k0, C_k0, Ginv_k0, SIMat, Ch_k0, pints, Cp_k0)
  #print(np.sum(np.abs(Cpcorr)));
  # correct velocities for k = 0
  BCs_k0_tw = np.asfortranarray(np.stack((BCR2, BCL2), axis = 0))
  C_k0_tw = BCs_k0_tw[:,0:Nz]
  D_k0_tw = -BCs_k0_tw[:,Nz::]
  
  Ginv_k0_tw = 1.0 / (D_k0_tw[0,0] * D_k0_tw[1,1] - D_k0_tw[0,1] * D_k0_tw[1,0]) * \
                   np.array([[D_k0_tw[1,1], -D_k0_tw[0,1]], [-D_k0_tw[1,0],D_k0_tw[0,0]]])
  
  Cucorr[:,0], Cvcorr[:,0] = DoublyPeriodic_wall_solveVelocityBVP_k0(\
                               u_RHS_k0, v_RHS_k0, w_RHS_k0, C_k0_tw,\
                               Ginv_k0_tw, SIMat, Cf_k0, Cg_k0)
  # add the solutions to the two subproblems
  P_hat_r += np.real(Cpcorr.reshape((Nz * Nyx,))); P_hat_i += np.imag(Cpcorr.reshape((Nz * Nyx)))
  U_hat_r[0::3] += np.real(Cucorr.reshape((Nz * Nyx,))); U_hat_i[0::3] += np.imag(Cucorr.reshape((Nz * Nyx,)))
  U_hat_r[1::3] += np.real(Cvcorr.reshape((Nz * Nyx,))); U_hat_i[1::3] += np.imag(Cvcorr.reshape((Nz * Nyx,)))
  U_hat_r[2::3] += np.real(Cwcorr.reshape((Nz * Nyx,))); U_hat_i[2::3] += np.imag(Cwcorr.reshape((Nz * Nyx,)))
  return U_hat_r, U_hat_i, P_hat_r, P_hat_i  

# BCs for DP problem (not usually called externally)
def DoublyPeriodic_no_wall_BCs(N):
  """
  This function gives the following BCs for the 
  no-wall Doubly periodic Stokes BVPs in z at each wave number:
  BCR1 = first derivative (first integral) evaluated at x=1
  BCR2 = function (second integral) evaluated at x=1
  BCL1 = first derivative (first integral) evaluated at x=-1
  BCL2 = NEGATIVE OF function (second integral) evaluated at x=-1

  Parameters:
    N - number of Chebyshev points
  Returns:
    BCR1, BCR2, BCL1, BCL2 - BCs for DP no-wall BVP solve in z (to be assembled with k) 
  """
  BCR1 = np.zeros((N+2,), dtype = np.double)
  BCR2 = np.zeros((N+2,), dtype = np.double)
  BCL1 = np.zeros((N+2,), dtype = np.double)
  BCL2 = np.zeros((N+2,), dtype = np.double)
  # Special cases - right
  BCR1[N+1] = 1; BCR2[N] = 1; BCR1[0] = 1; BCR1[2] = -1/2;
  BCR2[N+1] += 1; BCR2[1] = -1/8; BCR2[3] = 1/8;
  BCR1[1] += 1/4; BCR1[3] -= 1/4; BCR2[0] += 1/4; BCR2[2] -= (1/8 + 1/24); 
  BCR2[4] += 1/24;
  # Special cases -left 
  BCL1[N+1] = 1; BCL2[N] = -1; BCL1[0] = -1; BCL1[2] = 1/2;  
  BCL2[N+1] += 1; BCL2[1] = -1/8; BCL2[3] = 1/8;  
  BCL1[1] += 1/4; BCL1[3] -= 1/4; BCL2[0] -= 1/4; BCL2[2] += 1/8 + 1/24;  
  BCL2[4] -= 1/24;  
  # Easy cases    
  jj = np.arange(3,N)
  BCR1[2:N-1] += 1 / (2 * jj)  
  BCL1[2:N-1] += np.power(-1, jj) / (2 * jj)  
  BCR1[4:N+1] -= 1 / (2 * jj) * (jj < N - 1)
  BCL1[4:N+1] -= np.power(-1, jj) / (2 * jj) * (jj < N - 1)    
  BCR2[1:N-2] += 1 / (2 * jj) * 1 / (2 * jj - 2)
  BCL2[1:N-2] -= 1 / (2 * jj) * 1 / (2 * jj - 2) * np.power(-1, jj)  
  BCR2[5:N+2] += 1 / (2 * jj) * 1 / (2 * jj + 2) * (jj < N - 2)
  BCL2[5:N+2] -= 1 / (2 * jj) * 1 / (2 * jj + 2) * np.power(-1, jj) * (jj < N - 2) 
  BCR2[3:N] -= (1 / (2 * jj) * 1 / (2 * jj - 2) + 1 / (2 * jj) * 1 / (2 * jj + 2) * (jj < N - 1))
  BCL2[3:N] += (1 / (2 * jj) * 1 / (2 * jj - 2) + 1 / (2 * jj) * 1 / (2 * jj + 2) * (jj < N - 1)) * np.power(-1, jj)
  return BCR1, BCR2, BCL1, BCL2 

###################################################################################
####################### Pressure solve subroutines ################################
###################################################################################

def DoublyPeriodicStokes_no_wall_solvePressureBVP_k(p_RHS, LU, PIV, C, Ainv_B, Ginv, SIMat, FIMat):
  """
  This function solves the pressure poisson equation in the doubly periodic domain for k > 0
  Lap(p) = div(f)

  Parameters:
    p_RHS - Nz x Nyx Fourier-Chebyshev coeffs of div(f)
    LU - (2 * kl + ku + 1) x Nz x Nyx tensor containing LU decomp of first block of solve
         matrix for each k
    PIV - Nz x Nyx pivot information to reconstruct first block of solve matrix for each k 
    C - 2 x Nz x Nyx third block of solve matrix for each k
    Ainv_B - Nz x 2 x Nyx A^{-1} B where B is second block of solve matrix for each k
    Ginv - 2 x 2 x Nyx equal to (C A^{-1} B - D)^{-1} for each k
    SIMat - second Chebyshev integral matrix (Nz x Nz + 2)
    FIMat - first Chebyshev integral matrix  (Nz x Nz + 2)
   
  Returns:
    Cp, Dp - Fourier-Chebyshev coeffs of pressure and its derivative 

  """ 
  Nz, Nyx = p_RHS.shape
  p_RHS_real = np.asfortranarray(np.real(p_RHS))
  p_RHS_imag = np.asfortranarray(np.imag(p_RHS))
  bandedSchurSolve(LU, p_RHS_real, PIV, 2, 2, Nyx, Nz)
  bandedSchurSolve(LU, p_RHS_imag, PIV, 2, 2, Nyx, Nz)
  Ainv_F = p_RHS_real + 1j * p_RHS_imag
  Y = np.einsum('ijk, jk->ik', Ginv,\
        np.einsum('ijk, jk->ik', C, Ainv_F))
  X = Ainv_F - np.einsum('ijk, jk->ik', Ainv_B, Y)
  SecD = np.vstack((X,Y))
  Cp = np.einsum('ij, jl->il', SIMat, SecD)
  Dp = np.einsum('ij, jl->il', FIMat, SecD)
  Cp[:,0] = Dp[:,0] = 0
  return Cp, Dp

def DoublyPeriodicStokes_no_wall_solvePressureBVP_k0(p_RHS, C, Ginv, SIMat, FIMat, Ch, pints, k0):
  """
  This function solves the pressure poisson equation for k = 0, based on the switch k0.

  Parameters:
    p_RHS - Nz x 1 Fourier-Chebyshev coeffs of div(f) at k = 0
    C - 2 x Nz third block of solve matrix for k = 0 (see DoublyPeriodicStokes_no_wall() for details)
    Ginv - 2 x 2 inverse of schur complement of first block of solve matrix at k = 0
    SIMat - second Chebyshev integral matrix (Nz x Nz + 2)
    FIMat - first Chebyshev integral matrix (Nz x Nz + 2)
    Ch - Fourier-Chebyshev coeffs of z-component of f at k = 0
    pints - precomputed integrals of Chebyshev polynomials for pressure correction
    k0 - switch to handle k = 0 mode 
       - if k0 = 0, k = 0 mode of RHS is assumed to be 0, and sol will be 0
       - if k0 = 1, the k = 0 mode of RHS is assumed to be non-zero. There
         will be a correction to the non-zero solution.

  Returns:
    Cp, Dp - Fourier-Chebyshev coeffs of pressure and its derivative at k = 0
  """
  Nz = p_RHS.shape[0]
  # sol will be 0 if k=0 of RHS is 0
  if k0 == 0:
    Cp = np.zeros((Nz,))
    Dp = np.zeros((Nz,))
  # if k=0 of RHS is not 0, we solve the pressure poblem with homogenous BCs
  # and must correct the 2nd Chebyshev coefficient of that mode (see eq. 18 in Ondrej's report)
  elif k0 == 1:
    secD = np.concatenate((p_RHS, Ginv @ (C @ p_RHS)))
    Cp = SIMat @ secD
    Dp = FIMat @ secD
    Cp[1] += 0.5 * pints @ Ch
  return Cp, Dp

def DoublyPeriodicStokes_wall_solvePressureBVP_k0(p_RHS, C, Ginv, SIMat, Ch, pints, cp0):
  """
  This function gives the correction for k=0 of pressure in the DP+correction method.
  It works for either bottom wall or slit channel.

  Parameters:
    p_RHS - Nz x 1 Fourier-Chebyshev coeffs of div(f) at k = 0
    C - 2 x Nz third block of solve matrix for k = 0 (see DoublyPeriodicStokes_no_wall() for details)
    Ginv - 2 x 2 inverse of schur complement of first block of solve matrix at k = 0
    SIMat - second Chebyshev integral matrix (Nz x Nz + 2)
    Ch - Fourier-Chebyshev coeffs of z-component of f at k = 0
    pints - precomputed integrals of Chebyshev polynomials for pressure correction
    cp0 - coeff of first cheb poly for k = 0 component of the DP subproblem
  Returns:
    Cp, Dp - Fourier-Chebyshev coeffs of pressure and its derivative at k = 0
  """

  Nz = p_RHS.shape[0]
  secD = np.concatenate((p_RHS, Ginv @ (C @ p_RHS)))
  Cp = SIMat @ secD
  Cp[1] += 0.5 * pints @ Ch
  Cp[0] = cp0 + 0.5 * pints @ Ch
  return Cp

###################################################################################
####################### Velocity solve subroutines ################################
###################################################################################

def DoublyPeriodicStokes_no_wall_solveVelocityBVP_k(u_RHS, v_RHS, w_RHS, LU, PIV, C, \
                                                    Ainv_B, Ginv, SIMat, u_bc_RHS, \
                                                    v_bc_RHS, w_bc_RHS):
  """
  This function solves the velocity BVP in the doubly periodic domain for k > 0

  Parameters:
    u_RHS - Nz x Nyx Fourier-Cheb coeffs of (dp/dx - f_x) / eta
    v_RHS - Nz x Nyx Fourier-Cheb coeffs of (dp/dy - f_y) / eta
    w_RHS - Nz x Nyx Fourier-Cheb coeffs of (dp/dz - f_z) / eta
    LU - (2 * kl + ku + 1) x Nz x Nyx tensor containing LU decomp of first block of solve
         matrix for each k
    PIV - Nz x Nyx pivot information to reconstruct first block of solve matrix for each k 
    C - 2 x Nz x Nyx third block of solve matrix for each k
    Ainv_B - Nz x 2 x Nyx A^{-1} B where B is second block of solve matrix for each k
    Ginv - 2 x 2 x Nyx equal to (C A^{-1} B - D)^{-1} for each k
    SIMat - second Chebyshev integral matrix (Nz x Nz + 2)
    FIMat - first Chebyshev integral matrix  (Nz x Nz + 2)
    u_bc_RHS, v_bc_RHS, w_bc_RHS - 2 x Nyx RHS (alpha,beta) of linear system at each k
   
  Returns:
    Cu, Cv, Cw - Fourier-Chebyshev coeffs of velocity components
  """
  Nz, Nyx = u_RHS.shape
  u_RHS_real = np.asfortranarray(np.real(u_RHS))
  u_RHS_imag = np.asfortranarray(np.imag(u_RHS))
  v_RHS_real = np.asfortranarray(np.real(v_RHS))
  v_RHS_imag = np.asfortranarray(np.imag(v_RHS))
  w_RHS_real = np.asfortranarray(np.real(w_RHS))
  w_RHS_imag = np.asfortranarray(np.imag(w_RHS))
  bandedSchurSolve(LU, u_RHS_real, PIV, 2, 2, Nyx, Nz)
  bandedSchurSolve(LU, u_RHS_imag, PIV, 2, 2, Nyx, Nz)
  bandedSchurSolve(LU, v_RHS_real, PIV, 2, 2, Nyx, Nz)
  bandedSchurSolve(LU, v_RHS_imag, PIV, 2, 2, Nyx, Nz)
  bandedSchurSolve(LU, w_RHS_real, PIV, 2, 2, Nyx, Nz)
  bandedSchurSolve(LU, w_RHS_imag, PIV, 2, 2, Nyx, Nz)
  # x vel
  Ainv_F = u_RHS_real + 1j * u_RHS_imag
  Y = np.einsum('ijk, jk->ik', Ginv,\
        np.einsum('ijk, jk->ik', C, Ainv_F) - u_bc_RHS)
  X = Ainv_F - np.einsum('ijk, jk->ik', Ainv_B, Y)
  SecD = np.vstack((X,Y))
  Cu = np.einsum('ij, jl->il', SIMat, SecD)
  # y vel
  Ainv_F = v_RHS_real + 1j * v_RHS_imag
  Y = np.einsum('ijk, jk->ik', Ginv,\
        np.einsum('ijk, jk->ik', C, Ainv_F) - v_bc_RHS)
  X = Ainv_F - np.einsum('ijk, jk->ik', Ainv_B, Y)
  SecD = np.vstack((X,Y))
  Cv = np.einsum('ij, jl->il', SIMat, SecD)
  # z vel
  Ainv_F = w_RHS_real + 1j * w_RHS_imag
  Y = np.einsum('ijk, jk->ik', Ginv,\
        np.einsum('ijk, jk->ik', C, Ainv_F) - w_bc_RHS)
  X = Ainv_F - np.einsum('ijk, jk->ik', Ainv_B, Y)
  SecD = np.vstack((X,Y))
  Cw = np.einsum('ij, jl->il', SIMat, SecD)
  Cu[:,0] = Cv[:,0] = Cw[:,0] = 0
  return Cu, Cv, Cw                                               

def DoublyPeriodic_no_wall_solveVelocityBVP_k0(u_RHS, v_RHS, w_RHS, C, Ginv, SIMat,\
                                               Cf, Cg, uvints, eta, k0):
  """
  This function solves the velocity BVPs for k = 0, based on the switch k0.

  Parameters:
    u_RHS, v_RHS, w_RHS - Nz x 1 Fourier-Chebyshev coeffs of div(f) at k = 0
    C - 2 x Nz third block of solve matrix for k = 0 (see DoublyPeriodicStokes_no_wall() for details)
    Ginv - 2 x 2 inverse of schur complement of first block of solve matrix at k = 0
    SIMat - second Chebyshev integral matrix (Nz x Nz + 2)
    FIMat - first Chebyshev integral matrix (Nz x Nz + 2)
    Cf, Cg - Fourier-Chebyshev coeffs of x and y components of f at k = 0
    uvints - precomputed integrals of Chebyshev polynomials for velocity correction
    k0 - switch to handle k = 0 mode 
       - if k0 = 0, k = 0 mode of RHS is assumed to be 0, and sol will be 0
       - if k0 = 1, the k = 0 mode of RHS is assumed to be non-zero. There
         will be a correction to the non-zero solution.

  Returns:
    Cu, Cv, Cw - Fourier-Chebyshev coeffs of velocity components at k = 0
  """
  Nz = u_RHS.shape[0]
  # sol will be 0 if k=0 of RHS is 0
  if k0 == 0:
    Cu = np.zeros((Nz,))
    Cv = np.zeros((Nz,))
    Cw = np.zeros((Nz,))
  # if k=0 of RHS is not 0, we solve with homogenous BCs and
  # add the linear in z term for the null space   
  elif k0 == 1:
    au = np.concatenate((u_RHS, Ginv @ (C @ u_RHS)))
    av = np.concatenate((v_RHS, Ginv @ (C @ v_RHS)))
    aw = np.concatenate((w_RHS, Ginv @ (C @ w_RHS)))
    Cu = SIMat @ au
    Cv = SIMat @ av
    Cw = SIMat @ aw
    Cu[1] += 0.5 / eta * uvints @ Cf
    Cv[1] += 0.5 / eta * uvints @ Cg
  return Cu, Cv, Cw

def DoublyPeriodic_wall_solveVelocityBVP_k0(u_RHS, v_RHS, w_RHS, C, Ginv, SIMat,Cf, Cg):
  """
  This function gives the correction for k=0 of the velocity in the DP+correction method.
  It works for either bottom wall or slit channel.

  Parameters:
    u_RHS, v_RHS, w_RHS - Nz x 1 Fourier-Chebyshev coeffs of div(f) at k = 0
    C - 2 x Nz third block of solve matrix for k = 0 (see DoublyPeriodicStokes_no_wall() for details)
    Ginv - 2 x 2 inverse of schur complement of first block of solve matrix at k = 0
    SIMat - second Chebyshev integral matrix (Nz x Nz + 2)
    FIMat - first Chebyshev integral matrix (Nz x Nz + 2)
    Cf, Cg - Fourier-Chebyshev coeffs of x and y components of f at k = 0

  Returns:
    Cu, Cv - Fourier-Chebyshev coeffs of velocity components at k = 0
  """
  Nz = u_RHS.shape[0]
  au = np.concatenate((u_RHS, Ginv @ (C @ u_RHS)))
  av = np.concatenate((v_RHS, Ginv @ (C @ v_RHS)))
  Cu = SIMat @ au
  Cv = SIMat @ av
  return Cu, Cv


###################################################################################
####################### DP wall correction subroutines ############################
###################################################################################

# define wrappers for dptools
def evalTheta(phi_in, theta, Nyx, Nz, dof):
  """
    This function is used to evaluate a Chebyshev series at a given
    value of theta (point on the cheb grid)
    
    Parameters: 
      in - the input array (size (Nz * Nyx * dof,1))
         - these are the Fourier-Chebyshev coeffs on the grid
      theta - determines the slice in z
            - eg) theta = pi is z = 0, theta = 0 is z - Lz 
      Nyx - total number of points in x,y
      Nz  - number of points in z
      dof - degrees of freedom
    
    Side Effects: None
    
    Returns : Phi_out - the output array (size (Nyx * dof, 1))
                      - these are the Fourier-Chebyshev coeffs on the x-y
                      - plane at a given z value 
  """
  phi_out = np.zeros((Nyx * dof,), dtype = np.double)
  libDPTools.evalTheta(phi_in.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),\
                       phi_out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),\
                       theta, Nyx, Nz, dof)
  return phi_out

def evalCorrectionSol_bottomWall(Cu_r, Cu_i, zpts, Kx, Ky, eta, Nx, Ny, Nz, dof):
  """ 
  Evaluate the analytical correction to the DP solve to enforce no-slip 
  BCs at the bottom wall (calls c lib)
 
  Parameters : 
    Cu_r, Cu_i - real and complex part of Fourier-Cheb coeffs at bottom wall
    zpts - Chebyshev points in z
    Kx, Ky - meshgrid of wave numbers in x,y
    eta - viscosity
    Nx, Ny, Nz - num points in x,y,z
    dof - degrees of freedom   
  
  Side Effects : None
  Returns : C(p,u,v,w)corr - combined real/complex Fourier-Chebyshev 
                           - correction for pressure and velocity
                           - the k = 0 element is 0
                           - shape (Ny * Nx, Nz)
  """
  Nyx = Ny * Nx;
  Cpcorr_r = np.zeros((Nyx * Nz, 1), dtype = np.double) 
  Cpcorr_i = np.zeros((Nyx * Nz, 1), dtype = np.double) 
  Cucorr_r = np.zeros((Nyx * Nz, 1), dtype = np.double) 
  Cucorr_i = np.zeros((Nyx * Nz, 1), dtype = np.double) 
  Cvcorr_r = np.zeros((Nyx * Nz, 1), dtype = np.double) 
  Cvcorr_i = np.zeros((Nyx * Nz, 1), dtype = np.double) 
  Cwcorr_r = np.zeros((Nyx * Nz, 1), dtype = np.double) 
  Cwcorr_i = np.zeros((Nyx * Nz, 1), dtype = np.double) 
  libDPTools.evalCorrectionSol_bottomWall(Cpcorr_r.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),\
                                          Cpcorr_i.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),\
                                          Cucorr_r.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),\
                                          Cucorr_i.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),\
                                          Cvcorr_r.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),\
                                          Cvcorr_i.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),\
                                          Cwcorr_r.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),\
                                          Cwcorr_i.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),\
                                          Cu_r.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),\
                                          Cu_i.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),\
                                          Kx.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),\
                                          Ky.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),\
                                          zpts.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),\
                                          eta, Nyx, Nz, dof)
  Cpcorr = np.asfortranarray(np.transpose(np.reshape(Cpcorr_r + 1j * Cpcorr_i, (Ny * Nx, Nz)), (1,0)))
  Cucorr = np.asfortranarray(np.transpose(np.reshape(Cucorr_r + 1j * Cucorr_i, (Ny * Nx, Nz)), (1,0))) 
  Cvcorr = np.asfortranarray(np.transpose(np.reshape(Cvcorr_r + 1j * Cvcorr_i, (Ny * Nx, Nz)), (1,0))) 
  Cwcorr = np.asfortranarray(np.transpose(np.reshape(Cwcorr_r + 1j * Cwcorr_i, (Ny * Nx, Nz)), (1,0))) 
  return Cpcorr, Cucorr, Cvcorr, Cwcorr

def evalCorrectionSol_slitChannel(Cub_r, Cub_i, Cut_r, Cut_i, zpts, Kx, Ky, eta, Lz, Nx, Ny, Nz, dof):
  """ 
  Evaluate the analytical correction to the DP solve to enforce no-slip 
  BCs at the bottom and top wall (calls c lib)
 
  Parameters : 
    Cub_r, Cub_i - real and complex part of Fourier-Cheb coeffs at bottom wall
    Cut_r, Cut_i - real and complex part of Fourier-Cheb coeffs at top wall
    zpts - Chebyshev points in z
    Kx, Ky - meshgrid of wave numbers in x,y
    eta - viscosity
    Lz - extent of z grid
    Nx, Ny, Nz - num points in x,y,z
    dof - degrees of freedom   
  
  Side Effects : None
  Returns : C(p,u,v,w)corr - combined real/complex Fourier-Chebyshev 
                           - correction for pressure and velocity
                           - the k = 0 element is 0
                           - shape (Ny * Nx, Nz)
  """
  Nyx = Ny * Nx;
  Cpcorr_r = np.zeros((Nyx * Nz, 1), dtype = np.double) 
  Cpcorr_i = np.zeros((Nyx * Nz, 1), dtype = np.double) 
  Cucorr_r = np.zeros((Nyx * Nz, 1), dtype = np.double) 
  Cucorr_i = np.zeros((Nyx * Nz, 1), dtype = np.double) 
  Cvcorr_r = np.zeros((Nyx * Nz, 1), dtype = np.double) 
  Cvcorr_i = np.zeros((Nyx * Nz, 1), dtype = np.double) 
  Cwcorr_r = np.zeros((Nyx * Nz, 1), dtype = np.double) 
  Cwcorr_i = np.zeros((Nyx * Nz, 1), dtype = np.double) 
  libDPTools.evalCorrectionSol_slitChannel(Cpcorr_r.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),\
                                          Cpcorr_i.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),\
                                          Cucorr_r.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),\
                                          Cucorr_i.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),\
                                          Cvcorr_r.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),\
                                          Cvcorr_i.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),\
                                          Cwcorr_r.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),\
                                          Cwcorr_i.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),\
                                          Cub_r.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),\
                                          Cub_i.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),\
                                          Cut_r.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),\
                                          Cut_i.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),\
                                          Kx.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),\
                                          Ky.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),\
                                          zpts.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),\
                                          Lz, eta, Nyx, Nz, dof)
  Cpcorr = np.asfortranarray(np.transpose(np.reshape(Cpcorr_r + 1j * Cpcorr_i, (Ny * Nx, Nz)), (1,0)))
  Cucorr = np.asfortranarray(np.transpose(np.reshape(Cucorr_r + 1j * Cucorr_i, (Ny * Nx, Nz)), (1,0))) 
  Cvcorr = np.asfortranarray(np.transpose(np.reshape(Cvcorr_r + 1j * Cvcorr_i, (Ny * Nx, Nz)), (1,0))) 
  Cwcorr = np.asfortranarray(np.transpose(np.reshape(Cwcorr_r + 1j * Cwcorr_i, (Ny * Nx, Nz)), (1,0))) 
  return Cpcorr, Cucorr, Cvcorr, Cwcorr


###################################################################################
####################### Wrappers and helpers ######################################
###################################################################################

# define wrappers for linear solvers
def precomputeBandedLinOps(A, B, C, D, G, Ginv, PIV, kl, ku, Nyx, Nz):
  """
  Given a block linear system of the form 
  
  |A B||a |   |f|
  |C D||c0| = |alpha| 
       |d0|   |beta|,  
              
  
  where A is Nz x Nz and banded, B is Nz x 2, C is 2 x Nz and D is 2x2 (all real), 
  this function computes the inverse of the schur complement of A,
    i.e. Ginv = (C A^{-1} B - D)^{-1},
  as well as the LU decomposition of A and A^{-1}B by making calls to LAPACK's
  general banded solve dgbsv routines. This does it for every k \in [0, Ny * Nx]

  Parameters:
    A - (2 * kl + ku + 1) x Nz x Nyx tensor of diagonals 
        this is stored in LAPACK's banded matrix format in Fortran order
    B - Nz x 2 x Nyx tensor (stored in Fortran order)
    C - 2 x Nz x Nyx tensor (stored in Fortran order)
    D - 2 x 2 x Nyx tensor (stored in Fortran order)
    G - 2 x 2 x Nyx tensor of zeros (stored in Fortran order)
    Ginv - 2 x 2 x Nyx tensor of zeros (stored in Fortran order)
    PIV - Nz x Nyx matrix of zeros
    kl - number of lower diagonals 
    ku - number of upper diagonals
    Nyx - Nx * Ny (total points in x-y plane)
    Nz -  num points in z
    
  Side Effects:
    A is overwritten with its LU decomposition
    PIV contains the pivoting information for reconstructing A from LU
    B is overwritten with A^{-1}B (from LAPACK solve)
    G is overwritten with (C A^{-1} B - D)
    Ginv is overwritten with G^{-1} explicitly computed using 2x2 inv formula

    A, B and PIV can be reused for future solves with LAPACK's dgbtrs routine,
    eg) they are passed to bandedSchurSolve

  """
  libLinSolve.precomputeBandedLinOps(A.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),\
                                     B.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),\
                                     C.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),\
                                     D.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),\
                                     G.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),\
                                     Ginv.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),\
                                     PIV.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),\
                                     kl, ku, Nyx, Nz)

def bandedSchurSolve(LU, RHS, PIV, kl, ku, Nyx, Nz):
  """
  Given a block linear system of the form 
  
  |A B||a |   |f|
  |C D||c0| = |alpha| 
       |d0|   |beta|,  
              
  this function takes the LU decomposition of A and computes
  A^{-1}f by making calls to LAPACK's general banded LU solve
  routines (dgbtrs) for each k.

  Parameters:
    LU - (2 * kl + ku + 1) x Nz x Nyx tensor of diagonals containing LU decomposition
         of A. This is stored in LAPACK's banded matrix format in Fortran order
    RHS - Nz x Nyx matrix containing f stored in Fortran order
    PIV - Nz x Nyx matrix containing pivot info obtained by precomputeBandedLinOps()
    kl - number of lower diagonals 
    ku - number of upper diagonals
    Nyx - Nx * Ny (total points in x-y plane)
    Nz -  num points in z
    
  Side Effects:
    RHS is overwritten with A^{-1}RHS (from LAPACK solve)
  """
  libLinSolve.bandedSchurSolve(LU.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),\
                               RHS.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),\
                               PIV.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),\
                               kl, ku, Nyx, Nz)

def tobanded(A, kl, ku, _dtype):
  """ 
  A is an n x n ndarray
  This function converts a square matrix to diagonal ordered form
  for use with lapack banded routines (eg. dgbsv, dgbtrs)
  
  Parameters:
    A - n x n ndarray
    kl - number of lower diagonals
    ku - number of upper diagonals
    _dtype - type of the output array

  Returns:
    Ab - (2*kl + ku + 1) x n array of diagonals
  """
  n = A.shape[1]
  ldab = 2 * kl + ku + 1
  Ab = np.zeros((ldab,n), dtype = _dtype, order='F')
  for j in range(1, n + 1): 
    for i in range(max(1, j - ku), min(n, j + kl) + 1): 
      Ab[kl + ku + 1 + i - j - 1, j - 1] = A[i-1, j-1]
  return Ab

###################################################################################
##################### Library function declarations ###############################
###################################################################################

# declare lin solver lib funcs
libLinSolve.precomputeBandedLinOps.argtypes = [ctypes.POINTER(ctypes.c_double),\
                                               ctypes.POINTER(ctypes.c_double),\
                                               ctypes.POINTER(ctypes.c_double),\
                                               ctypes.POINTER(ctypes.c_double),\
                                               ctypes.POINTER(ctypes.c_double),\
                                               ctypes.POINTER(ctypes.c_double),\
                                               ctypes.POINTER(ctypes.c_int),\
                                               ctypes.c_int, ctypes.c_int,\
                                               ctypes.c_int, ctypes.c_int]
libLinSolve.precomputeBandedLinOps.restype = None

libLinSolve.bandedSchurSolve.argtypes = [ctypes.POINTER(ctypes.c_double),\
                                         ctypes.POINTER(ctypes.c_double),\
                                         ctypes.POINTER(ctypes.c_int),\
                                         ctypes.c_int, ctypes.c_int,\
                                         ctypes.c_int, ctypes.c_int]
libLinSolve.bandedSchurSolve.restype = None

# declare dptools lib funcs
libDPTools.evalTheta.argtypes = [ctypes.POINTER(ctypes.c_double),\
                                 ctypes.POINTER(ctypes.c_double),\
                                 ctypes.c_double, ctypes.c_uint,\
                                 ctypes.c_uint, ctypes.c_uint]
libDPTools.evalTheta.restype = None

libDPTools.evalCorrectionSol_bottomWall.argtypes = [ctypes.POINTER(ctypes.c_double),\
                                                    ctypes.POINTER(ctypes.c_double),\
                                                    ctypes.POINTER(ctypes.c_double),\
                                                    ctypes.POINTER(ctypes.c_double),\
                                                    ctypes.POINTER(ctypes.c_double),\
                                                    ctypes.POINTER(ctypes.c_double),\
                                                    ctypes.POINTER(ctypes.c_double),\
                                                    ctypes.POINTER(ctypes.c_double),\
                                                    ctypes.POINTER(ctypes.c_double),\
                                                    ctypes.POINTER(ctypes.c_double),\
                                                    ctypes.POINTER(ctypes.c_double),\
                                                    ctypes.POINTER(ctypes.c_double),\
                                                    ctypes.POINTER(ctypes.c_double),\
                                                    ctypes.c_double, ctypes.c_uint,\
                                                    ctypes.c_uint, ctypes.c_uint]
libDPTools.evalCorrectionSol_bottomWall.restype = None

libDPTools.evalCorrectionSol_slitChannel.argtypes = [ctypes.POINTER(ctypes.c_double),\
                                                     ctypes.POINTER(ctypes.c_double),\
                                                     ctypes.POINTER(ctypes.c_double),\
                                                     ctypes.POINTER(ctypes.c_double),\
                                                     ctypes.POINTER(ctypes.c_double),\
                                                     ctypes.POINTER(ctypes.c_double),\
                                                     ctypes.POINTER(ctypes.c_double),\
                                                     ctypes.POINTER(ctypes.c_double),\
                                                     ctypes.POINTER(ctypes.c_double),\
                                                     ctypes.POINTER(ctypes.c_double),\
                                                     ctypes.POINTER(ctypes.c_double),\
                                                     ctypes.POINTER(ctypes.c_double),\
                                                     ctypes.POINTER(ctypes.c_double),\
                                                     ctypes.POINTER(ctypes.c_double),\
                                                     ctypes.POINTER(ctypes.c_double),\
                                                     ctypes.c_double, ctypes.c_double,
                                                     ctypes.c_uint, ctypes.c_uint,\
                                                     ctypes.c_uint]
libDPTools.evalCorrectionSol_slitChannel.restype = None

