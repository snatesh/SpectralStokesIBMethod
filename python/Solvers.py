import numpy as np

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
  
  Note: We ensure the net force on the unit cell is 0 by *ignoring* 
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
  U_hat_r = np.empty((Ntotal,), dtype = np.double)
  U_hat_r[0::3] = np.real(u_hat)
  U_hat_r[1::3] = np.real(v_hat)
  U_hat_r[2::3] = np.real(w_hat)
  U_hat_i = np.empty((Ntotal,), dtype = np.double)
  U_hat_i[0::3] = np.imag(u_hat)
  U_hat_i[1::3] = np.imag(v_hat)
  U_hat_i[2::3] = np.imag(w_hat)
  return U_hat_r, U_hat_i
