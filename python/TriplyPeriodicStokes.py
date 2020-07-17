from Grid import *
from Species import *
from SpreadInterp import *
from Transform import *

# grid info 
Nx = 64; Ny = 64; Nz = 32; dof = 3 
hx = 0.5; hy = 0.5; hz = 0.5; 
Lx = Nx * hx; Ly = Ny * hy; Lz = Nz * hz; 
# number of particles
nP = 100
# viscocity
eta = 1/4/np.sqrt(np.pi)
# instantiate grid wrapper
gridGen = GridGen(Lx, Ly, Lz, hx, hy, hz, Nx, Ny, Nz, dof)
# make triply periodic grid
grid = gridGen.MakeTP()
# instantiate species wrapper
speciesGen = SpeciesGen(nP)
# make random configuration of particles in terms of size, force and position
species = speciesGen.RandomConfig(grid)

# spread forces on the particles
fG = Spread(species, grid, gridGen.Ntotal)
# write the grid with spread to file
gridGen.WriteGrid(grid, 'spread.txt')
gridGen.WriteCoords(grid, 'coords.txt')  

# instantiate transform wrapper with spread forces
Tf = Transformer(fG, None, Nx, Ny, Nz, dof)
# compute forward transform
Forward = Tf.Ftransform()
# get the Fourier coefficients
fG_hat_r = Tf.GetRealOut(Forward)
fG_hat_i = Tf.GetComplexOut(Forward)
# separate x,y,z components
f_hat = fG_hat_r[0::3] + 1j * fG_hat_i[0::3]
g_hat = fG_hat_r[1::3] + 1j * fG_hat_i[1::3]
h_hat = fG_hat_r[2::3] + 1j * fG_hat_i[2::3]

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
u_hat = I2 * (f_hat - (1j * Kx * rhs))
v_hat = I2 * (g_hat - (1j * Ky * rhs))
w_hat = I2 * (h_hat - (1j * Kz * rhs))

# ignore k = 0
u_hat[0] = 0
v_hat[0] = 0
w_hat[0] = 0
# interleave solution components and split
# real/imaginary parts before passing back to c
U_hat_r = np.empty((Tf.Ntotal,), dtype = np.double)
U_hat_r[0::3] = np.real(u_hat)
U_hat_r[1::3] = np.real(v_hat)
U_hat_r[2::3] = np.real(w_hat)
U_hat_i = np.empty((Tf.Ntotal,), dtype = np.double)
U_hat_i[0::3] = np.imag(u_hat)
U_hat_i[1::3] = np.imag(v_hat)
U_hat_i[2::3] = np.imag(w_hat)

# instantiate back transform wrapper with velocities on grid
Tb = Transformer(U_hat_r, U_hat_i, Nx, Ny, Nz, dof)
Backward = Tb.Btransform()
# get real part of back transform and normalize
uG_r = Tb.GetRealOut(Backward) / Tb.N
uG_i = Tb.GetComplexOut(Backward) / Tb.N
print(np.min(uG_i))

# set velocity as new grid spread
gridGen.SetGridSpread(grid, uG_r)

# interpolate velocities on the particles
vP = Interpolate(species, grid, nP * dof)

# write species with interpolated vel to file
speciesGen.WriteSpecies(species, 'particles.txt')
# write grid velocities
gridGen.WriteGrid(grid, 'velocities.txt')

# free memory persisting b/w C and python
Tf.Clean(Forward)
Tf.Delete(Forward)
Tb.Clean(Backward)
Tb.Delete(Backward)
gridGen.Clean(grid)
gridGen.Delete(grid)
speciesGen.Clean(species)
speciesGen.Delete(species)
