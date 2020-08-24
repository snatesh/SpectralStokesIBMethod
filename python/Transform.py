import ctypes
import numpy as np
libTransform = ctypes.CDLL('../lib/libtransform.so')

class Transformer(object):
  """
  Python wrappers for C library Transform routines.
  
  This class can be thought of as a generator and
  manager of the underlying C++ Transform struct.
  
  See "extern" in Transform.h.
  
  Attributes:
    in_real (double array) - real part of the input data.
    in_complex (double array or None) - complex part of the input data.
    Nx, Ny, Nz - number of points in x, y and z.
    N - Nx * Ny * Nz.
    dof - degrees of freedom of the data.
    Ntotal - N * dof.
    out_real (double array) - real part of output transform.
    out_complex (double array) - complex part of output transform.
    transform (ptr to C++ struct) - a pointer to the generated C++ Transform struct
  """
  def __init__(self, _in_real, _in_complex, _Nx, _Ny, _Nz, _dof):
    """ 
    The constructor for the Transformer class.
    
    Parameters:
      in_real (doubles) - real part of input.
      in_complex (doubles) - complex part of input.
      Nx, Ny, Nz (int) - number of points in x,y,z
      dof (int) - degrees of freedom.

    Side Effects:
      The prototypes for relevant functions from the 
      C++ Transform library are declared. Any functions added
      to the "extern" definition in Transform.h should be
      declared here. The attributes out_real, out_complex 
      and transform are set to None.
    """ 
    libTransform.Ftransform.argtypes = [ctypes.POINTER(ctypes.c_double), \
                                        ctypes.c_uint, ctypes.c_uint, \
                                        ctypes.c_uint, ctypes.c_uint]
    libTransform.Ftransform.restype = ctypes.c_void_p

    libTransform.Btransform.argtypes = [ctypes.POINTER(ctypes.c_double), \
                                        ctypes.POINTER(ctypes.c_double), \
                                        ctypes.c_uint, ctypes.c_uint, \
                                        ctypes.c_uint, ctypes.c_uint]
    libTransform.Btransform.restype = ctypes.c_void_p

    libTransform.CleanTransform.argtypes = [ctypes.c_void_p]
    libTransform.CleanTransform.restype = None
  
    libTransform.DeleteTransform.argtypes = [ctypes.c_void_p]
    libTransform.DeleteTransform.restype = None

    libTransform.getRealOut.argtypes = [ctypes.c_void_p]
    libTransform.getRealOut.restype = ctypes.POINTER(ctypes.c_double)

    libTransform.getComplexOut.argtypes = [ctypes.c_void_p]
    libTransform.getComplexOut.restype = ctypes.POINTER(ctypes.c_double)

    # real part of input data (double array)
    self.in_real = _in_real
    # complex part of input data (double array)
    self.in_complex = _in_complex
    # number of points in x,y,z
    self.Nx = _Nx
    self.Ny = _Ny
    self.Nz = _Nz
    # degrees of freedom
    self.dof = _dof
    # get total nums
    self.N = self.Nx * self.Ny * self.Nz
    self.Ntotal = self.N * self.dof
    # pointer to c++ Transform struct
    self.transform = None
    # outputs
    self.out_real = None
    self.out_complex = None
  

  def Ftransform(self):
    """
    Python wrapper for the Ftransform(...) C lib routine.

    This computes the forward plan and executes a forward
    transform on the input data, assuming that it is real.

    Parameters: None
    Side Effects:
      self.transform is assigned the pointer to the C++ Transform instance
      self.out_real is populated with the real part of the output transform
      self.out_complex is populated with the complex part of the output transform

    """
    self.transform = libTransform.Ftransform(self.in_real.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),\
                                             self.Nx, self.Ny, self.Nz, self.dof)
    self.out_real = np.ctypeslib.as_array(libTransform.getRealOut(self.transform), shape=(self.Ntotal,))
    self.out_complex = np.ctypeslib.as_array(libTransform.getComplexOut(self.transform), shape=(self.Ntotal,))
  
  def Ftransform_cheb(self):
    """
    Python wrapper for the Ftransform(...) C lib routine when
    the z axis is Chebyshev.

    This computes the forward plan and executes a forward
    transform on the input data, assuming that it is real.
    The results in out_real/out_complex will be the Fourier-Chebyshev
    coefficients of the input

    Parameters: None
    Side Effects:
      self.transform is assigned the pointer to the C++ Transform instance
      self.in_real is replaced by itself with the z axis periodically extended (doubled and flipped)
      self.out_real is populated with the real part of the output transform
      self.out_complex is populated with the complex part of the output transform

    """
    U = np.empty(((2 * self.Nz - 2) * self.Ny * self.Nx * self.dof,), dtype = np.double).reshape(\
          ((2 * self.Nz - 2), self.Ny, self.Nx, self.dof))
    in_real_rs = np.reshape(self.in_real, (self.Nz, self.Ny, self.Nx, self.dof))
    U[0:self.Nz,:,:,:] = in_real_rs
    U[self.Nz::,:,:,:] = in_real_rs[-2:0:-1,:,:,:]
    self.in_real = np.reshape(U, (self.dof * self.Nx * self.Ny * (2 * self.Nz - 2),)) 
    self.transform = libTransform.Ftransform(self.in_real.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),\
                                             self.Nx, self.Ny, 2 * self.Nz - 2, self.dof)
    _out_real = np.ctypeslib.as_array(libTransform.getRealOut(self.transform), shape=(2 * self.Nz - 2, self.Ny, self.Nx, self.dof))
    _out_complex = np.ctypeslib.as_array(libTransform.getComplexOut(self.transform), shape=(2 * self.Nz - 2, self.Ny, self.Nx, self.dof))
    self.out_real = np.empty((self.Nz, self.Ny, self.Nx, self.dof), dtype = np.double)
    self.out_complex = np.empty((self.Nz, self.Ny, self.Nx, self.dof), dtype = np.double)
    self.out_real[0,:,:,:] = _out_real[0,:,:,:]
    self.out_real[1:-1,:,:,:] = _out_real[1:self.Nz-1,:,:,:] + _out_real[-1:self.Nz-1:-1,:,:,:]
    self.out_real[-1,:,:,:] = _out_real[self.Nz-1,:,:,:]
    self.out_complex[0,:,:,:] = _out_complex[0,:,:,:]
    self.out_complex[1:-1,:,:,:] = _out_complex[1:self.Nz-1,:,:,:] + _out_complex[-1:self.Nz-1:-1,:,:,:]
    self.out_complex[-1,:,:,:] = _out_complex[self.Nz-1,:,:,:]
    self.out_real /= (2 * self.Nz - 2)
    self.out_complex /= (2 * self.Nz - 2)
  def Btransform(self):
    """
    Python wrapper for the Btransform(...) C lib routine.

    This computes the backward plan and executes a backward
    transform on the input data. The output is normalized 
    by self.N

    Parameters: None
    Side Effects:
      self.transform is assigned the pointer to the C++ Transform instance
      self.out_real is populated with the real part of the output transform
      self.out_complex is populated with the complex part of the output transform
    """
    self.transform = libTransform.Btransform(self.in_real.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),\
                                             self.in_complex.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), \
                                             self.Nx, self.Ny, self.Nz, self.dof)
    self.out_real = np.ctypeslib.as_array(libTransform.getRealOut(self.transform), shape=(self.Ntotal,)) / self.N
    self.out_complex = np.ctypeslib.as_array(libTransform.getComplexOut(self.transform), shape=(self.Ntotal,)) / self.N

  def Btransform_cheb(self):
    Ur = np.empty(((2 * self.Nz - 2) * self.Ny * self.Nx * self.dof,), dtype = np.double).reshape(\
          ((2 * self.Nz - 2), self.Ny, self.Nx, self.dof))
    Uc = np.empty(((2 * self.Nz - 2) * self.Ny * self.Nx * self.dof,), dtype = np.double).reshape(\
          ((2 * self.Nz - 2), self.Ny, self.Nx, self.dof))
    in_real_rs = self.in_real.reshape((self.Nz, self.Ny, self.Nx, self.dof))
    in_complex_rs = self.in_complex.reshape((self.Nz, self.Ny, self.Nx, self.dof))
    Ur[0,:,:,:] = in_real_rs[0,:,:,:]
    Ur[1:self.Nz-1,:,:,:] = in_real_rs[1:-1,:,:,:] / 2
    Ur[self.Nz-1,:,:,:] = in_real_rs[-1,:,:,:]
    Ur[self.Nz::,:,:,:] = in_real_rs[-2:0:-1,:,:,:] / 2
    Ur *= (2 * self.Nz - 2)
    Uc[0,:,:,:] = in_complex_rs[0,:,:,:]
    Uc[1:self.Nz-1,:,:,:] = in_complex_rs[1:-1,:,:,:] / 2
    Uc[self.Nz-1,:,:,:] = in_complex_rs[-1,:,:,:]
    Uc[self.Nz::,:,:,:] = in_complex_rs[-2:0:-1,:,:,:] / 2
    Uc *= (2 * self.Nz - 2)
    self.in_real = Ur.reshape((self.dof * self.Nx * self.Ny * (2 * self.Nz - 2),))
    self.in_complex = Uc.reshape((self.dof * self.Nx * self.Ny * (2 * self.Nz - 2),))
    self.transform = libTransform.Btransform(self.in_real.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),\
                                             self.in_complex.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), \
                                             self.Nx, self.Ny, 2 * self.Nz - 2, self.dof)
    _out_real = np.ctypeslib.as_array(libTransform.getRealOut(self.transform), shape=(2 * self.Nz - 2, self.Ny, self.Nx, self.dof))
    _out_complex = np.ctypeslib.as_array(libTransform.getComplexOut(self.transform), shape=(2 * self.Nz - 2, self.Ny, self.Nx, self.dof)) 
    self.out_real = _out_real[0:self.Nz,:,:,:].reshape((self.Ntotal,)) / (self.Nx * self.Ny * (2 * self.Nz - 2))
    self.out_complex = _out_complex[0:self.Nz,:,:,:].reshape((self.Ntotal,)) / (self.Nx * self.Ny * (2 * self.Nz - 2))

  def Clean(self):
    """
    Python wrapper for the CleanTransform(..) C lib routine.
    This cleans the Transform struct returned by B/Ftransform,
    frees any memory internally allocated, and deletes the
    pointer to the transform struct stored in the class.

    Parameters: None
    Side Effects:
      self.transform is deleted and nullified
      self.out_real is deleted and nullified
      self.out_complex is deleted and nullified
    """
    libTransform.CleanTransform(self.transform)
    libTransform.DeleteTransform(self.transform)
