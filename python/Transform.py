import ctypes
import numpy as np

# Python wrappers for C library Transform routines
# This class can be thought of as a generator and
# manager of the underlying C++ Transform struct

# See "extern" in Transform.h

# TODO: Maybe combine real/complex output in C
#       using interleaved format so we don't need
#       get() functions in python wrapper
#       Then, python useage would just look like
#       
#       fG = Spread(species, grid, Ntotal)
#       fG_hat = Transform(fG, Nx, Ny, Nz, dof)
#       fG_hat_r = fG_hat[0::2]
#       fG_hat_i = fG_hat[1::2]
#       fG_back = Transform(fG_hat_r, fG_hat_i, Nx, Ny, Nz, dof)

libTransform = ctypes.CDLL('../lib/libtransform.so')

class Transformer(object):
  def __init__(self, _in_real, _in_complex, _Nx, _Ny, _Nz, _dof):
    
    # define the prototypes of the external C interface 
    # Any functions added to the "extern" definition in
    # Transform.h should be defined here
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

  # Python wrapper for the Ftransform(...) C lib routine
  # This computes the forward plan and executes a forward
  # transform on the input data, assuming that it is real.
  # It returns a pointer to the Transform struct 
  def Ftransform(self):
    return libTransform.Ftransform(self.in_real.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),\
                                   self.Nx, self.Ny, self.Nz, self.dof)

  # Python wrapper for the Btransform(...) C lib routine
  # This computes the backward plan and executes a backward
  # transform on the real+complex input data.
  # It returns a pointer to the Transform struct 
  def Btransform(self):
    return libTransform.Btransform(self.in_real.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),\
                                   self.in_complex.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), \
                                   self.Nx, self.Ny, self.Nz, self.dof)
  # Python wrapper for the getRealOut(..) C lib routine
  # This returns a numpy array of the real output
  def GetRealOut(self,obj):
    return np.ctypeslib.as_array(libTransform.getRealOut(obj), shape=(self.Ntotal,))
  
  # Python wrapper for the getComplexOut(..) C lib routine
  # This returns a numpy array of the complex output
  def GetComplexOut(self,obj):
    return np.ctypeslib.as_array(libTransform.getComplexOut(obj), shape=(self.Ntotal,))

  # Python wrapper for the CleanTransform(..) C lib routine
  # This cleans the Transform struct returned by B/Ftransform 
  # and frees any memory internally allocated 
  def Clean(self,obj):
    libTransform.CleanTransform(obj)

  # Python wrapper for the DeleteTransform(..) C lib routine
  # This deletes the pointer returned by B/Ftransform
  def Delete(self,obj):
    libTransform.DeleteTransform(obj)
