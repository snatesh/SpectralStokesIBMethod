import ctypes
import numpy as np

# Python wrappers for C library Transform routines

libTransform = ctypes.CDLL('../lib/libtransform.so')

class Transformer(object):
  def __init__(self, _in_real, _in_complex, _Nx, _Ny, _Nz, _dof):
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

    self.in_real = _in_real
    self.in_complex = _in_complex
    self.Nx = _Nx
    self.Ny = _Ny
    self.Nz = _Nz
    self.dof = _dof
    self.N = self.Nx * self.Ny * self.Nz
    self.Ntotal = self.N * self.dof

  def Ftransform(self):
    return libTransform.Ftransform(self.in_real.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),\
                                   self.Nx, self.Ny, self.Nz, self.dof)

  def Btransform(self):
    return libTransform.Btransform(self.in_real.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),\
                                   self.in_complex.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), \
                                   self.Nx, self.Ny, self.Nz, self.dof)

  def GetRealOut(self,obj):
    return np.ctypeslib.as_array(libTransform.getRealOut(obj), shape=(self.Ntotal,))
  
  def GetComplexOut(self,obj):
    return np.ctypeslib.as_array(libTransform.getComplexOut(obj), shape=(self.Ntotal,))

  def Clean(self,obj):
    libTransform.CleanTransform(obj)

  def Delete(self,obj):
    libTransform.DeleteTransform(obj)
