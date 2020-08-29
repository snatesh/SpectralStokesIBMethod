import ctypes
import numpy as np

"""
Python wrappers for C library Spread/Interp routines

See "extern" in SpreadInterp.h.

The prototypes for relevant functions from the 
C++ SpreadInterp library are declared. Any functions added
to the "extern" definition in SpreadInterp.h should be
declared here.
"""
libSpreadInterp = ctypes.CDLL('../lib/libspreadInterp.so')

libSpreadInterp.Spread.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
libSpreadInterp.Spread.restype = ctypes.POINTER(ctypes.c_double)
libSpreadInterp.Interpolate.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
libSpreadInterp.Interpolate.restype = ctypes.POINTER(ctypes.c_double)

def Spread(s, g, N):
  """
  Spread data from the particles s onto the grid g.
  
  Parameters:
    s - a pointer to the C++ ParticleList struct
    g - a pointer to the C++ Grid struct
    N - total number of elements (Nx * Ny * Nz * dof)
  
  Returns:
    fG - a flat numpy array containing the spread data (Nz*Ny*Nx*dof,1) 
  
  Side Effects:
    The C++ Grid data member g.fG is populated with the spread data
  """
  return np.ctypeslib.as_array(libSpreadInterp.Spread(s,g), shape=(N, ))

def Interpolate(s, g, N):
  """
  Interpolate data from the grid g onto the particles s.
  
  Parameters:
    s - a pointer to the C++ ParticleList struct
    g - a pointer to the C++ Grid struct
    N - total number of elements (nP * dof)
  
  Returns:
    fP - a flat numpy array containing the interpolated data on the particles (nP*dof,1)
  
  Side Effects:
    The C++ ParticleList data member s.fP is populated with the interpolated data
  """
  return np.ctypeslib.as_array(libSpreadInterp.Interpolate(s,g), shape=(N, ))
