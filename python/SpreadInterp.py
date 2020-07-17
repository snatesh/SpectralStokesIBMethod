import ctypes
import numpy as np

libSpreadInterp = ctypes.CDLL('../lib/libspreadInterp.so')

libSpreadInterp.Spread.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
libSpreadInterp.Spread.restype = ctypes.POINTER(ctypes.c_double)
libSpreadInterp.Interpolate.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
libSpreadInterp.Interpolate.restype = ctypes.POINTER(ctypes.c_double)

def Spread(s,g, N):
  return np.ctypeslib.as_array(libSpreadInterp.Spread(s,g), shape=(N, ))

def Interpolate(s,g, N):
  return np.ctypeslib.as_array(libSpreadInterp.Interpolate(s,g), shape=(N, ))
