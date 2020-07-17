import ctypes

libSpecies = ctypes.CDLL('../lib/libspecies.so')

class SpeciesGen(object):
  def __init__(self, _nP):
    
    libSpecies.RandomConfig.argtypes = [ctypes.c_void_p, ctypes.c_uint]
    libSpecies.RandomConfig.restype = ctypes.c_void_p

    libSpecies.CleanSpecies.argtypes = [ctypes.c_void_p]
    libSpecies.CleanSpecies.restype = None
  
    libSpecies.DeleteSpecies.argtypes = [ctypes.c_void_p] 
    libSpecies.DeleteSpecies.restype = None
    
    libSpecies.WriteSpecies.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    libSpecies.WriteSpecies.restype = None

    self.nP = _nP

  def RandomConfig(self, obj):
    return libSpecies.RandomConfig(obj, self.nP)

  def Clean(self, obj):
    libSpecies.CleanSpecies(obj)
  
  def Delete(self, obj):
    libSpecies.DeleteSpecies(obj)
    
  def WriteSpecies(self, obj, fname):
    b_fname = fname.encode('utf-8')
    libSpecies.WriteSpecies(obj, b_fname)    
