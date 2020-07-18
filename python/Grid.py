import ctypes

# Python wrappers for C library Grid routines

libGrid = ctypes.CDLL('../lib/libgrid.so')

class GridGen(object):
  def __init__(self, _Lx, _Ly, _Lz, _hx, _hy, _hz, _Nx, _Ny, _Nz, _dof):

    libGrid.MakeTP.argtypes = [ctypes.c_double, ctypes.c_double, ctypes.c_double,
                               ctypes.c_double, ctypes.c_double, ctypes.c_double,
                               ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint]
    libGrid.MakeTP.restype = ctypes.c_void_p
    
    libGrid.CleanGrid.argtypes = [ctypes.c_void_p]
    libGrid.CleanGrid.restype = None     

    libGrid.DeleteGrid.argtypes = [ctypes.c_void_p]
    libGrid.DeleteGrid.restype = None     
  
    libGrid.WriteGrid.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    libGrid.WriteGrid.restype = None
  
    libGrid.WriteCoords.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    libGrid.WriteCoords.restype = None
  
    libGrid.setGridSpread.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double)] 
    libGrid.setGridSpread.restype = None 

    self.Lx = _Lx
    self.Ly = _Ly
    self.Lz = _Lz
    self.hx = _hx
    self.hy = _hy
    self.hz = _hz
    self.Nx = _Nx
    self.Ny = _Ny
    self.Nz = _Nz
    self.dof = _dof
    self.N = self.Nx * self.Ny * self.Nz
    self.Ntotal = self.N * self.dof

  def MakeTP(self):
    return libGrid.MakeTP(self.Lx, self.Ly, self.Lz, self.hx, self.hy, self.hz,\
                          self.Nx, self.Ny, self.Nz, self.dof) 
  
  def Clean(self,obj):
    libGrid.CleanGrid(obj)
  
  def Delete(self,obj):
    libGrid.DeleteGrid(obj)

  def WriteGrid(self, obj, fname):
    b_fname = fname.encode('utf-8')
    libGrid.WriteGrid(obj, b_fname)

  def WriteCoords(self, obj, fname):
    b_fname = fname.encode('utf-8')
    libGrid.WriteCoords(obj, b_fname)

  def SetGridSpread(self, obj, data):
    libGrid.setGridSpread(obj, data.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
