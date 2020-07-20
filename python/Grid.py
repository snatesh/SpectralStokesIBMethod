import ctypes

# Python wrappers for C library Grid routines
# This class can be thought of as a generator and
# manager of the underlying C++ Grid struct.

# See "extern" in Grid.h

libGrid = ctypes.CDLL('../lib/libgrid.so')

class GridGen(object):
  def __init__(self, _Lx, _Ly, _Lz, _hx, _hy, _hz, _Nx, _Ny, _Nz, _dof, _periodicity):

    # define the prototypes of the external C interface 
    # Any functions added to the "extern" definition in
    # Grid.h should be defined here
    libGrid.MakeGrid.argtypes = [ctypes.c_double, ctypes.c_double, ctypes.c_double,
                                 ctypes.c_double, ctypes.c_double, ctypes.c_double,
                                 ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, 
                                 ctypes.c_uint, ctypes.c_uint]
    libGrid.MakeGrid.restype = ctypes.c_void_p
    
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

    # length in x,y,z
    self.Lx = _Lx
    self.Ly = _Ly
    self.Lz = _Lz
    # grid spacing in x,y,z
    self.hx = _hx
    self.hy = _hy
    self.hz = _hz
    # number of points in x,y,z
    self.Nx = _Nx
    self.Ny = _Ny
    self.Nz = _Nz
    # degrees of freedom of data on the grid
    self.dof = _dof
    # periodicity of the grid ( 1 <= periodicity <= 3)
    self.periodicity = _periodicity
    # getting total nums
    self.N = self.Nx * self.Ny * self.Nz
    self.Ntotal = self.N * self.dof

  # The python wrapper for the MakeGrid() C lib routine
  # This function instantiates a Grid struct and returns its pointer
  def Make(self):
    return libGrid.MakeGrid(self.Lx, self.Ly, self.Lz, self.hx, self.hy, self.hz,\
                            self.Nx, self.Ny, self.Nz, self.dof, self.periodicity) 
 
  # The python wrapper for the CleanGrid(grid) C lib routine 
  # This cleans the Grid struct returned by Make()
  # and frees memory internally allocated by the Grid struct
  def Clean(self,grid):
    libGrid.CleanGrid(grid)
 
  # The python wrapper for the DeleteGrid(grid) C lib routine 
  # This deletes the pointer returned by Make()
  def Delete(self,grid):
    libGrid.DeleteGrid(grid)

  # Python wrapper for the WriteGrid(grid,fname) C lib routine
  # write the current data of the Grid struct to file
  def WriteGrid(self, grid, fname):
    b_fname = fname.encode('utf-8')
    libGrid.WriteGrid(grid, b_fname)

  # Python wrapper for the WriteCoords(grid,fname) C lib routine
  # write the grid coordinates to file
  def WriteCoords(self, grid, fname):
    b_fname = fname.encode('utf-8')
    libGrid.WriteCoords(grid, b_fname)

  # Python wrapper for the setGridSpread(grid) C lib routine
  # This sets new data on the Grid by overwriting
  # the data member grid.fG with new_data
  def SetGridSpread(self, grid, new_data):
    libGrid.setGridSpread(grid, new_data.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
