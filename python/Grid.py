import ctypes
libGrid = ctypes.CDLL('../lib/libgrid.so')

class GridGen(object):
  """
  Python wrappers for C library Grid routines.
  
  This class can be thought of as a generator and
  manager of the underlying C++ Grid struct.
  
  See "extern" in Grid.h.
  
  Attributes:
    Lx, Ly, Lz (double) - length in x,y and z
    hx, hy, hz (double) - grid spacing in x, y and z
    Nx, Ny, Nz (int) - number of points in x,y and z
    N (int) = Nx * Ny * Nz
    dof (int) - degrees of freedom
    Ntotal (int) = N * dof
    periodicity (int) - of the domain (can be 2 or 3 for now) 
    grid (ptr to C++ struct) - a pointer to the generated C++ Grid struct  
  """
  def __init__(self, _Lx, _Ly, _Lz, _hx, _hy, _hz, _Nx, _Ny, _Nz, _dof, _periodicity):
    """ 
    The constructor for the GridGen class.
    
    Parameters:
      Lx, Ly, Lz (double) - length in x,y and z
      hx, hy, hz (double) - grid spacing in x, y and z
      Nx, Ny, Nz (int) - number of points in x,y and z
      N (int) = Nx * Ny * Nz
      dof (int) - degrees of freedom
      Ntotal (int) = N * dof
      periodicity (int) - of the domain (can be 2 or 3 for now) 

    Side Effects:
      The prototypes for relevant functions from the 
      C++ Grid library are declared. Any functions added
      to the "extern" definition in Grid.h should be
      declared here.
    """ 
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
    # pointer to C++ Grid struct
    self.grid = None

  # The python wrapper for the MakeGrid() C lib routine
  # This function instantiates a Grid struct and returns its pointer
  def Make(self):
    """
    Python wrapper for the MakeGrid(...) C lib routine.

    This instantiates a Grid object and stores a pointer to it.

    Parameters: None
    Side Effects:
      self.grid is assigned the pointer to the C++ Grid instance
    """
    self.grid = libGrid.MakeGrid(self.Lx, self.Ly, self.Lz, self.hx, self.hy, self.hz,\
                                 self.Nx, self.Ny, self.Nz, self.dof, self.periodicity) 
  

  def SetGridSpread(self, new_data):
    """
    Python wrapper for the setGridSpread(grid) C lib routine
    This sets new data on the Grid by overwriting
    the data member grid.fG with new_data
    
    Parameters: 
      new_data (doubles) - new data to set on the grid with total size Nx * Ny * Nz * dof
    Side Effects:
      self.grid.fG (in C) is overwritten with new_data, so the data pointed to by grid is changed
    """
    libGrid.setGridSpread(self.grid, new_data.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))

  def WriteGrid(self, fname):
    """
    Python wrapper for the WriteGrid(grid,fname) C lib routine
    This writes the current data of the Grid to file  
 
    Parameters: 
      fname (string) - desired name of file
    Side Effects: None, besides file creation and write 
    """
    b_fname = fname.encode('utf-8')
    libGrid.WriteGrid(self.grid, b_fname)

  def WriteCoords(self, fname):
    """
    Python wrapper for the WriteCoords(grid,fname) C lib routine
    This writes the coordinates of the Grid to file  
 
    Parameters: 
      fname (string) - desired name of file
    Side Effects: None, besides file creation and write 
    """
    b_fname = fname.encode('utf-8')
    libGrid.WriteCoords(self.grid, b_fname)

  def Clean(self):
    """
    Python wrapper for the CleanGrid(..) C lib routine.
    This cleans the Grid struct returned by Make(),
    frees any memory internally allocated, and deletes the
    pointer to the Grid struct stored in the class.

    Parameters: None
    Side Effects:
      self.grid is deleted (along with underlying data) and nullified
    """
    libGrid.CleanGrid(self.grid)
    libGrid.DeleteGrid(self.grid)
