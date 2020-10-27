import ctypes
import numpy as np
libGrid = ctypes.CDLL('../lib/libgrid.so')

class GridGen(object):
  """
  Python wrappers for C library Grid routines.
  
  This class can be thought of as a generator and
  manager of the underlying C++ Grid struct.
  
  See GridWrapper.cpp..
  
  Attributes:
    Lx, Ly, Lz (double) - length in x,y and z
    hx, hy, hz (double) - grid spacing in x, y and z
                        - if zpts/zwts are provided, hz is ignored
    Nx, Ny, Nz (int) - number of points in x,y and z
    N (int) = Nx * Ny * Nz
    dof (int) - degrees of freedom
    periodic(x,y,z) (bool) - periodicity of eaxh axis
    Ntotal (int) = N * dof
    BCs - Boundary conditions for each variable on grid, at end of each axis (dof x 6)
    grid (ptr to C++ struct) - a pointer to the generated C++ Grid struct  
  """
  def __init__(self, _Lx, _Ly, _Lz, _hx, _hy, _hz, _Nx, _Ny, _Nz, _dof, _periodic_x,
               _periodic_y, _periodic_z, _BCs, _zpts = None, _zwts = None):
    """ 
    The constructor for the GridGen class.
    
    Parameters:
      Lx, Ly, Lz (double) - length in x,y and z
      hx, hy, hz (double) - grid spacing in x, y and z
      Nx, Ny, Nz (int) - number of points in x,y and z
      N (int) = Nx * Ny * Nz
      dof (int) - degrees of freedom
      periodic(x,y,z) (bool) - periodicity of eaxh axis
      zpts, zwts - z grid and associated quadrature weights
      BCs - Boundary conditions for each variable on grid, at end of each axis (dof x 6)
      Ntotal (int) = N * dof

    Side Effects:
      The prototypes for relevant functions from the 
      C++ Grid library are declared. Any functions added
      to the "extern" definition in Grid.h should be
      declared here.
    """ 
    libGrid.MakeGrid.argtypes = None
    libGrid.MakeGrid.restype = ctypes.c_void_p
    
    libGrid.SetL.argtypes = [ctypes.c_void_p, ctypes.c_double, 
                             ctypes.c_double, ctypes.c_double]
    libGrid.SetL.restype = None
    
    libGrid.SetN.argtypes = [ctypes.c_void_p, ctypes.c_uint, 
                             ctypes.c_uint, ctypes.c_uint]
    libGrid.SetN.restype = None
    
    libGrid.Seth.argtypes = [ctypes.c_void_p, ctypes.c_double, 
                             ctypes.c_double, ctypes.c_double]
    libGrid.Seth.restype = None
    
    libGrid.SetZ.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double),
                             ctypes.POINTER(ctypes.c_double)]
    libGrid.SetZ.restype = None
    
    libGrid.SetPeriodicity.argtypes = [ctypes.c_void_p, ctypes.c_bool,
                                       ctypes.c_bool, ctypes.c_bool]
    libGrid.SetPeriodicity.restype = None
    
    libGrid.SetBCs.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint)]
    libGrid.SetBCs.restype = None
  
    libGrid.Setdof.argtypes = [ctypes.c_void_p, ctypes.c_uint]
    libGrid.Setdof.restype = None

    libGrid.ZeroExtGrid.argtypes = [ctypes.c_void_p]
    libGrid.ZeroExtGrid.restype = None 
 
    libGrid.SetupGrid.argtypes = [ctypes.c_void_p]
    libGrid.SetupGrid.restype = None
    
    libGrid.CleanGrid.argtypes = [ctypes.c_void_p]
    libGrid.CleanGrid.restype = None     

    libGrid.DeleteGrid.argtypes = [ctypes.c_void_p]
    libGrid.DeleteGrid.restype = None     
  
    libGrid.WriteGrid.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    libGrid.WriteGrid.restype = None
  
    libGrid.WriteCoords.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    libGrid.WriteCoords.restype = None
  
    libGrid.SetSpread.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double)] 
    libGrid.SetSpread.restype = None 

    libGrid.GetSpread.argtypes = [ctypes.c_void_p] 
    libGrid.GetSpread.restype = ctypes.POINTER(ctypes.c_double) 
    
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
    # z grid and weights, if provided
    self.zpts = _zpts
    self.zwts = _zwts
    # getting total nums
    self.N = self.Nx * self.Ny * self.Nz
    self.Ntotal = self.N * self.dof
    # periodicity of each axis
    self.periodic_x = _periodic_x
    self.periodic_y = _periodic_y
    self.periodic_z = _periodic_z
    # boundary conditions
    self.BCs = _BCs 
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
      self.grid is assigned the pointer to the C++ Grid instance, 
      and the struct pointed to is initialized with attributes of self
    """
    self.grid = libGrid.MakeGrid();
    libGrid.SetL(self.grid, self.Lx, self.Ly, self.Lz)
    libGrid.SetN(self.grid, self.Nx, self.Ny, self.Nz)  
    if self.zpts is None:
      libGrid.Seth(self.grid, self.hx, self.hy, self.hz)
    else:
      libGrid.Seth(self.grid, self.hx, self.hy, 0.0)
      libGrid.SetZ(self.grid, self.zpts.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), \
                   self.zwts.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
    libGrid.SetPeriodicity(self.grid, self.periodic_x, self.periodic_y, self.periodic_z)
    libGrid.Setdof(self.grid, self.dof) 
    libGrid.SetBCs(self.grid, self.BCs.ctypes.data_as(ctypes.POINTER(ctypes.c_uint))) 
    libGrid.SetupGrid(self.grid)  

  def ZeroExtGrid(self):
    """
    Python wrapper for ZeroExtGrid(grid) C lib routine
    This zeros the extended grid, and must be called
    before spreading.

    Parameter : none
    Side Effects: 
      self.grid.fG_unwrap is overwritten with 0s 
    """
    libGrid.ZeroExtGrid(self.grid)

  def SetSpread(self, new_data):
    """
    Python wrapper for the SetSpread(grid) C lib routine
    This sets new data on the Grid by overwriting
    the data member grid.fG with new_data
    
    Parameters: 
      new_data (doubles) - new data to set on the grid with total size Nx * Ny * Nz * dof
    Side Effects:
      self.grid.fG (in C) is overwritten with new_data, so the data pointed to by grid is changed
    """
    libGrid.SetSpread(self.grid, new_data.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
  
  def GetSpread(self):
    """
    Python wrapper for the GetSpread(grid) C lib routine
    This sets gets a pointer to the data spread on the grid
    
    Parameters: None 
    Side Effects: None
    Returns: numpy array containing pointer to spread data on grid
    """
    return np.ctypeslib.as_array(libGrid.GetSpread(self.grid), shape=(self.Ntotal, ))

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
