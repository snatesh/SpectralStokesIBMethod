import ctypes
libSpecies = ctypes.CDLL('../lib/libspecies.so')

class SpeciesGen(object):
  """
  Python wrappers for C library SpeciesList routines.
  
  This class can be thought of as a generator and
  manager of the underlying C++ SpeciesList struct.
  
  See "extern" in SpeciesList.h.
  
  Attributes:
    nP (int) - number of particles, must be specified.
    dof (int) - degrees of freedom, must be specified.
    xP (doubles or None) - particle positions.
    fP (doubles or None) - forces on particles.
    radP (doubles or None) - radii of particles.
    wfP (unsigned shorts or None) - width of kernel for each particle.
    cwfP (doubles or None) - dimensionless radii of particles.
    betafP (doubles or None) - ES kernel beta parameter for each particle.
    species (ptr to C++ struct) - a pointer to the generated C++ SpeciesList struct
    
    If any of the inputs that can be None are None, the assumption is that
    they will be populated by a call to the C library, such as RandomConfig(grid)
  """
  def __init__(self, _nP, _dof, \
              _xP = None, _fP = None, _radP = None, _wfP = None, _cwfP = None, _betafP = None):
    """ 
    The constructor for the SpeciesGen class.
    
    Parameters:
      nP (int) - number of particles, must be specified.
      dof (int) - degrees of freedom, must be specified.
      xP (doubles or None) - particle positions.
      fP (doubles or None) - forces on particles.
      radP (doubles or None) - radii of particles.
      wfP (unsigned shorts or None) - width of kernel for each particle.
      cwfP (doubles or None) - dimensionless radii of particles.
      betafP (doubles or None) - ES kernel beta parameter for each particle.

    Side Effects:
      The prototypes for relevant functions from the 
      C++ SpeciesList library are declared. Any functions added
      to the "extern" definition in SpeciesList.h should be
      declared here.
    """ 
    libSpecies.MakeSpecies.argtypes = [ctypes.POINTER(ctypes.c_double), \
                                       ctypes.POINTER(ctypes.c_double), \
                                       ctypes.POINTER(ctypes.c_double), \
                                       ctypes.POINTER(ctypes.c_double), \
                                       ctypes.POINTER(ctypes.c_double), \
                                       ctypes.POINTER(ctypes.c_ushort), \
                                       ctypes.c_uint, ctypes.c_uint]
    libSpecies.MakeSpecies.restype = ctypes.c_void_p 
    
    libSpecies.Setup.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    libSpecies.Setup.restype = None  
    
    libSpecies.RandomConfig.argtypes = [ctypes.c_void_p, ctypes.c_uint]
    libSpecies.RandomConfig.restype = ctypes.c_void_p

    libSpecies.CleanSpecies.argtypes = [ctypes.c_void_p]
    libSpecies.CleanSpecies.restype = None
  
    libSpecies.DeleteSpecies.argtypes = [ctypes.c_void_p] 
    libSpecies.DeleteSpecies.restype = None
    
    libSpecies.WriteSpecies.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    libSpecies.WriteSpecies.restype = None

    # number of particles
    self.nP = _nP
    # deg of freedom
    self.dof = _dof
    # particle positions
    self.xP = _xP
    # particle forces
    self.fP = _fP
    # beta for ES kernel for each particle (from table)
    self.betafP = _betafP
    # dimensionless radii given ES kernel for each particle (from table)
    self.cwfP = _cwfP
    # width of ES kernel given dimensionless radii (from table)
    self.wfP = _wfP
    # actual radii of the particles
    self.radP = _radP
    # pointer to c++ SpeciesList struct
    self.species = None
  

  def Make(self):
    """
    Python wrapper for the MakeSpecies(...) C lib routine.

    This instantiates a SpeciesList object and stores a pointer to it.

    Parameters: None
    Side Effects:
      self.species is assigned the pointer to the C++ SpeciesList instance
    """
    self.species = libSpecies.MakeSpecies(self.xP.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), \
                                          self.fP.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), \
                                          self.radP.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), \
                                          self.betafP.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), \
                                          self.cwfP.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), \
                                          self.wfP.ctypes.data_as(ctypes.POINTER(ctypes.c_ushort)), \
                                          self.nP, self.dof)


  def Setup(self, grid):
    """
    The python wrapper for the Setup(species,grid) C lib routine
    This function computes internal data structures, like
    the species-grid locator, effective kernel widths, etc.

    Parameters:
      grid - a pointer to a valid C++ Grid instance (stored in GridGen)
    Side Effects:
      The data pointed to by self.species is modified and extended with
      additional information given the grid.
      The data pointed to by grid is modified and extended with
      additional information given the species.
    """
    libSpecies.Setup(self.species, grid)

  def WriteSpecies(self, fname):
    """
    Python wrapper for the WriteSpecies(species,fname) C lib routine
    This writes the current state of the SpeciesList to file  
 
    Parameters: 
      fname (string) - desired name of file
    Side Effects: None, besides file creation and write 
    """
    b_fname = fname.encode('utf-8')
    libSpecies.WriteSpecies(self.species, b_fname)    
  
  def Clean(self):
    """
    Python wrapper for the CleanSpecies(..) C lib routine.
    This cleans the SpeciesList struct returned by Make(),
    frees any memory internally allocated, and deletes the
    pointer to the SpeciesList struct stored in the class.

    Parameters: None
    Side Effects:
      self.species is deleted (along with underlying data) and nullified
    """
    libSpecies.CleanSpecies(self.species)
    libSpecies.DeleteSpecies(self.species)
