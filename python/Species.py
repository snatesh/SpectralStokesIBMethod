import ctypes

# Python wrappers for C library SpeciesList routines
# This class can be thought of as a generator and
# manager of the underlying C++ SpeciesList struct

# See "extern" in SpeciesList.h

libSpecies = ctypes.CDLL('../lib/libspecies.so')

class SpeciesGen(object):
  def __init__(self, _nP, _dof, \
              _xP = None, _fP = None, _radP = None, _wfP = None, _cwfP = None, _betafP = None):
    
    # define the prototypes of the external C interface 
    # Any functions added to the "extern" definition in
    # SpeciesList.h should be defined here
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

  # The python wrapper for the MakeSpecies() C lib routine
  # This instantiates a SpeciesList struct and returns its pointer
  def Make(self):
    return libSpecies.MakeSpecies(self.xP.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), \
                                  self.fP.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), \
                                  self.radP.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), \
                                  self.betafP.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), \
                                  self.cwfP.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), \
                                  self.wfP.ctypes.data_as(ctypes.POINTER(ctypes.c_ushort)), \
                                  self.nP, self.dof)

  # The python wrapper for the Setup(species,grid) C lib routine
  # This function computes internal data structures, like
  # the species-grid locator, effective kernel widths, etc.
  def Setup(self,species,grid):
    libSpecies.Setup(species, grid)

  # The python wrapper for the RandomConfig(grid,nP) C lib routine
  # This function configures nP particles randomly on the grid
  # in terms of w = [4,5,6] and the corresponding beta/Rh values
  # Note, use Make() and then Setup(species,grid) instead of this
  # if you want to define particle configurations in python
  def RandomConfig(self, grid):
    return libSpecies.RandomConfig(grid, self.nP)

  # Python wrapper for the CleanSpecies(species) C lib routine 
  # This cleans the SpeciesList struct returned by Make (or RandomConfig)
  # and frees memory internally allocated by the SpeciesList struct
  def Clean(self, obj):
    libSpecies.CleanSpecies(obj)
  
  # Python wrapper for the DeleteSpecies(species) C lib routine
  # This deletes the pointer returned by Make (or RandomConfig)
  def Delete(self, obj):
    libSpecies.DeleteSpecies(obj)
   
  # Python wrapper for the WriteSpecies(species,fname) C lib routine
  # This writes the current state of the SpeciesList to file  
  def WriteSpecies(self, obj, fname):
    b_fname = fname.encode('utf-8')
    libSpecies.WriteSpecies(obj, b_fname)    
