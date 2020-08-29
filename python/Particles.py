import ctypes
libParticles = ctypes.CDLL('../lib/libparticles.so')

class ParticlesGen(object):
  """
  Python wrappers for C library ParticlesList routines.
  
  This class can be thought of as a generator and
  manager of the underlying C++ ParticlesList struct.
  
  See "extern" in ParticlesList.h.
  
  Attributes:
    nP (int) - number of particles, must be specified.
    dof (int) - degrees of freedom, must be specified.
    xP (doubles or None) - particle positions.
    fP (doubles or None) - forces on particles.
    radP (doubles or None) - radii of particles.
    wfP (unsigned shorts or None) - width of kernel for each particle.
    cwfP (doubles or None) - dimensionless radii of particles.
    betafP (doubles or None) - ES kernel beta parameter for each particle.
    particles (ptr to C++ struct) - a pointer to the generated C++ ParticlesList struct
    
    If any of the inputs that can be None are None, the assumption is that
    they will be populated by a call to the C library, such as RandomConfig(grid)
  """
  def __init__(self, _nP, _dof, \
              _xP = None, _fP = None, _radP = None, _wfP = None, _cwfP = None, _betafP = None):
    """ 
    The constructor for the ParticlesGen class.
    
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
      C++ ParticlesList library are declared. Any functions added
      to the "extern" definition in ParticlesList.h should be
      declared here.
    """ 
    libParticles.MakeParticles.argtypes = [ctypes.POINTER(ctypes.c_double), \
                                       ctypes.POINTER(ctypes.c_double), \
                                       ctypes.POINTER(ctypes.c_double), \
                                       ctypes.POINTER(ctypes.c_double), \
                                       ctypes.POINTER(ctypes.c_double), \
                                       ctypes.POINTER(ctypes.c_ushort), \
                                       ctypes.c_uint, ctypes.c_uint]
    libParticles.MakeParticles.restype = ctypes.c_void_p 
    
    libParticles.Setup.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    libParticles.Setup.restype = None  
    
    libParticles.RandomConfig.argtypes = [ctypes.c_void_p, ctypes.c_uint]
    libParticles.RandomConfig.restype = ctypes.c_void_p

    libParticles.CleanParticles.argtypes = [ctypes.c_void_p]
    libParticles.CleanParticles.restype = None
  
    libParticles.DeleteParticles.argtypes = [ctypes.c_void_p] 
    libParticles.DeleteParticles.restype = None
    
    libParticles.WriteParticles.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    libParticles.WriteParticles.restype = None

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
    # pointer to c++ ParticlesList struct
    self.particles = None
  

  def Make(self):
    """
    Python wrapper for the MakeParticles(...) C lib routine.

    This instantiates a ParticlesList object and stores a pointer to it.

    Parameters: None
    Side Effects:
      self.particles is assigned the pointer to the C++ ParticlesList instance
    """
    self.particles = libParticles.MakeParticles(self.xP.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), \
                                          self.fP.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), \
                                          self.radP.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), \
                                          self.betafP.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), \
                                          self.cwfP.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), \
                                          self.wfP.ctypes.data_as(ctypes.POINTER(ctypes.c_ushort)), \
                                          self.nP, self.dof)


  def Setup(self, grid):
    """
    The python wrapper for the Setup(particles,grid) C lib routine
    This function computes internal data structures, like
    the particles-grid locator, effective kernel widths, etc.

    Parameters:
      grid - a pointer to a valid C++ Grid instance (stored in GridGen)
    Side Effects:
      The data pointed to by self.particles is modified and extended with
      additional information given the grid.
      The data pointed to by grid is modified and extended with
      additional information given the particles.
    """
    libParticles.Setup(self.particles, grid)

  def WriteParticles(self, fname):
    """
    Python wrapper for the WriteParticles(particles,fname) C lib routine
    This writes the current state of the ParticlesList to file  
 
    Parameters: 
      fname (string) - desired name of file
    Side Effects: None, besides file creation and write 
    """
    b_fname = fname.encode('utf-8')
    libParticles.WriteParticles(self.particles, b_fname)    
  
  def Clean(self):
    """
    Python wrapper for the CleanParticles(..) C lib routine.
    This cleans the ParticlesList struct returned by Make(),
    frees any memory internally allocated, and deletes the
    pointer to the ParticlesList struct stored in the class.

    Parameters: None
    Side Effects:
      self.particles is deleted (along with underlying data) and nullified
    """
    libParticles.CleanParticles(self.particles)
    libParticles.DeleteParticles(self.particles)
