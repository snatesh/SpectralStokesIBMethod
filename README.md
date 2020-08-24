# Spectral_Stokes_IBMethod
A spectral immersed boundary method for particle suspensions in Stokes flow on doubly and triply periodic domains.

### Build Dependencies ###
You will need to `apt install` at least the following dependencies:

* build-essential
* cmake
* libomp-dev
* gcc 7.5.0 or later (eg. module load gcc-9.2 on cims machines)
* FFTW
* LAPACK and the C interface LAPACKE 
    (https://www.assistedcoding.eu/2017/11/04/how-to-install-lapacke-ubuntu/)
* Python 3+ with NumPy/SciPy (latter only used for diags)

### Build Instructions ###
Now, execute the following from the top of the source tree: 
```
$ mkdir build && cd build
$ cmake ..
$ make -j6 (or however many threads you'd like to use)
$ make install
```
Executing the commands above will build all libraries and executables. The libraries are
installed in `$INSTALL_PATH/lib`. Executables are installed in `$INSTALL_PATH/bin`. 
By default, `$INSTALL_PATH` is the top of the source tree.

### Organization ###
The C++ library files are in `src` and `include`. 

The Python wrappers to the exposed C library, as well as currently available solvers are in `python`.

Examples using the python wrappers and solvers can be found in `examples`.

The `Doc` folder contains a presentation on the spreading/interpolation algorithm (`spread_interp.pdf`),
and also a description of the triply periodic solver and Python interface (`stokes_solver_doc.pdf`).
