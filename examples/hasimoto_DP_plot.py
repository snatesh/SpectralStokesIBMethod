import numpy as np
import matplotlib.pyplot as plt

# load output from hasimoto.py
mobx_u = np.loadtxt('x_mobility_unit.txt')
mobx_nu = np.loadtxt('x_mobility_nonUnit_DP.txt')

# set sim params for unit/non_unit grid spacing
eta = 1/4/np.sqrt(np.pi)
F = 1
Rh_u = 1.5539 
Rh_nu = 1.5539 * 0.5
mu0_u = 1 / (6 * np.pi * eta * Rh_u)
mu0_nu = 1/ (6 * np.pi * eta * Rh_nu)
Ls_u = np.linspace(60.,200.,5)
Ls_nu = Ls_u / 2
nTrials = 10

# linear fit to normalized mobility (should have intercept ~(0,1))
#fit_nu = np.poly1d(np.polyfit(1 / Ls_nu, np.mean(mobx_nu / mu0_nu, axis = 1), 1))
fit_nu = np.poly1d(np.polyfit(1 / Ls_nu, mobx_nu / mu0_nu, 1))
# extend 1/L with 0
Ls_nu_ext_inv = np.concatenate((0, 1 / Ls_nu), axis = None)
# plot
fig, ax = plt.subplots(1,1)
ax.plot(1 / Ls_nu, mobx_nu / mu0_nu, 'bs', fillstyle = 'none')
ax.plot(Ls_nu_ext_inv, fit_nu(Ls_nu_ext_inv),'b-', fillstyle = 'none', label = '$h = 0.5$')
ax.set_xlabel('$1 / L$', fontsize = 15)
ax.set_ylabel('$6\pi\eta R_h V$', fontsize = 15)
#ax.set_ylim([0.9,1.002])
ax.set_xlim([0,max(1 / Ls_nu)])
ax.legend(fontsize = 15)
ax.set_title('Checking that normalized mobility goes to 1 for $L = \infty$ and agreement with Hasimoto\'s correction', fontsize = 20) 
plt.show()
