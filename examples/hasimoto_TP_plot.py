import numpy as np
import matplotlib.pyplot as plt

# load output from hasimoto.py
mobx_u = np.loadtxt('x_mobility_unit.txt')
mobx_nu = np.loadtxt('x_mobility_nonUnit.txt')

# set sim params for unit/non_unit grid spacing
eta = 1/4/np.sqrt(np.pi)
F = 1
Rh_u = 1.5539 
Rh_nu = 1.5539 * 0.5
mu0_u = 1 / (6 * np.pi * eta * Rh_u)
mu0_nu = 1/ (6 * np.pi * eta * Rh_nu)
Ls_u = np.linspace(60.,200.,5)
Ls_nu = Ls_u / 2
nTrials = 50

# compute F/(eta*V)
FVratio_u = F / (eta * mobx_u)
FVratio_nu = F / (eta * mobx_nu)
# compute pbc correction
pbc_corr_u = np.reshape(np.repeat(6. * np.pi * Rh_u / (1 - 2.8373 * (Rh_u / Ls_u) + \
                              4.19 * np.power(Rh_u / Ls_u, 3) - \
                              27.4 * np.power(Rh_u / Ls_u, 6)), nTrials), (Ls_u.size, nTrials))

pbc_corr_nu = np.reshape(np.repeat(6. * np.pi * Rh_nu / (1 - 2.8373 * (Rh_nu / Ls_nu) + \
                              4.19 * np.power(Rh_nu / Ls_nu, 3) - \
                              27.4 * np.power(Rh_nu / Ls_nu, 6)), nTrials), (Ls_nu.size, nTrials))
# relative difference in F/(eta*V) and pbc correction
rel_diff_u = np.abs(FVratio_u - pbc_corr_u)/np.abs(pbc_corr_u)
rel_diff_nu = np.abs(FVratio_nu - pbc_corr_nu)/np.abs(pbc_corr_nu)
# linear fit to normalized mobility (should have intercept ~(0,1))
fit_u = np.poly1d(np.polyfit(1 / Ls_u,  np.mean(mobx_u / mu0_u, axis = 1), 1))
fit_nu = np.poly1d(np.polyfit(1 / Ls_nu, np.mean(mobx_nu / mu0_nu, axis = 1), 1))
# extend 1/L with 0
Ls_u_ext_inv = np.concatenate((0, 1 / Ls_u), axis = None)
Ls_nu_ext_inv = np.concatenate((0, 1 / Ls_nu), axis = None)
# plot
fig, ax = plt.subplots(2,1)
ax[0].plot(1 / Ls_u, mobx_u / mu0_u, 'rs', fillstyle = 'none')
ax[0].plot(1 / Ls_nu, mobx_nu / mu0_nu, 'bs', fillstyle = 'none')
ax[0].plot(Ls_u_ext_inv, fit_u(Ls_u_ext_inv),'rs-', fillstyle = 'none', label = '$h = 1$')
ax[0].plot(Ls_nu_ext_inv, fit_nu(Ls_nu_ext_inv),'bs-', fillstyle = 'none', label = '$h = 0.5$')
ax[0].set_xlabel('$1 / L$', fontsize = 15)
ax[0].set_ylabel('$6\pi\eta R_h V$', fontsize = 15)
ax[0].set_ylim([0.9,1.002])
ax[0].legend(fontsize = 15)
ax[0].set_title('Checking that normalized mobility goes to 1 for $L = \infty$ and agreement with Hasimoto\'s correction', fontsize = 20) 

ax[1].plot(Ls_u, rel_diff_u, 'rs', fillstyle = 'none', label = '$h = 1$')
ax[1].plot(Ls_nu, rel_diff_nu, 'bs', fillstyle = 'none', label = '$h = 0.5$')
ax[1].set_xlabel('$L$', fontsize = 15)
ax[1].set_ylabel('Relative error in $\\frac{F}{\eta V}$ w.r.t. Hasimoto correction', fontsize = 15)
ax[0].tick_params(axis='both', which='major', labelsize=15)
ax[1].tick_params(axis='both', which='major', labelsize=15)
plt.show()
