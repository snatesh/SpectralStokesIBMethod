import numpy as np
import matplotlib.pyplot as plt

mobx = np.loadtxt('x_mobility.txt')
eta = 1/4/np.sqrt(np.pi)
F = 1
Rh = 1.5539 
mu0 = 1 / (6 * np.pi * eta * Rh)
Ls = np.linspace(60.,200.,5)
nTrials = 50

FVratio = F / (eta * mobx)
pbc_corr = np.reshape(np.repeat(6. * np.pi * Rh / (1 - 2.8373 * (Rh / Ls) + \
                              4.19 * np.power(Rh / Ls, 3) - \
                              27.4 * np.power(Rh / Ls, 6)), nTrials), (Ls.size, nTrials))

rel_diff = np.abs(FVratio - pbc_corr)/np.abs(pbc_corr)
fig, ax = plt.subplots(2,1)
ax[0].plot(1 / Ls, mobx / mu0, 'rs', fillstyle = 'none')
ax[1].plot(Ls, rel_diff, 'bs', fillstyle = 'none')
plt.show()
