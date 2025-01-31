import sys
sys.path.append('../')

import astra_functions as af

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from scipy.optimize import curve_fit
from pynverse import inversefunc

configs = af.read_config_file('../user.config')
main_file_path = configs[0]
fakes_data_file = configs[3]

feature_names = ['[FeVII]6088',
                '[FeX]6376',
                '[FeXI]7894',
                '[FeXIV]5304',
                '[FeVII]3759',
                '[FeVII]5160',
                '[FeVII]5722']

largest_eqw = -15

fakes_data = pd.read_csv(f'{main_file_path}/{fakes_data_file}')

feature_eqws = fakes_data[feature_names]

ave_eqws = np.mean(feature_eqws, axis=1) # Average EQW of the features
sort_ind = np.argsort(ave_eqws)
ave_eqws = ave_eqws[sort_ind]

detections = fakes_data['Detection'] # Expect boolean values
detections = detections[sort_ind]

detections = detections[ave_eqws > largest_eqw]
ave_eqws = ave_eqws[ave_eqws > largest_eqw]

# Binning

bin_width = 1
bins = np.arange(largest_eqw, 5+bin_width, bin_width)
eqws_binned = pd.cut(ave_eqws, bins)
cuts = np.cumsum(eqws_binned.value_counts())[:-1]
detections_binned = np.split(detections, cuts)
total_detections = np.sum(detections_binned, axis=1)
bin_sizes = np.sum(eqws_binned.value_counts())[:-1]

det_eff = total_detections/bin_sizes

det_error = np.sqrt(((total_detections+1)*(total_detections+2))/((bin_sizes+2)*(bin_sizes+3))-((total_detections+1)**2)/((bin_sizes+2)**2))

det_err = np.zeros((2, len(det_error)))

for i, error in enumerate(det_error):
    det_err[0][i] = error
    det_err[1][i] = error

    if det_eff[i]-det_err[0][i] < 0:
        det_err[0][i] = det_eff[i]

    if det_eff[i]+det_err[1][i] > 1:
        det_err[1][i] = 1 - det_eff[i]

eqw_err = np.full(len(det_error), np.sqrt(bin_width))/2

bin_centres = np.zeros(len(det_eff))
for i in range(len(bin_centres)):
    bin_centres[i] = (bins[i]+bins[i+1])/2

not_nans = ~np.isnan(det_eff)
fin_det_eff = det_eff[not_nans]
fin_det_err = det_error[not_nans]
fin_bin_centres = bin_centres[not_nans]

# Curves

def sigmoid(x, A, K, B, v, Q):
    return A+((K-A)/((1+Q*np.exp(-B*x))**(1/v)))

xs = np.linspace(largest_eqw, 5, 500)

# Fit curve parameters to data
par, cov = curve_fit(sigmoid, fin_bin_centres, fin_det_eff, sigma=fin_det_err, bounds=([0, 0, -np.inf, -np.inf, -np.inf], [1, 1, np.inf, np.inf, np.inf]), maxfev=5000)

curve = sigmoid(xs, par[0], par[1], par[2], par[3], par[4])

np.savetxt(f'{main_file_path}/data/det_eff_parameters.csv', np.array([par, np.sqrt(np.diag(cov))]))

fitted_func = lambda x: par[0]+((par[1]-par[0])/((1+par[4]*np.exp(-par[2]*x))**(1/par[3])))

inv_func = inversefunc(fitted_func)

half_det_eff = inv_func(0.5)

# Plotting

error_linewidth = 1.5
markersize = 7
cap_thickness = 1.5

fig, ax = plt.subplots(figsize = (10,6))

ax.errorbar(bin_centres, det_eff, xerr = eqw_err, yerr = det_err, fmt = '.k', ls = 'none', markersize = markersize, elinewidth=error_linewidth, capthick=cap_thickness)
ax.plot(xs, curve, 'k')

ax.set(xlim=[largest_eqw, 5], ylim=[0,1.1], xlabel=r'Average equivalent width $\left(\mathrm{\AA}\right)$', ylabel='Detection efficiency')
ax.vlines(half_det_eff, 0, 0.5, ls = '--', color = 'k')
ax.hlines(0.5, largest_eqw, half_det_eff, ls = '--', color = 'k')
ax.text(-12, 0.4, r'$W_{\lambda, 50\%} = $' + f'{np.round(half_det_eff, 2)}'+r'$\mathrm{\AA}$')

plt.savefig(f'{main_file_path}/figures/eqw_det_eff.pdf')