import sys

import numpy as np
import pandas as pd

from scipy.integrate import trapz
from scipy.stats import norm

def loglinlaw(x, a, b):
	return a*np.log10(x) + b

def schechter(M, M_star, a, phi):
    return (phi/M_star)*np.exp(-M/M_star)*(M/M_star)**a

def double_schechter(M, M_star, a1, phi1, a2, phi2):
    return schechter(M, M_star, a1, phi1) + schechter(M, M_star, a2, phi2)

mass_dist_integrals = np.loadtxt('./results/mass_dist_integrals.csv', delimiter=' ')

mass_start, mass_end = 7, 12
masses = np.logspace(mass_start, mass_end, int((mass_end-mass_start)/0.25)+1)

x_masses = np.logspace(mass_start, mass_end, 10000)

mass_trends = pd.read_csv('../../lowz/rates/results/random_mass_trends.csv')

baldry_params = (10**10.66, -0.35, 3.96*10**-3, -1.47, 0.79*10**-3)
baldry_errs = (10**10.66*(1-10**-0.05), 0.18, 0.34*10**-3, 0.05, 0.23*10**-3)

weaver_params = (10**10.89, -1.42, 0.73*10**-3, -0.46, 1.09*10**-3)
weaver_errs = (10**10.89*(1-10**-0.14), 0.06, 0.26*10**-3, 0.48, 0.52*10**-3)

cosmic_densities = []

rate_mass_trend = True
gsmf_name = 'baldry'

if gsmf_name=='baldry':
    gsmf_params = baldry_params
    gsmf_errs = baldry_errs
elif gsmf_name=='weaver':
    gsmf_params = weaver_params
    gsmf_errs = weaver_errs

for j in range(500):
    random_params = []
    for k, param in enumerate(gsmf_params):
        random_params.append(norm.rvs(loc=param, scale=gsmf_errs[k]))

    mass_function = double_schechter(masses, *random_params)

    bin_centres = []
    dn_dv = []

    for i in range(len(masses)-1):
        bin_centres.append(10**((np.log10(masses[i])+np.log10(masses[i+1]))/2))
        dn_dv.append(trapz([mass_function[i], mass_function[i+1]], [masses[i], masses[i+1]]))
    bin_centres = np.array(bin_centres)
    dn_dv = np.array(dn_dv)

    popt = mass_trends.iloc[j]
    fit_rates = 10**loglinlaw(x_masses, popt[0], popt[1])
    gsmf = np.interp(x_masses, bin_centres, dn_dv)
    numerator = trapz(gsmf*fit_rates*x_masses, x_masses)
    cosmic_densities.append(numerator/mass_dist_integrals[j])

np.savetxt(f'./cosmic_densities/cosmic_densities.csv', np.array(cosmic_densities))