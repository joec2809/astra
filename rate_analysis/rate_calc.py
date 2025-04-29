import sys
sys.path.append('../')

import numpy as np
import matplotlib.pyplot as plt
import astra_functions as af
import rate_functions as rf

from fitter import Fitter

configs = af.read_config_file('../user.config')
main_filepath = configs[0]
vis_time_file = configs[5]
cosmic_dens_file = configs[7]

no_objects = 1

poisson_intervals = rf.poisson_interval(no_objects, 0.32)
no_stat_err = np.array([no_objects-poisson_intervals[0], poisson_intervals[1]-no_objects])

vis_times = np.loadtxt(f'{main_filepath/vis_time_file}')
cosmic_densities = np.loadtxt(f'{main_filepath}/{cosmic_dens_file}')

# Use this block to find best fit visibility time and cosmic density distributions
# f=Fitter(vis_times)
# f=Fitter(cosmic_densities)
# f.fit()
# f.summary()
# plt.show()
# sys.exit()

# Use this line to import the necessary distributions
# from scipy.stats import "names of distributions"
# vis_time_dist = visibility time distribution
# vis_time_dist_name = "name of visibility time distribution"
# cosmic_distribution = cosmic density distribution
# cosmic_distribution_name = "name of cosmic density distribution"

vis_time_peak, vis_time_stat_err = rf.dist_fitter(vis_times, vis_time_dist, vis_time_dist_name)

gal_rate = no_objects/vis_time_peak
gal_stat_err = gal_rate*np.sqrt((no_stat_err/no_objects)**2+(vis_time_stat_err/vis_time_peak)**2)

np.savetxt(f'{main_filepath}/gal_rate.csv', np.array([[gal_rate, 0], gal_stat_err]))
print(gal_rate, gal_stat_err)
print((gal_stat_err)/gal_rate)

mass_rate = no_objects/vis_time_peak
mass_stat_err = mass_rate*np.sqrt((no_stat_err/no_objects)**2+(vis_time_stat_err/vis_time_peak)**2)

np.savetxt(f'{main_filepath}/mass_rate.csv', np.array([[mass_rate,0], mass_stat_err]))
print(mass_rate, mass_stat_err, )
print((mass_stat_err)/mass_rate)

cosmic_density_peak, cosmic_density_stat_err = rf.dist_fitter(cosmic_densities, cosmic_distribution, cosmic_distribution_name)
vol_rate = mass_rate*cosmic_density_peak
vol_stat_err = vol_rate*np.sqrt((mass_stat_err/mass_rate)**2+(cosmic_density_stat_err/cosmic_density_peak)**2)

np.savetxt(f'{main_filepath}/vol_rate.csv', np.array([[vol_rate,0], vol_stat_err]))
print(vol_rate, vol_stat_err)
print((vol_stat_err)/vol_rate)