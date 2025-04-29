import numpy as np

from scipy.stats import chi2 
from fitter import Fitter

def poisson_interval(k, alpha=0.05): 
    """
    Use chisquared info to get the poisson interval. Uses scipy.stats 
    """

    a = alpha
    low, high = (chi2.ppf(a/2, 2*k) / 2, chi2.ppf(1-a/2, 2*k + 2) / 2)
    if k == 0: 
        low = 0.0
	
    return low, high

def dist_fitter(vals, distribution, distribution_name):
	"""
	Fit a distribution to a set of values \n
	Must provide the distribution and the name of the distribution
	"""

	f=Fitter(vals, distributions=distribution_name)
	f.fit()
	dist_params = f.get_best()[distribution_name]

	x = np.linspace(np.min(vals), np.max(vals), 10000)
	dist = distribution.pdf(x, **dist_params)
	peak_idx = np.argmax(dist)
	peak = x[peak_idx]
	percentiles = np.percentile(vals, [16, 84])

	# Calculate the statistical error
	# If the negative error goes below zero, set it such that it goes to zero
	if percentiles[0]<peak and percentiles[1]>peak:
		stat_err = np.array([peak-percentiles[0], percentiles[1]-peak])
	elif percentiles[0]>peak:
		stat_err = np.array([peak-np.min(vals), percentiles[1]-peak])
	elif percentiles[1]<peak:
		stat_err = np.array([peak-percentiles[0], np.max(vals)-peak])
	
	return peak, stat_err