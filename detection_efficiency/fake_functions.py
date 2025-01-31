import sys
import spectres

import numpy as np
import scipy.constants as constants

c = constants.value('speed of light in vacuum') / 1000

def feature_area():
	"""
	These four parameters control what regions of the spectra are 'cut out' for analysis around each feature
	"""

	lower_wave_region = 125
	upper_wave_region = 125

	lower_v_shift = -7000
	upper_v_shift = 5000

	return \
		lower_wave_region, upper_wave_region, lower_v_shift, upper_v_shift

def spectral_line_list():
	"""
	Serves as the internal line list in and information storage location.
	The [OII] line is actually a doublet of 3727.320 and 3729.225
	"""

	spectral_lines = {
		'Halpha': 6562.79,
		'Hbeta': 4861.35,
		'Hgamma': 4340.472,
		'Hdelta': 4101.734,

		'[NII]6548': 6548,
		'[NII]6584': 6584,

		'[FeVII]6088': 6088,
		'[FeX]6376': 6376,
		'[FeXI]7894': 7894,
		'[FeXIV]5304': 5304,
		'[FeVII]3759': 3759,
		'[FeVII]5160': 5160,
		'[FeVII]5722': 5722,

		'[OI]6300': 6300,
		'[OI]6363': 6363,
		'[OII]3728': 3728.2725,
		'[OIII]4959': 4959,
		'[OIII]5007': 5007,

		'HeI4478': 4478,
		'HeII4686': 4686,

		'NaID': 5892.935,

		'[SII]6717': 6717,
		'[SII]6731': 6731,

		'[SII]6717_6731': 6724,

		'[SIII]6313': 6313,

		'[NeIII]3896': 3896,

		'[CaXV]5446': 5446,

		'MgI5175': 5175

	}

	return spectral_lines


"""
Fake generation functions
"""

def cut_region(
		shift,
		wave,
		flux,
		low_cut,
		high_cut,
		mode='Shift',
		error=0
):
	"""
	Cuts down a spectrum, flux, wavelength and determined shift, based on either a shift or wavelength region
	Defaults to shift for compatibility
	Expects wave and flux to have a unit but does NOT preserve these
	"""

	shift = shift.value
	wave = wave.value
	flux = flux.value

	cut_shift = []
	cut_wave = []
	cut_flux = []
	cut_error = []

	if mode.upper() in {'SHIFT', 'VELOCITY'}:
		for xx, item in enumerate(shift):

			if low_cut <= shift[xx] <= high_cut:
				cut_shift.append(shift[xx])
				cut_wave.append(wave[xx])
				cut_flux.append(flux[xx])
				if type(error) != int:
					cut_error.append(error[xx])

			elif shift[xx] > high_cut:
				break

	elif mode.upper() in {'WAVELENGTH', 'LAMBDA'}:

		for xx, item in enumerate(wave):

			if low_cut <= wave[xx] <= high_cut:
				cut_shift.append(shift[xx])
				cut_wave.append(wave[xx])
				cut_flux.append(flux[xx])
				if type(error) != int:
					cut_error.append(error[xx])

			elif wave[xx] > high_cut:
				break

	else:
		print('Mode selection not recognised.\nPlease check and try again.')
		sys.exit()

	return cut_shift, cut_wave, cut_flux, cut_error


def make_continuum(
		xaxis,
		flux,
		line_name,
		line_loc,
		mode='SHIFT'
):
	"""
	Estimates the continuum emission over a given region of the spectrum.
	Fits a straight line between the edges of the region given.
	"""
	if mode.upper() == 'WAVELENGTH':

		if line_name.lower() in {'halpha', '[nii]6548', '[nii]6584'}:

			blue_start = 6460
			blue_end = 6470

			red_start = 6620
			red_end = 6635

		elif line_name.lower() in {'hbeta', '[oiii]5007'}:

			blue_start = line_loc - 40
			blue_end = line_loc - 30

			red_start = line_loc + 30
			red_end = line_loc + 40

		elif line_name.lower() in {'[fexiv]5304'}:

			blue_start = line_loc - 70
			blue_end = line_loc - 30

			red_start = line_loc + 30
			red_end = line_loc + 50

		elif line_name.lower() in {'heii4686'}:

			blue_start = line_loc - 45
			blue_end = line_loc - 30

			red_start = line_loc + 30
			red_end = line_loc + 45

		elif line_name.lower() in {'[oiii]4959'}:

			blue_start = line_loc - 25
			blue_end = line_loc - 15

			red_start = line_loc + 15
			red_end = line_loc + 25

		elif line_name.lower() in {'[fevii]3759'}:

			blue_start = line_loc - 25
			blue_end = line_loc - 10

			red_start = line_loc + 15
			red_end = line_loc + 25

		else:
			blue_start = line_loc - 70
			blue_end = line_loc - 30

			red_start = line_loc + 30
			red_end = line_loc + 70

	elif mode.upper() == 'SHIFT':

		if line_name.lower() in {'halpha', '[nii]6548', '[nii]6584'}:

			# Set reference position to be Halpha

			line_loc_ref = 6562.79

			# Calculate offset
			offset = ((line_loc * c) / line_loc_ref) - c

			blue_start = -4000 - offset
			blue_end = -2500 - offset

			red_start = 2500 - offset
			red_end = 4000 - offset

		elif line_name.lower() in {'[sii]6717', '[sii]6731', '[sii]6717_6731'}:

			# Set reference position to be the average of the doublet line positions

			line_loc_ref = 6724

			# Calculate offset
			offset = ((line_loc * c) / line_loc_ref) - c

			blue_start = -4000 - offset
			blue_end = -2500 - offset

			red_start = 2500 - offset
			red_end = 4000 - offset

		elif line_name.lower() in {'hbeta', 'hgamma'}:

			blue_start = -4000
			blue_end = -2500

			red_start = 2500
			red_end = 4000

		elif line_name.lower() in {'[oi]6363'}:

			blue_start = -2500
			blue_end = -1500

			red_start = 1500
			red_end = 3000

		elif line_name.lower() in {'[fevii]3759'}:

			blue_start = -1750
			blue_end = -1000

			red_start = 1500
			red_end = 3000

		elif line_name.lower() in {'[oiii]5007'}:

			blue_start = -2250
			blue_end = -1250

			red_start = 1500
			red_end = 3000

		elif line_name.lower() in {'[oi]6300', '[oiii]4959', '[siii]6313'}:

			blue_start = -3000
			blue_end = -1500

			red_start = 1500
			red_end = 2500

		else:
			blue_start = -3000
			blue_end = -1500

			red_start = 1500
			red_end = 3000

	else:
		print('Mode: {mode} not recognised - please check and try again\nContinua maker failure')
		sys.exit()

	continuum_regions = [blue_start, blue_end, red_start, red_end]

	blue_middle = (blue_start + blue_end) / 2
	red_middle = (red_start + red_end) / 2

	blue_flux = []
	red_flux = []

	for xx, point in enumerate(xaxis):
		if blue_start < xaxis[xx] < blue_end:
			blue_flux.append(flux[xx])
		if red_start < xaxis[xx] < red_end:
			red_flux.append(flux[xx])

	try:
		m = (np.nanmean(red_flux) - np.nanmean(blue_flux)) / (red_middle - blue_middle)
		d = np.nanmean(blue_flux) - (m * blue_middle)

	except (FloatingPointError, RuntimeWarning):
		print(
			"Something has gone wrong with the selection of part of the continuum - likely no valid points"
			"\nSkipping object for now"
		)
		print(f"{line_name}")

		print(xaxis)

		return [], [], []

	continuum = []

	for xx, point in enumerate(xaxis):
		continuum.append((m * xaxis[xx]) + d)

	try:
		scaled_flux = np.array(flux) / np.array(continuum)
	except:
		print("Continuum Scaling Failure")
		print(flux)
		print(continuum)

		scaled_flux = np.array(flux)

	return continuum, scaled_flux, continuum_regions

def isolate_features(wavelengths, flux, line_names, mode='select'):
	"""
	Produces spectrum with only the features of interest (mode='select') or with the features removed (mode='remove')
	"""
	flux_without_peaks = np.zeros(len(flux))
	for i, value in enumerate(flux):
		flux_without_peaks[i] = value
	lines = spectral_line_list()
	for ii, item in enumerate(lines):
		if item in line_names:
			line_location = lines[item][0]
			shift = (((np.array(wavelengths) * c) / line_location) - c)
			shift_region, wave_region, flux_region, error_region = cut_region(
					shift, 
					wavelengths,
					flux,
					line_location - 40,
					line_location + 40,
					mode='wavelength'
				)
			continuum = make_continuum(
					wave_region,
					flux_region,
					item,
					line_location,
					mode='wavelength'
				)
			peak_start = find_nearest(wavelengths, (continuum[2][0]+continuum[2][1])/2)
			try:
				flux_without_peaks[peak_start:peak_start+len(continuum[0])] = continuum[0]
			except ValueError:
				len_to_end = len(flux_without_peaks) - peak_start
				new_continuum = continuum[0][0:len_to_end]
				flux_without_peaks[peak_start:peak_start+len(new_continuum)] = new_continuum

	if mode == 'remove':
		result_flux = flux_without_peaks
	elif mode == 'select':
		flux -= flux_without_peaks
		negatives = np.argwhere(flux < 0)
		flux[negatives] = 0
		result_flux = flux
	
	return wavelengths, result_flux


def spec_resample(new_wavelength, wave_to_resamp, flux_to_resamp, flux_errs_to_resamp):
	"""
	Resample higher resolution spectrum to match resolution of lower resolution spectrum
	"""
	resamp_flux, resamp_flux_errs = spectres.spectres(new_wavelength, wave_to_resamp, flux_to_resamp, spec_errs=flux_errs_to_resamp)

	return resamp_flux, resamp_flux_errs


""" Rounding functions"""

def find_nearest(array, value):
	""" Find the index and value in an array that is closest to a given value """
	array = np.asarray(array)
	idx = (np.abs(array - value)).argmin()

	return array[idx], idx