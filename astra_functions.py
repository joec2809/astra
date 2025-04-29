import os
import configparser

import numpy as np

from astropy.io import fits

def read_config_file(config_name):
	""" Read Config File """
	config = configparser.ConfigParser()
	config.read(config_name)
	main_file_path = config['general']['main_file_path']

	base_data_file = config['fakes']['base_data_file']
	features_data_file = config['fakes']['features_data_file']
	fake_data_file = config['fakes']['fake_data_file']

	gal_sample_file = config['rates']['gal_sample_file']
	vis_times_file = config['rates']['vis_times_file']
	cosmic_dens_file = config['rates']['cosmic_dens_file']

	return main_file_path, base_data_file, features_data_file, fake_data_file, gal_sample_file, vis_times_file, cosmic_dens_file

def create_directory(main_file_path, directory):
	""" Create Directory """
	if not os.path.exists(f'{main_file_path}/{directory}'):
		os.makedirs(f'{main_file_path}/{directory}')
	return None

def open_fits_spectrum(file_path, redshift):
	""" Open FITS Spectrum and shift to rest wavelength """
	with fits.open(file_path, comments='#') as hdul:
		header = hdul[0].header
		spectrum_data = hdul[1].data

	#Assumes the spectrum data is in the format:
	#Wavelength, Flux, FluxError

	wave = np.array(spectrum_data['wavelength'])
	flux = np.array(spectrum_data['flux'])
	flux_err = np.array(spectrum_data['flux_err'])

	observed_wavelength = 10**wave

	spec_res = (observed_wavelength[1]-observed_wavelength[0])  # Assumes that the file has a fixed wavelength resolution

	###########
	# Redshift Correction
	###########

	rest_wavelength = rest_wavelength_converter(observer_frame_wave=observed_wavelength, z=redshift)

	return rest_wavelength, flux, flux_err, spec_res


def save_fits_spectrum(main_file_path, directory, spectrum_ID, suffix, wave, flux, flux_err):
	""" Save FITS Spectrum """
	hdu = fits.BinTableHDU.from_columns(
		[fits.Column(name='wavelength', format='E', array = wave),
		fits.Column(name='flux', format='E', array = flux),
		fits.Column(name = 'flux_err', format = 'E', array = flux_err)])
	
	save_file_path = f'{main_file_path}/{directory}/{spectrum_ID}_{suffix}.fits'
	hdu.writeto(save_file_path, overwrite=True)

	return None
	

def rest_wavelength_converter(observer_wavelength, redshift):
	""" Observed Wavelength to Rest Wavelength Converter """
	
	rest_wavelength = observer_wavelength / (1 + redshift)
	
	return rest_wavelength


def observed_wavelength_converter(rest_wavelength, redshift):
	""" Rest Wavelength to Observed Wavelength Converter """
	
	observed_wavelength = rest_wavelength * (1 + redshift)
	
	return observed_wavelength