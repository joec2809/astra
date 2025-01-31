import sys
sys.path.append('../')

import astra_functions as af
import fake_functions as ff
import numpy as np
import pandas as pd

from random import uniform

feature_area = ff.feature_area()  # Draws from centralised parameter declarations

Lower_Wave = feature_area[0]
Upper_Wave = feature_area[1]

Lower_Shift = feature_area[2]
Upper_Shift = feature_area[3]

feature_names = ['[FeVII]6088',
			'[FeX]6376',
			'[FeXI]7894',
			'[FeXIV]5304',
			'[FeVII]3759',
			'[FeVII]5160',
			'[FeVII]5722']

configs = af.read_config_file('../user.config')
main_filepath = configs[0]
base_data_file = configs[1]
features_data_file = configs[2]

base_filepaths = ''
features_filepaths = ''

# Load in base and features spectra
base_data = pd.read_csv(f'{main_filepath}/{base_data_file}')
base_redshifts = base_data['Redshift']
base_IDs = base_data['ID']

features_data = pd.read_csv(f'{main_filepath}/{features_data_file}')
features_redshifts = features_data['Redshift']

# Create fake spectra directory
af.create_directory(main_filepath, 'fake_spectra')

for i in range(len(base_data)):
	
	# Load in base spectra
	base_rest_wavelength, base_flux, base_flux_err, base_spec_res = af.open_fits_spectrum(base_filepaths[i], base_redshifts[i])

	# Remove features from spectra and replace with continua

	base_flux_with_continua = ff.isolate_features(base_flux, base_rest_wavelength, feature_names, mode='remove')

	# Choose random features to add to base spectra

	features_file_idx = np.random.randint(0,len(features_data))
	features_filepath = features_filepaths[features_file_idx]
	features_redshift = features_redshifts[features_file_idx]

	features_rest_wavelength, features_flux, features_flux_err, features_spec_res = af.open_fits_spectrum(features_filepath, features_redshift)

	# Apply random scaling factor to features and error

	scale_factor_max = 1
	scale_factor = uniform(0, scale_factor_max)
	scaled_features_flux = features_flux * scale_factor
	scaled_features_flux_err = features_flux_err * scale_factor

	# Resample features spectrum and base spectrum to same resolution

	if base_spec_res > features_spec_res:
		features_flux, features_flux_err = ff.spec_resample(base_rest_wavelength, features_rest_wavelength, scaled_features_flux, scaled_features_flux_err)
		base_flux = base_flux_with_continua

	elif base_spec_res < features_spec_res:
		base_flux, base_flux_err = ff.spec_resample(features_rest_wavelength, base_rest_wavelength, base_flux_with_continua, base_flux_err)
		features_flux = scaled_features_flux
		features_flux_err = scaled_features_flux_err

	else:
		base_flux = base_flux_with_continua
		features_flux = scaled_features_flux
		features_flux_err = scaled_features_flux_err

	# Create fake spectra and error propagation

	fake_flux = features_flux + base_flux
	fake_flux_err = np.sqrt(base_flux_err**2 + features_flux_err**2)

	# Change wavelength back to form stored in fits file

	fake_wavelength = af.observed_wavelength_converter(base_rest_wavelength, base_redshifts[i])

	# Save fake spectra to fits file
	af.save_fits_spectrum(main_filepath, 'fake_spectra', base_IDs[i], 'fake', fake_wavelength, fake_flux, fake_flux_err)