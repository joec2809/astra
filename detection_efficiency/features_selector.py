import sys
sys.path.append('../')

import astra_functions as af
import fake_functions as ff
import numpy as np
import pandas as pd

configs = af.read_config_file('../user.config')
main_filepath = configs[0]
base_data_file = configs[1]

feature_spectra_filepaths = ''

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

spectra_metadata = pd.read_csv(f'{main_filepath}/{base_data_file}')
spectra_names = spectra_metadata['Spectrum']
spectra_redshifts = spectra_metadata['Redshift']

af.create_directory(main_filepath, 'features_spectra')

for i, redshift in enumerate(spectra_redshifts):

    wave, flux, flux_err, spec_res = af.open_fits_spectrum(f'{main_filepath}/{feature_spectra_filepaths[i]}', redshift)
    
    # Set most of spectra flux to zero, leaving the features only

    features_flux = ff.isolate_features(flux, wave, feature_names, mode='select')

    # Error propagation of subtracting flux

    features_flux_err = np.sqrt(2)*flux_err

    # Save the features only spectrum
    af.save_fits_spectrum(main_filepath, 'features_spectra', spectra_names[i], 'features', wave, features_flux, features_flux_err)