#!/usr/bin/env python3
"""
Create pre-reconstructed CT images for Deliverable 2.
"""

import numpy as np
from skimage.transform import iradon
import shutil

# Load sinograms and transpose (iradon expects [detectors, angles])
sino_180 = np.load('sinogram_180.npy').T
sino_90 = np.load('sinogram_90.npy').T  
sino_45 = np.load('sinogram_45.npy').T

# Load projection angles
angles_180 = np.load('projection_angles_180.npy')
angles_90 = np.load('projection_angles_90.npy')
angles_45 = np.load('projection_angles_45.npy')

print(f'Transposed sinogram shapes: {sino_180.shape}, {sino_90.shape}, {sino_45.shape}')

# Reconstruct using iradon (FBP)
print('Reconstructing with 180 projections...')
recon_180 = iradon(sino_180, theta=angles_180, filter_name='ramp')
print('Reconstructing with 90 projections...')
recon_90 = iradon(sino_90, theta=angles_90, filter_name='ramp')
print('Reconstructing with 45 projections...')
recon_45 = iradon(sino_45, theta=angles_45, filter_name='ramp')

# Save pre-reconstructed images
np.save('reconstructed_180proj.npy', recon_180)
np.save('reconstructed_90proj.npy', recon_90)
np.save('reconstructed_45proj.npy', recon_45)

# Also copy ct_phantom_clean.npy to ct_phantom.npy for consistency
shutil.copy('ct_phantom_clean.npy', 'ct_phantom.npy')

# Create HU reference file with standard tissue values
hu_data = """Material,HU_min,HU_max,HU_typical
Air,-1000,-950,-1000
Lung,-950,-500,-700
Fat,-120,-60,-100
Water,-10,10,0
Soft Tissue,20,80,40
Muscle,35,55,45
Blood,30,50,40
Liver,40,70,55
Bone (Cortical),300,2000,1000
Bone (Trabecular),100,300,200
"""

with open('hu_reference.csv', 'w') as f:
    f.write(hu_data)

print('\nCreated pre-reconstructed images:')
print(f'  reconstructed_180proj.npy: shape {recon_180.shape}')
print(f'  reconstructed_90proj.npy: shape {recon_90.shape}')
print(f'  reconstructed_45proj.npy: shape {recon_45.shape}')
print('  ct_phantom.npy: copied from ct_phantom_clean.npy')
print('  hu_reference.csv: HU reference values')
