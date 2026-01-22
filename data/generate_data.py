"""
Data Generation Script for Deliverable 2
BMEN 509/623 - Introduction to Biomedical Imaging

This script generates all data files for the X-ray Projection & CT assignment.
Run this script once to create all necessary data files.
"""

import numpy as np
import pandas as pd
from ct_utils import (
    create_shepp_logan_phantom, 
    forward_project,
    add_beam_hardening,
    add_ring_artifact
)

print("=" * 60)
print("Generating data files for Deliverable 2")
print("=" * 60)

# =============================================================================
# 1. Projection Geometry Scenarios
# =============================================================================
print("\n1. Creating projection geometry scenarios...")

scenarios = {
    'scenario_id': [1, 2, 3, 4, 5],
    'name': [
        'Chest PA',
        'Hand/Wrist',
        'Lateral Spine',
        'Skull AP',
        'Magnification Mammography'
    ],
    'body_part_thickness_cm': [25, 3, 35, 20, 5],
    'object_depth_cm': [12, 1.5, 17, 10, 2.5],  # Depth of structure of interest
    'min_resolution_lp_mm': [2.0, 5.0, 1.5, 2.5, 10.0],  # Required spatial resolution
    'receptor_blur_mm': [0.15, 0.10, 0.15, 0.15, 0.05],  # Digital detector blur
    'max_mAs': [50, 10, 100, 40, 150],  # Tube loading limit
    'reference_SID_cm': [180, 100, 180, 100, 65],  # Typical clinical SID
    'notes': [
        'Standard chest technique',
        'Extremity technique, fine detail',
        'High mAs for penetration',
        'Table-top technique',
        'Contact + magnification comparison'
    ]
}

df_scenarios = pd.DataFrame(scenarios)
df_scenarios.to_csv('projection_scenarios.csv', index=False)
print(f"   Created projection_scenarios.csv ({len(df_scenarios)} scenarios)")

# =============================================================================
# 2. Shepp-Logan Phantom
# =============================================================================
print("\n2. Creating Shepp-Logan phantom...")

phantom = create_shepp_logan_phantom(256)
np.save('shepp_logan_phantom.npy', phantom)
print(f"   Created shepp_logan_phantom.npy (shape: {phantom.shape})")

# =============================================================================
# 3. Sinograms at Different Angular Sampling
# =============================================================================
print("\n3. Generating sinograms at different angular sampling...")

for n_angles in [180, 90, 45]:
    angles = np.linspace(0, 180, n_angles, endpoint=False)
    sinogram = forward_project(phantom, angles)
    
    # Add small amount of noise
    noise = np.random.normal(0, 0.01 * np.max(sinogram), sinogram.shape)
    sinogram_noisy = sinogram + noise
    
    np.save(f'sinogram_{n_angles}.npy', sinogram_noisy)
    print(f"   Created sinogram_{n_angles}.npy (shape: {sinogram_noisy.shape})")

# Also save angles for reference
np.save('projection_angles_180.npy', np.linspace(0, 180, 180, endpoint=False))
np.save('projection_angles_90.npy', np.linspace(0, 180, 90, endpoint=False))
np.save('projection_angles_45.npy', np.linspace(0, 180, 45, endpoint=False))
print("   Created angle arrays")

# =============================================================================
# 4. CT Phantom Images for Artifact Analysis
# =============================================================================
print("\n4. Creating CT phantom images with artifacts...")

# Create a simpler phantom for HU analysis (circular with inserts)
def create_hu_phantom(size=256):
    """Create a water phantom with tissue-equivalent inserts."""
    phantom = np.zeros((size, size))
    center = size // 2
    
    # Create coordinate grids
    y, x = np.ogrid[:size, :size]
    r = np.sqrt((x - center) ** 2 + (y - center) ** 2)
    
    # Outer water bath (HU = 0, mu ~ 0.019 mm^-1)
    water_mask = r < 0.4 * size
    phantom[water_mask] = 0.019
    
    # Insert positions (angle, radius_fraction, insert_radius)
    inserts = [
        # (angle_deg, r_frac, radius, mu_value, material_name)
        (0, 0.25, 15, 0.0, 'air'),           # Air: HU = -1000
        (60, 0.25, 15, 0.016, 'fat'),        # Fat: HU ~ -100
        (120, 0.25, 15, 0.019, 'water'),     # Water: HU = 0
        (180, 0.25, 15, 0.021, 'muscle'),    # Muscle: HU ~ 40
        (240, 0.25, 15, 0.025, 'liver'),     # Liver: HU ~ 60
        (300, 0.25, 15, 0.038, 'bone'),      # Bone: HU ~ 1000
    ]
    
    insert_info = []
    for angle_deg, r_frac, ins_radius, mu_val, name in inserts:
        angle = np.radians(angle_deg)
        cx = center + int(r_frac * size * np.cos(angle))
        cy = center + int(r_frac * size * np.sin(angle))
        
        # Create insert mask
        ins_mask = (x - cx) ** 2 + (y - cy) ** 2 < ins_radius ** 2
        phantom[ins_mask] = mu_val
        
        insert_info.append({
            'material': name,
            'mu_mm_inv': mu_val,
            'expected_hu': 1000 * (mu_val - 0.019) / 0.019 if mu_val > 0 else -1000,
            'center_x': cx,
            'center_y': cy,
            'radius': ins_radius
        })
    
    return phantom, insert_info

hu_phantom, insert_info = create_hu_phantom(256)

# Save clean phantom
np.save('ct_phantom_clean.npy', hu_phantom)
print(f"   Created ct_phantom_clean.npy")

# Create versions with artifacts
phantom_beam_hard = add_beam_hardening(hu_phantom, strength=0.005)
np.save('ct_phantom_beam_hardening.npy', phantom_beam_hard)
print(f"   Created ct_phantom_beam_hardening.npy")

phantom_ring = add_ring_artifact(hu_phantom, n_rings=4, strength=0.003)
np.save('ct_phantom_ring.npy', phantom_ring)
print(f"   Created ct_phantom_ring.npy")

# Combined artifacts
phantom_combined = add_ring_artifact(
    add_beam_hardening(hu_phantom, strength=0.003),
    n_rings=2, strength=0.002
)
np.save('ct_phantom_combined_artifacts.npy', phantom_combined)
print(f"   Created ct_phantom_combined_artifacts.npy")

# =============================================================================
# 5. HU Calibration Data
# =============================================================================
print("\n5. Creating HU calibration data...")

# Standard materials for HU calibration
calibration_data = {
    'material': ['air', 'lung', 'fat', 'water', 'muscle', 'liver', 
                 'blood', 'trabecular_bone', 'cortical_bone'],
    'expected_hu_low': [-1000, -900, -120, -5, 35, 50, 55, 100, 800],
    'expected_hu_high': [-1000, -600, -60, 5, 55, 70, 75, 300, 1900],
    'typical_hu': [-1000, -750, -90, 0, 45, 60, 65, 200, 1000],
    'mu_at_70keV_mm_inv': [0.0, 0.005, 0.016, 0.019, 0.021, 0.021, 
                           0.021, 0.025, 0.038],
    'density_g_cm3': [0.0012, 0.26, 0.92, 1.00, 1.05, 1.05, 
                      1.06, 1.18, 1.85]
}

df_calibration = pd.DataFrame(calibration_data)
df_calibration.to_csv('hu_calibration.csv', index=False)
print(f"   Created hu_calibration.csv ({len(df_calibration)} materials)")

# Insert location info for the phantom
df_inserts = pd.DataFrame(insert_info)
df_inserts.to_csv('phantom_inserts.csv', index=False)
print(f"   Created phantom_inserts.csv ({len(df_inserts)} inserts)")

# =============================================================================
# 6. Focal Spot Information
# =============================================================================
print("\n6. Creating focal spot data...")

focal_spot_data = {
    'spot_type': ['fine', 'broad', 'micro'],
    'nominal_size_mm': [0.3, 1.0, 0.1],
    'actual_size_mm': [0.4, 1.2, 0.15],  # Measured sizes often larger
    'max_power_kW': [15, 80, 5],
    'typical_use': [
        'Extremities, mammography, detail work',
        'General radiography, fluoroscopy, high output',
        'Magnification mammography only'
    ]
}

df_focal = pd.DataFrame(focal_spot_data)
df_focal.to_csv('focal_spot_data.csv', index=False)
print(f"   Created focal_spot_data.csv")

# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 60)
print("Data generation complete!")
print("=" * 60)
print("\nFiles created:")
print("  - projection_scenarios.csv    (clinical geometry scenarios)")
print("  - shepp_logan_phantom.npy     (standard CT test phantom)")
print("  - sinogram_180.npy            (180 projection sinogram)")
print("  - sinogram_90.npy             (90 projection sinogram)")  
print("  - sinogram_45.npy             (45 projection sinogram)")
print("  - projection_angles_*.npy     (angle arrays)")
print("  - ct_phantom_clean.npy        (HU calibration phantom)")
print("  - ct_phantom_beam_hardening.npy")
print("  - ct_phantom_ring.npy")
print("  - ct_phantom_combined_artifacts.npy")
print("  - hu_calibration.csv          (material HU values)")
print("  - phantom_inserts.csv         (insert locations)")
print("  - focal_spot_data.csv         (X-ray tube focal spots)")
