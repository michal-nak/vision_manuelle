"""Calibration I/O operations - save, load, verify calibration data"""

import json
import re
from datetime import datetime
from pathlib import Path


def save_calibration(calibration):
    """
    Save calibration to skin_detection_config.json and create backup JSON
    Supports both simple calibration (color ranges only) and optimized config (with processing params)
    """
    # Prepare config data
    config = {
        'ycrcb_lower': calibration['ycrcb_lower'].tolist(),
        'ycrcb_upper': calibration['ycrcb_upper'].tolist(),
        'hsv_lower': calibration['hsv_lower'].tolist(),
        'hsv_upper': calibration['hsv_upper'].tolist()
    }
    
    # Add processing parameters if they exist (from auto-optimize)
    if 'denoise_h' in calibration:
        config['denoise_h'] = calibration['denoise_h']
        config['kernel_small'] = calibration['kernel_small']
        config['kernel_large'] = calibration['kernel_large']
        config['morph_iterations'] = calibration['morph_iter']
        config['min_contour_area'] = calibration['min_area']
        config['max_contour_area'] = calibration.get('max_area', 50000)
    
    # Create backup JSON
    backup = {
        'timestamp': datetime.now().isoformat(),
        **config
    }
    backup_path = Path(__file__).parent.parent.parent / 'calibration_backup.json'
    with open(backup_path, 'w') as f:
        json.dump(backup, f, indent=2)
    print(f"\n✓ Backup saved to {backup_path}")
    
    # Save to skin_detection_config.json in root directory
    config_path = Path(__file__).parent.parent.parent / 'skin_detection_config.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    if 'denoise_h' in calibration:
        print(f"✓ {config_path.name} updated with optimized processing parameters!")
    else:
        print(f"✓ {config_path.name} updated!")


def verify_calibration():
    """
    Verify current calibration status from skin_detection_config.json
    Shows current values and whether they are default or custom
    """
    print("\n" + "=" * 70)
    print("CALIBRATION STATUS")
    print("=" * 70)
    
    config_path = Path(__file__).parent.parent.parent / 'skin_detection_config.json'
    
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        ycrcb_l = config.get('ycrcb_lower', [])
        ycrcb_u = config.get('ycrcb_upper', [])
        hsv_l = config.get('hsv_lower', [])
        hsv_u = config.get('hsv_upper', [])
        
        print("Current values in skin_detection_config.json:")
        print(f"  YCrCb: {ycrcb_l} to {ycrcb_u}")
        print(f"  HSV:   {hsv_l} to {hsv_u}")
        
        is_default = (ycrcb_l == [0, 133, 77] and ycrcb_u == [255, 173, 127] and
                     hsv_l == [0, 30, 60] and hsv_u == [20, 150, 255])
        
        if is_default:
            print("\n⚠ Using DEFAULT values (not calibrated)")
        else:
            print("\n✓ Using CALIBRATED values")
    else:
        print("⚠ No skin_detection_config.json found")
        print("Using default values from code")
    
    backup_path = Path(__file__).parent.parent.parent / 'calibration_backup.json'
    if backup_path.exists():
        with open(backup_path, 'r') as f:
            backup = json.load(f)
        print(f"\nBackup found: {backup.get('timestamp', 'unknown')}")
    else:
        print("\n⚠ No backup file")
    print("=" * 70)
