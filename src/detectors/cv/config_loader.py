"""
Configuration loading for CV detector
"""
import json
import numpy as np
from pathlib import Path
from ...core.config import YCRCB_LOWER, YCRCB_UPPER, HSV_LOWER, HSV_UPPER


def load_skin_detection_config():
    """Load skin detection color bounds from JSON config or use defaults"""
    config_path = Path(__file__).parent.parent.parent.parent / 'skin_detection_config.json'
    
    if config_path.exists():
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            ycrcb_lower = np.array(config['ycrcb_lower'], dtype=np.uint8)
            ycrcb_upper = np.array(config['ycrcb_upper'], dtype=np.uint8)
            hsv_lower = np.array(config['hsv_lower'], dtype=np.uint8)
            hsv_upper = np.array(config['hsv_upper'], dtype=np.uint8)
            
            return ycrcb_lower, ycrcb_upper, hsv_lower, hsv_upper
        except Exception as e:
            print(f"⚠️  Failed to load skin detection config: {e}")
    
    # Fall back to defaults from config.py
    ycrcb_lower = np.array(YCRCB_LOWER, dtype=np.uint8)
    ycrcb_upper = np.array(YCRCB_UPPER, dtype=np.uint8)
    hsv_lower = np.array(HSV_LOWER, dtype=np.uint8)
    hsv_upper = np.array(HSV_UPPER, dtype=np.uint8)
    
    return ycrcb_lower, ycrcb_upper, hsv_lower, hsv_upper


def load_processing_params():
    """Load processing parameters from JSON config or use defaults"""
    config_path = Path(__file__).parent.parent.parent.parent / 'skin_detection_config.json'
    
    # Default values
    defaults = {
        'denoise_h': 10,
        'kernel_small': 3,
        'kernel_large': 7,
        'morph_iterations': 1,  # Reduced from 3 - was over-eroding hands (80% detection failure)
        'min_contour_area': 1000,
        'max_contour_area': 50000
    }
    
    if config_path.exists():
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Load processing params if they exist, otherwise use defaults
            return {
                'denoise_h': config.get('denoise_h', defaults['denoise_h']),
                'kernel_small': config.get('kernel_small', defaults['kernel_small']),
                'kernel_large': config.get('kernel_large', defaults['kernel_large']),
                'morph_iterations': config.get('morph_iterations', defaults['morph_iterations']),
                'min_contour_area': config.get('min_contour_area', defaults['min_contour_area']),
                'max_contour_area': config.get('max_contour_area', defaults['max_contour_area'])
            }
        except Exception as e:
            print(f"⚠️  Failed to load processing params: {e}")
    
    return defaults
