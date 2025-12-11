"""Auto-optimization mode - tests different parameter combinations"""

import cv2
import numpy as np
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.detectors.cv.cv_detector import CVDetector
from .auto_calibrate import auto_calibrate
from .ui_display import print_optimization_results


def auto_optimize(cap, base_calibration=None, use_mediapipe_validation=False):
    """
    Auto-optimization mode tests multiple parameter presets
    Measures FPS and detection rate for each preset
    
    If use_mediapipe_validation=True, uses MediaPipe as ground truth to measure
    detection accuracy instead of relying on contour detection alone.
    
    Returns best configuration with optimized parameters
    """
    print("\n" + "=" * 70)
    print("AUTO-OPTIMIZATION MODE")
    print("=" * 70)
    if use_mediapipe_validation:
        print("Using MediaPipe as ground truth for detection quality...")
    print("Testing parameter combinations to maximize FPS and detection quality...")
    print("This will take ~2 minutes. Position your hand in frame.")
    print("=" * 70)
    
    if base_calibration is None:
        print("\nColor calibration options:")
        print("1. Use saved calibration from skin_detection_config.json")
        print("2. Run new auto-calibration (5 seconds)")
        choice = input("\nChoice (1-2): ").strip()
        
        if choice == '1':
            # Load existing calibration
            import json
            config_path = Path(__file__).parent.parent.parent / 'skin_detection_config.json'
            if config_path.exists():
                try:
                    with open(config_path, 'r') as f:
                        config = json.load(f)
                    base_calibration = {
                        'ycrcb_lower': np.array(config['ycrcb_lower'], dtype=np.uint8),
                        'ycrcb_upper': np.array(config['ycrcb_upper'], dtype=np.uint8),
                        'hsv_lower': np.array(config['hsv_lower'], dtype=np.uint8),
                        'hsv_upper': np.array(config['hsv_upper'], dtype=np.uint8)
                    }
                    print(f"\n✓ Loaded calibration from {config_path.name}")
                except Exception as e:
                    print(f"⚠ Failed to load calibration: {e}")
                    print("Running auto-calibration instead...")
                    base_calibration = auto_calibrate(cap)
            else:
                print("⚠ No saved calibration found. Running auto-calibration...")
                base_calibration = auto_calibrate(cap)
        else:
            print("\nRunning auto-calibration...")
            base_calibration = auto_calibrate(cap)
        
        if not base_calibration:
            print("Calibration cancelled")
            return None
    
    best_config = {
        'calibration': base_calibration,
        'denoise_h': 10,
        'kernel_small': 5,
        'kernel_large': 11,
        'morph_iter': 2,
        'min_area': 3000,
        'max_area': 50000,
        'fps': 0,
        'quality_score': 0
    }
    
    # Parameter ranges to test
    test_configs = [
        # Fast (prioritize FPS)
        {'denoise_h': 5, 'kernel_small': 3, 'kernel_large': 7, 'morph_iter': 1, 'min_area': 4000, 'max_area': 40000, 'name': 'Fast'},
        # Balanced
        {'denoise_h': 7, 'kernel_small': 5, 'kernel_large': 9, 'morph_iter': 2, 'min_area': 3000, 'max_area': 50000, 'name': 'Balanced'},
        # Quality (prioritize detection)
        {'denoise_h': 10, 'kernel_small': 5, 'kernel_large': 11, 'morph_iter': 2, 'min_area': 2500, 'max_area': 60000, 'name': 'Quality'},
        # Ultra Fast
        {'denoise_h': 3, 'kernel_small': 3, 'kernel_large': 7, 'morph_iter': 1, 'min_area': 5000, 'max_area': 35000, 'name': 'Ultra-Fast'},
        # Custom variants
        {'denoise_h': 7, 'kernel_small': 3, 'kernel_large': 9, 'morph_iter': 1, 'min_area': 3500, 'max_area': 45000, 'name': 'Fast+Quality'},
        {'denoise_h': 5, 'kernel_small': 5, 'kernel_large': 9, 'morph_iter': 2, 'min_area': 3000, 'max_area': 50000, 'name': 'Smooth'},
    ]
    
    results = []
    
    # Initialize MediaPipe if using for validation
    mp_detector = None
    if use_mediapipe_validation:
        from src.detectors import MediaPipeDetector
        mp_detector = MediaPipeDetector()
        print("MediaPipe initialized for ground truth validation")
    
    for idx, config in enumerate(test_configs):
        print(f"\n[{idx+1}/{len(test_configs)}] Testing {config['name']} preset...")
        
        detector = CVDetector()
        detector.ycrcb_lower = base_calibration['ycrcb_lower']
        detector.ycrcb_upper = base_calibration['ycrcb_upper']
        detector.hsv_lower = base_calibration['hsv_lower']
        detector.hsv_upper = base_calibration['hsv_upper']
        detector.denoise_h = config['denoise_h']
        detector.kernel_small = np.ones((config['kernel_small'], config['kernel_small']), np.uint8)
        detector.kernel_large = np.ones((config['kernel_large'], config['kernel_large']), np.uint8)
        detector.morph_iterations = config['morph_iter']
        detector.min_contour_area = config['min_area']
        detector.max_contour_area = config['max_area']
        
        # Test for 3 seconds
        fps_samples = []
        detection_count = 0
        correct_detections = 0
        false_positives = 0
        false_negatives = 0
        total_frames = 0
        start_time = time.time()
        
        while time.time() - start_time < 3.0:
            frame_start = time.time()
            ret, frame = cap.read()
            if not ret:
                break
            
            # Get CV detector result
            result = detector.process_frame(frame.copy())
            cv_detected = result['detected']
            
            # Create visualization frame
            h, w = frame.shape[:2]
            vis_frame = result['annotated_frame'].copy()
            
            # Get ground truth from MediaPipe if enabled
            if use_mediapipe_validation and mp_detector:
                # Use palm center for proper alignment with CV detector during optimization
                mp_result = mp_detector.process_frame(frame.copy(), use_palm_center=True)
                mp_detected = mp_result['detected']
                
                # Draw MediaPipe detection in blue
                if mp_detected:
                    mp_x, mp_y = mp_result['hand_x'], mp_result['hand_y']
                    mp_x_px = int(mp_x * w)
                    mp_y_px = int(mp_y * h)
                    cv2.circle(vis_frame, (mp_x_px, mp_y_px), 15, (255, 0, 0), 3)  # Blue for MediaPipe
                    cv2.putText(vis_frame, "MP", (mp_x_px + 20, mp_y_px), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                
                # Draw CV detection in green
                if cv_detected:
                    cv_x, cv_y = result['hand_x'], result['hand_y']
                    cv_x_px = int(cv_x * w)
                    cv_y_px = int(cv_y * h)
                    cv2.circle(vis_frame, (cv_x_px, cv_y_px), 15, (0, 255, 0), 3)  # Green for CV
                    cv2.putText(vis_frame, "CV", (cv_x_px + 20, cv_y_px + 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Compare detections using spatial overlap
                if cv_detected and mp_detected:
                    # Calculate IoU (Intersection over Union) between bounding boxes
                    # CV detector bounding box
                    cv_x, cv_y = result['hand_x'], result['hand_y']
                    cv_bbox_size = 100  # Approximate hand size
                    cv_x1 = max(0, int(cv_x * w - cv_bbox_size/2))
                    cv_y1 = max(0, int(cv_y * h - cv_bbox_size/2))
                    cv_x2 = min(w, int(cv_x * w + cv_bbox_size/2))
                    cv_y2 = min(h, int(cv_y * h + cv_bbox_size/2))
                    
                    # MediaPipe bounding box (normalized coordinates)
                    mp_x, mp_y = mp_result['hand_x'], mp_result['hand_y']
                    mp_x_px = int(mp_x * w)
                    mp_y_px = int(mp_y * h)
                    mp_bbox_size = 100
                    mp_x1 = max(0, mp_x_px - mp_bbox_size//2)
                    mp_y1 = max(0, mp_y_px - mp_bbox_size//2)
                    mp_x2 = min(w, mp_x_px + mp_bbox_size//2)
                    mp_y2 = min(h, mp_y_px + mp_bbox_size//2)
                    
                    # Draw bounding boxes
                    cv2.rectangle(vis_frame, (cv_x1, cv_y1), (cv_x2, cv_y2), (0, 255, 0), 2)  # CV in green
                    cv2.rectangle(vis_frame, (mp_x1, mp_y1), (mp_x2, mp_y2), (255, 0, 0), 2)  # MP in blue
                    
                    # Calculate intersection
                    inter_x1 = max(cv_x1, mp_x1)
                    inter_y1 = max(cv_y1, mp_y1)
                    inter_x2 = min(cv_x2, mp_x2)
                    inter_y2 = min(cv_y2, mp_y2)
                    
                    intersection = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
                    cv_area = (cv_x2 - cv_x1) * (cv_y2 - cv_y1)
                    mp_area = (mp_x2 - mp_x1) * (mp_y2 - mp_y1)
                    union = cv_area + mp_area - intersection
                    
                    iou = intersection / union if union > 0 else 0
                    
                    # Draw IoU on frame
                    cv2.putText(vis_frame, f"IoU: {iou:.2f}", (10, h - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    # Consider it a match if IoU > 0.3 (30% overlap)
                    if iou > 0.3:
                        correct_detections += 1
                    else:
                        # Both detected but in different locations
                        false_positives += 1
                        false_negatives += 1
                elif cv_detected and not mp_detected:
                    false_positives += 1
                elif not cv_detected and mp_detected:
                    false_negatives += 1
                # else: both missed (true negative, expected)
                
                detection_count = correct_detections
            else:
                # Original behavior: trust CV detector
                if cv_detected:
                    detection_count += 1
            
            total_frames += 1
            
            frame_time = time.time() - frame_start
            if frame_time > 0:
                fps_samples.append(1.0 / frame_time)
            
            # Add status overlay
            elapsed = time.time() - start_time
            remaining = 3.0 - elapsed
            cv2.putText(vis_frame, f"Testing: {config['name']}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            cv2.putText(vis_frame, f"Time: {remaining:.1f}s", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(vis_frame, f"Detections: CV={correct_detections} FP={false_positives} FN={false_negatives}", 
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            cv2.imshow('Optimizing', vis_frame)
            cv2.waitKey(1)
        
        avg_fps = np.mean(fps_samples) if fps_samples else 0
        detection_rate = (detection_count / total_frames * 100) if total_frames > 0 else 0
        
        # Calculate quality metrics
        if use_mediapipe_validation and mp_detector:
            # Precision: correct / (correct + false_positives)
            precision = (correct_detections / (correct_detections + false_positives)) * 100 if (correct_detections + false_positives) > 0 else 0
            # Recall: correct / (correct + false_negatives)
            recall = (correct_detections / (correct_detections + false_negatives)) * 100 if (correct_detections + false_negatives) > 0 else 0
            # F1 score
            f1_score = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0
            
            # Quality score: balance FPS, precision, and recall
            quality_score = (avg_fps * 0.3) + (precision * 0.35) + (recall * 0.35)
        else:
            # Original quality score
            quality_score = (avg_fps * 0.4) + (detection_rate * 0.6)
            precision = recall = f1_score = None
        
        result_entry = {
            'config': config,
            'fps': avg_fps,
            'detection_rate': detection_rate,
            'quality_score': quality_score
        }
        
        if use_mediapipe_validation and mp_detector:
            result_entry.update({
                'precision': precision,
                'recall': recall,
                'f1_score': f1_score,
                'correct': correct_detections,
                'false_positives': false_positives,
                'false_negatives': false_negatives
            })
            print(f"  → FPS: {avg_fps:.1f} | Precision: {precision:.1f}% | Recall: {recall:.1f}% | F1: {f1_score:.1f} | Score: {quality_score:.1f}")
        else:
            print(f"  → FPS: {avg_fps:.1f} | Detection: {detection_rate:.1f}% | Score: {quality_score:.1f}")
        
        results.append(result_entry)
        
        if quality_score > best_config['quality_score']:
            best_config.update({
                'denoise_h': config['denoise_h'],
                'kernel_small': config['kernel_small'],
                'kernel_large': config['kernel_large'],
                'morph_iter': config['morph_iter'],
                'min_area': config['min_area'],
                'max_area': config['max_area'],
                'fps': avg_fps,
                'quality_score': quality_score,
                'detection_rate': detection_rate,
                'name': config['name']
            })
            if use_mediapipe_validation and mp_detector:
                best_config.update({
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1_score
                })
    
    # Cleanup MediaPipe if used
    if mp_detector:
        mp_detector.cleanup()
    
    cv2.destroyAllWindows()
    
    # Display results
    print_optimization_results(results, best_config)
    
    # Add calibration to best config for saving
    best_config['ycrcb_lower'] = base_calibration['ycrcb_lower']
    best_config['ycrcb_upper'] = base_calibration['ycrcb_upper']
    best_config['hsv_lower'] = base_calibration['hsv_lower']
    best_config['hsv_upper'] = base_calibration['hsv_upper']
    
    return best_config
