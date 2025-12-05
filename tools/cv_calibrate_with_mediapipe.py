"""
CV Detector Calibration using MediaPipe as Ground Truth
Uses MediaPipe to identify where the hand actually is, then tunes CV parameters to match
"""
import cv2
import numpy as np
import sys
from pathlib import Path
from collections import defaultdict
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.detectors import CVDetector, MediaPipeDetector
from src.core.utils import find_camera, setup_camera, FPSCounter
from src.core.config import CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS

class CVCalibratorWithMediaPipe:
    def __init__(self):
        self.mp_detector = MediaPipeDetector()
        self.cv_detector = CVDetector()
        
        # Trackbar values
        self.ycrcb_y_min = 0
        self.ycrcb_y_max = 255
        self.ycrcb_cr_min = 133
        self.ycrcb_cr_max = 173
        self.ycrcb_cb_min = 77
        self.ycrcb_cb_max = 127
        
        self.hsv_h_min = 0
        self.hsv_h_max = 20
        self.hsv_s_min = 30
        self.hsv_s_max = 150
        self.hsv_v_min = 60
        self.hsv_v_max = 255
        
        self.denoise_h = 10
        self.kernel_small = 3
        self.kernel_large = 7
        self.morph_iter = 2
        self.min_area = 3000
        
        # Statistics
        self.stats = {
            'mp_detections': 0,
            'cv_detections': 0,
            'matches': 0,
            'misses': 0,
            'false_positives': 0,
            'total_frames': 0
        }
        
        self.hand_regions = []  # Store MediaPipe hand regions for analysis
        
        # Optimization presets (from calibrate.py AUTO-OPTIMIZE)
        self.presets = [
            {'name': 'Balanced', 'denoise_h': 10, 'kernel_small': 3, 'kernel_large': 7, 'morph_iter': 2, 'min_area': 3000},
            {'name': 'Speed', 'denoise_h': 5, 'kernel_small': 3, 'kernel_large': 5, 'morph_iter': 1, 'min_area': 2000},
            {'name': 'Accuracy', 'denoise_h': 15, 'kernel_small': 5, 'kernel_large': 9, 'morph_iter': 3, 'min_area': 4000},
            {'name': 'Low Noise', 'denoise_h': 20, 'kernel_small': 7, 'kernel_large': 11, 'morph_iter': 4, 'min_area': 3500},
            {'name': 'Fast Weak', 'denoise_h': 3, 'kernel_small': 3, 'kernel_large': 3, 'morph_iter': 1, 'min_area': 1500},
            {'name': 'Heavy', 'denoise_h': 12, 'kernel_small': 5, 'kernel_large': 9, 'morph_iter': 3, 'min_area': 3000}
        ]
        self.current_preset_idx = 0
        
    def create_trackbars(self):
        cv2.namedWindow('Controls')
        
        # YCrCb trackbars
        cv2.createTrackbar('YCrCb Y Min', 'Controls', self.ycrcb_y_min, 255, lambda x: setattr(self, 'ycrcb_y_min', x))
        cv2.createTrackbar('YCrCb Y Max', 'Controls', self.ycrcb_y_max, 255, lambda x: setattr(self, 'ycrcb_y_max', x))
        cv2.createTrackbar('YCrCb Cr Min', 'Controls', self.ycrcb_cr_min, 255, lambda x: setattr(self, 'ycrcb_cr_min', x))
        cv2.createTrackbar('YCrCb Cr Max', 'Controls', self.ycrcb_cr_max, 255, lambda x: setattr(self, 'ycrcb_cr_max', x))
        cv2.createTrackbar('YCrCb Cb Min', 'Controls', self.ycrcb_cb_min, 255, lambda x: setattr(self, 'ycrcb_cb_min', x))
        cv2.createTrackbar('YCrCb Cb Max', 'Controls', self.ycrcb_cb_max, 255, lambda x: setattr(self, 'ycrcb_cb_max', x))
        
        # HSV trackbars
        cv2.createTrackbar('HSV H Min', 'Controls', self.hsv_h_min, 179, lambda x: setattr(self, 'hsv_h_min', x))
        cv2.createTrackbar('HSV H Max', 'Controls', self.hsv_h_max, 179, lambda x: setattr(self, 'hsv_h_max', x))
        cv2.createTrackbar('HSV S Min', 'Controls', self.hsv_s_min, 255, lambda x: setattr(self, 'hsv_s_min', x))
        cv2.createTrackbar('HSV S Max', 'Controls', self.hsv_s_max, 255, lambda x: setattr(self, 'hsv_s_max', x))
        cv2.createTrackbar('HSV V Min', 'Controls', self.hsv_v_min, 255, lambda x: setattr(self, 'hsv_v_min', x))
        cv2.createTrackbar('HSV V Max', 'Controls', self.hsv_v_max, 255, lambda x: setattr(self, 'hsv_v_max', x))
        
        # Processing parameters
        cv2.createTrackbar('Denoise H', 'Controls', self.denoise_h, 30, lambda x: setattr(self, 'denoise_h', max(1, x)))
        cv2.createTrackbar('Kernel Small', 'Controls', self.kernel_small, 15, lambda x: setattr(self, 'kernel_small', max(1, x) | 1))
        cv2.createTrackbar('Kernel Large', 'Controls', self.kernel_large, 20, lambda x: setattr(self, 'kernel_large', max(1, x) | 1))
        cv2.createTrackbar('Morph Iter', 'Controls', self.morph_iter, 10, lambda x: setattr(self, 'morph_iter', x))
        cv2.createTrackbar('Min Area/100', 'Controls', self.min_area // 100, 100, lambda x: setattr(self, 'min_area', x * 100))
        
    def apply_cv_parameters(self):
        """Apply current trackbar values to CV detector"""
        self.cv_detector.ycrcb_lower = np.array([self.ycrcb_y_min, self.ycrcb_cr_min, self.ycrcb_cb_min], dtype=np.uint8)
        self.cv_detector.ycrcb_upper = np.array([self.ycrcb_y_max, self.ycrcb_cr_max, self.ycrcb_cb_max], dtype=np.uint8)
        self.cv_detector.hsv_lower = np.array([self.hsv_h_min, self.hsv_s_min, self.hsv_v_min], dtype=np.uint8)
        self.cv_detector.hsv_upper = np.array([self.hsv_h_max, self.hsv_s_max, self.hsv_v_max], dtype=np.uint8)
        self.cv_detector.denoise_h = self.denoise_h
        self.cv_detector.kernel_small = max(1, self.kernel_small) | 1
        self.cv_detector.kernel_large = max(1, self.kernel_large) | 1
        self.cv_detector.morph_iterations = self.morph_iter
        self.cv_detector.min_area = self.min_area
    
    def get_hand_bbox_from_mediapipe(self, result):
        """Extract bounding box from MediaPipe landmarks"""
        if not result['detected']:
            return None
        
        # Get hand position (normalized)
        hand_x = result['hand_x']
        hand_y = result['hand_y']
        
        # Estimate bbox size (MediaPipe doesn't give bbox directly, so we estimate)
        # Typical hand is about 15-20% of frame width
        bbox_w = 0.2
        bbox_h = 0.25
        
        x1 = max(0, hand_x - bbox_w/2)
        y1 = max(0, hand_y - bbox_h/2)
        x2 = min(1, hand_x + bbox_w/2)
        y2 = min(1, hand_y + bbox_h/2)
        
        return (x1, y1, x2, y2)
    
    def compute_iou(self, bbox1, bbox2):
        """Compute Intersection over Union between two bboxes"""
        if bbox1 is None or bbox2 is None:
            return 0.0
        
        x1_min, y1_min, x1_max, y1_max = bbox1
        x2_min, y2_min, x2_max, y2_max = bbox2
        
        # Intersection
        xi_min = max(x1_min, x2_min)
        yi_min = max(y1_min, y2_min)
        xi_max = min(x1_max, x2_max)
        yi_max = min(y1_max, y2_max)
        
        if xi_max < xi_min or yi_max < yi_min:
            return 0.0
        
        intersection = (xi_max - xi_min) * (yi_max - yi_min)
        
        # Union
        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def get_cv_bbox(self, result, frame_shape):
        """Get CV detector bbox from hand position"""
        if not result['detected']:
            return None
        
        h, w = frame_shape[:2]
        hand_x = result['hand_x']
        hand_y = result['hand_y']
        
        # Estimate bbox
        bbox_w = 0.2
        bbox_h = 0.25
        
        x1 = max(0, hand_x - bbox_w/2)
        y1 = max(0, hand_y - bbox_h/2)
        x2 = min(1, hand_x + bbox_w/2)
        y2 = min(1, hand_y + bbox_h/2)
        
        return (x1, y1, x2, y2)
    
    def analyze_hand_region(self, frame, mp_result):
        """Analyze color values in the MediaPipe-detected hand region"""
        if not mp_result['detected']:
            return
        
        h, w = frame.shape[:2]
        bbox = self.get_hand_bbox_from_mediapipe(mp_result)
        if bbox is None:
            return
        
        x1, y1, x2, y2 = bbox
        x1_px, y1_px = int(x1 * w), int(y1 * h)
        x2_px, y2_px = int(x2 * w), int(y2 * h)
        
        # Extract hand region
        hand_region = frame[y1_px:y2_px, x1_px:x2_px]
        if hand_region.size == 0:
            return
        
        # Convert to color spaces
        ycrcb_region = cv2.cvtColor(hand_region, cv2.COLOR_BGR2YCrCb)
        hsv_region = cv2.cvtColor(hand_region, cv2.COLOR_BGR2HSV)
        
        # Store for analysis
        self.hand_regions.append({
            'ycrcb': ycrcb_region,
            'hsv': hsv_region
        })
        
        # Keep only last 100 regions
        if len(self.hand_regions) > 100:
            self.hand_regions.pop(0)
    
    def get_optimal_thresholds(self):
        """Compute optimal thresholds from collected hand regions"""
        if len(self.hand_regions) < 10:
            return None
        
        ycrcb_pixels = []
        hsv_pixels = []
        
        for region in self.hand_regions[-50:]:  # Use last 50 regions
            ycrcb_pixels.append(region['ycrcb'].reshape(-1, 3))
            hsv_pixels.append(region['hsv'].reshape(-1, 3))
        
        ycrcb_pixels = np.vstack(ycrcb_pixels)
        hsv_pixels = np.vstack(hsv_pixels)
        
        # Compute percentiles for robustness
        ycrcb_lower = np.percentile(ycrcb_pixels, 5, axis=0).astype(np.uint8)
        ycrcb_upper = np.percentile(ycrcb_pixels, 95, axis=0).astype(np.uint8)
        hsv_lower = np.percentile(hsv_pixels, 5, axis=0).astype(np.uint8)
        hsv_upper = np.percentile(hsv_pixels, 95, axis=0).astype(np.uint8)
        
        return {
            'ycrcb_lower': ycrcb_lower,
            'ycrcb_upper': ycrcb_upper,
            'hsv_lower': hsv_lower,
            'hsv_upper': hsv_upper
        }
    
    def run_calibration(self):
        print("=" * 70)
        print("CV CALIBRATION WITH MEDIAPIPE GROUND TRUTH")
        print("=" * 70)
        print("\nMediaPipe (left) provides ground truth for CV tuning (right)")
        print("\nControls:")
        print("  'a' - Auto-suggest thresholds from MediaPipe hand regions")
        print("  'p' - Cycle through optimization presets (Balanced/Speed/Accuracy/etc)")
        print("  'o' - Auto-optimize: test all presets and pick best")
        print("  's' - Save current parameters")
        print("  'r' - Reset statistics")
        print("  'q' - Quit")
        print("=" * 70)
        
        cap = find_camera()
        if not cap:
            print("Could not open camera")
            return
        
        setup_camera(cap, CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS)
        self.create_trackbars()
        
        fps_counter = FPSCounter()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Apply current parameters
            self.apply_cv_parameters()
            
            # Process with both detectors
            frame_mp = frame.copy()
            frame_cv = frame.copy()
            
            mp_result = self.mp_detector.process_frame(frame_mp)
            cv_result = self.cv_detector.process_frame(frame_cv)
            
            # Analyze hand region from MediaPipe
            self.analyze_hand_region(frame, mp_result)
            
            # Update statistics
            self.stats['total_frames'] += 1
            if mp_result['detected']:
                self.stats['mp_detections'] += 1
            if cv_result['detected']:
                self.stats['cv_detections'] += 1
            
            # Compute match quality (IoU)
            mp_bbox = self.get_hand_bbox_from_mediapipe(mp_result)
            cv_bbox = self.get_cv_bbox(cv_result, frame.shape)
            iou = self.compute_iou(mp_bbox, cv_bbox)
            
            if mp_result['detected'] and cv_result['detected']:
                if iou > 0.3:  # Threshold for "match"
                    self.stats['matches'] += 1
                else:
                    self.stats['misses'] += 1
            elif mp_result['detected'] and not cv_result['detected']:
                self.stats['misses'] += 1
            elif not mp_result['detected'] and cv_result['detected']:
                self.stats['false_positives'] += 1
            
            # Draw comparison
            h, w = frame.shape[:2]
            
            # Draw bboxes on original frames
            if mp_bbox:
                x1, y1, x2, y2 = mp_bbox
                cv2.rectangle(mp_result['annotated_frame'], 
                            (int(x1*w), int(y1*h)), (int(x2*w), int(y2*h)), 
                            (0, 255, 0), 2)
                cv2.putText(mp_result['annotated_frame'], "MediaPipe", 
                           (int(x1*w), int(y1*h)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            if cv_bbox:
                x1, y1, x2, y2 = cv_bbox
                color = (0, 255, 0) if iou > 0.3 else (0, 0, 255)
                cv2.rectangle(cv_result['annotated_frame'], 
                            (int(x1*w), int(y1*h)), (int(x2*w), int(y2*h)), 
                            color, 2)
                cv2.putText(cv_result['annotated_frame'], f"CV (IoU: {iou:.2f})", 
                           (int(x1*w), int(y1*h)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Combine frames
            combined = np.hstack([mp_result['annotated_frame'], cv_result['annotated_frame']])
            
            # Add statistics overlay at BOTTOM to avoid masking camera
            fps = fps_counter.update()
            combined_h = combined.shape[0]
            stats_y = combined_h - 160  # Start from bottom
            line_height = 22
            
            # Semi-transparent background for stats
            overlay = combined.copy()
            cv2.rectangle(overlay, (0, combined_h - 165), (400, combined_h), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, combined, 0.3, 0, combined)
            
            cv2.putText(combined, f"FPS: {int(fps)}", (10, stats_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            stats_y += line_height
            
            if self.stats['total_frames'] > 0:
                mp_rate = (self.stats['mp_detections'] / self.stats['total_frames']) * 100
                cv_rate = (self.stats['cv_detections'] / self.stats['total_frames']) * 100
                match_rate = (self.stats['matches'] / max(1, self.stats['mp_detections'])) * 100
                
                cv2.putText(combined, f"MP: {mp_rate:.1f}% | CV: {cv_rate:.1f}%", (10, stats_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
                stats_y += line_height
                
                match_color = (0, 255, 0) if match_rate > 70 else (0, 255, 255) if match_rate > 50 else (0, 0, 255)
                cv2.putText(combined, f"Match: {match_rate:.1f}% | IoU: {iou:.2f}", (10, stats_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.45, match_color, 1)
                stats_y += line_height
                cv2.putText(combined, f"Miss: {self.stats['misses']} | FP: {self.stats['false_positives']}", (10, stats_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 100, 100), 1)
                stats_y += line_height
                cv2.putText(combined, f"Hand Regions: {len(self.hand_regions)}", (10, stats_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
            
            cv2.imshow('Calibration: MediaPipe (Ground Truth) | CV (Tuning)', combined)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('p'):
                # Cycle through presets
                self.current_preset_idx = (self.current_preset_idx + 1) % len(self.presets)
                preset = self.presets[self.current_preset_idx]
                print(f"\nApplying preset: {preset['name']}")
                cv2.setTrackbarPos('Denoise H', 'Controls', preset['denoise_h'])
                cv2.setTrackbarPos('Kernel Small', 'Controls', preset['kernel_small'])
                cv2.setTrackbarPos('Kernel Large', 'Controls', preset['kernel_large'])
                cv2.setTrackbarPos('Morph Iter', 'Controls', preset['morph_iter'])
                cv2.setTrackbarPos('Min Area/100', 'Controls', preset['min_area'] // 100)
            elif key == ord('o'):
                # Auto-optimize: test all presets
                print("\n" + "=" * 70)
                print("AUTO-OPTIMIZATION: Testing all presets...")
                print("=" * 70)
                best_preset = self.auto_optimize_presets(cap)
                if best_preset:
                    print(f"\nBest preset: {best_preset['name']} (Match rate: {best_preset['match_rate']:.1f}%)")
                    print("Applying best preset...")
                    cv2.setTrackbarPos('Denoise H', 'Controls', best_preset['denoise_h'])
                    cv2.setTrackbarPos('Kernel Small', 'Controls', best_preset['kernel_small'])
                    cv2.setTrackbarPos('Kernel Large', 'Controls', best_preset['kernel_large'])
                    cv2.setTrackbarPos('Morph Iter', 'Controls', best_preset['morph_iter'])
                    cv2.setTrackbarPos('Min Area/100', 'Controls', best_preset['min_area'] // 100)
            elif key == ord('a'):
                # Auto-suggest thresholds
                optimal = self.get_optimal_thresholds()
                if optimal:
                    print("\n" + "=" * 70)
                    print("AUTO-SUGGESTED THRESHOLDS (from MediaPipe hand regions)")
                    print("=" * 70)
                    print(f"YCrCb Lower: {optimal['ycrcb_lower']}")
                    print(f"YCrCb Upper: {optimal['ycrcb_upper']}")
                    print(f"HSV Lower: {optimal['hsv_lower']}")
                    print(f"HSV Upper: {optimal['hsv_upper']}")
                    print("=" * 70)
                    print("Applying to trackbars...")
                    
                    # Update trackbars
                    cv2.setTrackbarPos('YCrCb Y Min', 'Controls', int(optimal['ycrcb_lower'][0]))
                    cv2.setTrackbarPos('YCrCb Cr Min', 'Controls', int(optimal['ycrcb_lower'][1]))
                    cv2.setTrackbarPos('YCrCb Cb Min', 'Controls', int(optimal['ycrcb_lower'][2]))
                    cv2.setTrackbarPos('YCrCb Y Max', 'Controls', int(optimal['ycrcb_upper'][0]))
                    cv2.setTrackbarPos('YCrCb Cr Max', 'Controls', int(optimal['ycrcb_upper'][1]))
                    cv2.setTrackbarPos('YCrCb Cb Max', 'Controls', int(optimal['ycrcb_upper'][2]))
                    cv2.setTrackbarPos('HSV H Min', 'Controls', int(optimal['hsv_lower'][0]))
                    cv2.setTrackbarPos('HSV S Min', 'Controls', int(optimal['hsv_lower'][1]))
                    cv2.setTrackbarPos('HSV V Min', 'Controls', int(optimal['hsv_lower'][2]))
                    cv2.setTrackbarPos('HSV H Max', 'Controls', int(optimal['hsv_upper'][0]))
                    cv2.setTrackbarPos('HSV S Max', 'Controls', int(optimal['hsv_upper'][1]))
                    cv2.setTrackbarPos('HSV V Max', 'Controls', int(optimal['hsv_upper'][2]))
                else:
                    print("\nNot enough hand regions collected yet (need 10+)")
            elif key == ord('s'):
                self.save_parameters()
            elif key == ord('r'):
                self.stats = {
                    'mp_detections': 0,
                    'cv_detections': 0,
                    'matches': 0,
                    'misses': 0,
                    'false_positives': 0,
                    'total_frames': 0
                }
                self.hand_regions.clear()
                print("\nStatistics reset")
        
        cap.release()
        cv2.destroyAllWindows()
        self.mp_detector.cleanup()
        self.cv_detector.cleanup()
    
    def auto_optimize_presets(self, cap):
        """Test all presets and return the best one"""
        results = []
        test_frames = 50  # Test each preset for 50 frames
        
        for preset in self.presets:
            print(f"  Testing {preset['name']}...", end=' ')
            
            # Apply preset
            self.denoise_h = preset['denoise_h']
            self.kernel_small = preset['kernel_small']
            self.kernel_large = preset['kernel_large']
            self.morph_iter = preset['morph_iter']
            self.min_area = preset['min_area']
            
            # Test
            matches = 0
            mp_detections = 0
            
            for _ in range(test_frames):
                ret, frame = cap.read()
                if not ret:
                    break
                
                self.apply_cv_parameters()
                
                frame_mp = frame.copy()
                frame_cv = frame.copy()
                
                mp_result = self.mp_detector.process_frame(frame_mp)
                cv_result = self.cv_detector.process_frame(frame_cv)
                
                if mp_result['detected']:
                    mp_detections += 1
                    mp_bbox = self.get_hand_bbox_from_mediapipe(mp_result)
                    cv_bbox = self.get_cv_bbox(cv_result, frame.shape)
                    iou = self.compute_iou(mp_bbox, cv_bbox)
                    
                    if cv_result['detected'] and iou > 0.3:
                        matches += 1
            
            match_rate = (matches / max(1, mp_detections)) * 100
            results.append({
                'name': preset['name'],
                'match_rate': match_rate,
                **preset
            })
            print(f"Match rate: {match_rate:.1f}%")
        
        # Return best preset
        results.sort(key=lambda x: x['match_rate'], reverse=True)
        return results[0] if results else None
    
    def save_parameters(self):
        """Save current parameters to file"""
        params = {
            'timestamp': str(np.datetime64('now')),
            'ycrcb_lower': [int(self.ycrcb_y_min), int(self.ycrcb_cr_min), int(self.ycrcb_cb_min)],
            'ycrcb_upper': [int(self.ycrcb_y_max), int(self.ycrcb_cr_max), int(self.ycrcb_cb_max)],
            'hsv_lower': [int(self.hsv_h_min), int(self.hsv_s_min), int(self.hsv_v_min)],
            'hsv_upper': [int(self.hsv_h_max), int(self.hsv_s_max), int(self.hsv_v_max)],
            'denoise_h': int(self.denoise_h),
            'kernel_small': int(self.kernel_small),
            'kernel_large': int(self.kernel_large),
            'morph_iterations': int(self.morph_iter),
            'min_area': int(self.min_area),
            'statistics': self.stats
        }
        
        save_path = Path(__file__).parent.parent / 'calibration_mediapipe.json'
        with open(save_path, 'w') as f:
            json.dump(params, f, indent=2)
        
        print(f"\nParameters saved to {save_path}")
        print(f"Match rate: {(self.stats['matches'] / max(1, self.stats['mp_detections'])) * 100:.1f}%")

def main():
    calibrator = CVCalibratorWithMediaPipe()
    calibrator.run_calibration()

if __name__ == "__main__":
    main()
