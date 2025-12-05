"""UI display functions for calibration tool"""

import cv2
import numpy as np


def draw_progress_bar(frame, progress, message="Processing..."):
    """Draw a progress bar on frame with message"""
    h, w = frame.shape[:2]
    bar_y = int(h * 0.8)
    bar_x = int(w * 0.2)
    bar_width = int(w * 0.6)
    bar_height = 30
    
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (255, 255, 255), 2)
    fill_width = int(bar_width * progress)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height), (0, 255, 0), -1)
    
    percentage = f"{int(progress * 100)}%"
    cv2.putText(frame, percentage, (bar_x + bar_width + 10, bar_y + 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, message, (bar_x, bar_y - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)


def draw_roi_box(frame, roi_size=200):
    """Draw ROI box in center of frame"""
    h, w = frame.shape[:2]
    x = (w - roi_size) // 2
    y = (h - roi_size) // 2
    cv2.rectangle(frame, (x, y), (x + roi_size, y + roi_size), (0, 255, 0), 2)
    cv2.putText(frame, "Place hand here", (x, y - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return (x, y, roi_size, roi_size)


def show_masks_comparison(mask_ycrcb, mask_hsv, mask_combined):
    """Display three masks side by side for comparison"""
    masks = np.hstack([
        cv2.cvtColor(mask_ycrcb, cv2.COLOR_GRAY2BGR),
        cv2.cvtColor(mask_hsv, cv2.COLOR_GRAY2BGR),
        cv2.cvtColor(mask_combined, cv2.COLOR_GRAY2BGR)
    ])
    cv2.putText(masks, "YCrCb", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(masks, "HSV", (220, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(masks, "Combined", (430, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return masks


def print_header(title):
    """Print formatted header"""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def print_calibration_results(ycrcb_lower, ycrcb_upper, hsv_lower, hsv_upper, num_samples):
    """Print calibration results"""
    print(f"\nCalibration complete! ({num_samples} samples)")
    print(f"   YCrCb: {ycrcb_lower.tolist()} to {ycrcb_upper.tolist()}")
    print(f"   HSV: {hsv_lower.tolist()} to {hsv_upper.tolist()}")


def print_optimization_results(results, best_config):
    """Print optimization test results"""
    print("\n" + "=" * 70)
    print("OPTIMIZATION RESULTS")
    print("=" * 70)
    results.sort(key=lambda x: x['quality_score'], reverse=True)
    for idx, r in enumerate(results):
        marker = "★ BEST" if r['config']['name'] == best_config['name'] else ""
        print(f"{idx+1}. {r['config']['name']:12} | FPS: {r['fps']:5.1f} | Detection: {r['detection_rate']:5.1f}% | Score: {r['quality_score']:5.1f} {marker}")
    
    print("\n" + "=" * 70)
    print(f"WINNER: {best_config['name']}")
    print(f"   FPS: {best_config['fps']:.1f} | Detection: {best_config['detection_rate']:.1f}%")
    print("=" * 70)
    print(f"   Denoise: {best_config['denoise_h']}")
    print(f"   Kernel Small: {best_config['kernel_small']}x{best_config['kernel_small']}")
    print(f"   Kernel Large: {best_config['kernel_large']}x{best_config['kernel_large']}")
    print(f"   Morph Iterations: {best_config['morph_iter']}")
    print(f"   Min Area: {best_config['min_area']}")
    print("=" * 70)


def draw_performance_info(frame, avg_fps, total_time, hand_detected, show_debug=False, 
                         t_denoise=0, t_color=0, t_morph=0, t_contour=0):
    """Draw FPS and performance info on frame"""
    cv2.putText(frame, f"FPS: {int(avg_fps)}", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Processing: {total_time:.1f}ms", (10, 60),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, "Detected" if hand_detected else "Not detected", (10, 90),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if hand_detected else (0, 0, 255), 2)
    
    if show_debug:
        y_off = 120
        cv2.putText(frame, f"Denoise: {t_denoise:.1f}ms", (10, y_off), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"Color: {t_color:.1f}ms", (10, y_off + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"Morph: {t_morph:.1f}ms", (10, y_off + 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"Contour: {t_contour:.1f}ms", (10, y_off + 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Show bottleneck
        timings = [('Denoise', t_denoise), ('Color', t_color), ('Morph', t_morph), ('Contour', t_contour)]
        slowest = max(timings, key=lambda x: x[1])
        cv2.putText(frame, f"Bottleneck: {slowest[0]}", (10, y_off + 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
