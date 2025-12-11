"""
Analyze finger counting benchmark results and identify specific problems
"""

import json
import numpy as np
from pathlib import Path


def analyze_finger_benchmark(cv_file, mp_file):
    """Analyze both CV and MediaPipe results to identify problems"""
    
    with open(cv_file, 'r') as f:
        cv_data = json.load(f)
    
    with open(mp_file, 'r') as f:
        mp_data = json.load(f)
    
    print("\n" + "="*80)
    print("DETAILED FINGER COUNTING ANALYSIS")
    print("="*80)
    
    print(f"\nCV Accuracy:        {cv_data['overall_accuracy']:.1f}%")
    print(f"MediaPipe Accuracy: {mp_data['overall_accuracy']:.1f}%")
    print(f"Gap:                {mp_data['overall_accuracy'] - cv_data['overall_accuracy']:.1f}%")
    
    print("\n--- DETECTION PATTERNS ---")
    print(f"{'Expected':<12} {'CV Detects':<30} {'MP Detects':<30}")
    print("-" * 80)
    
    for finger in sorted([int(k) for k in cv_data['per_finger_results'].keys()]):
        cv_counts = cv_data['per_finger_results'][str(finger)]['detected_counts']
        mp_counts = mp_data['per_finger_results'][str(finger)]['detected_counts']
        
        # Calculate most common detection
        cv_mode = max(set(cv_counts), key=cv_counts.count) if cv_counts else -1
        mp_mode = max(set(mp_counts), key=mp_counts.count) if mp_counts else -1
        
        cv_mode_pct = (cv_counts.count(cv_mode) / len(cv_counts) * 100) if cv_counts else 0
        mp_mode_pct = (mp_counts.count(mp_mode) / len(mp_counts) * 100) if mp_counts else 0
        
        cv_mean = np.mean(cv_counts) if cv_counts else 0
        mp_mean = np.mean(mp_counts) if mp_counts else 0
        
        cv_summary = f"{cv_mode} ({cv_mode_pct:.0f}%), avg={cv_mean:.1f}"
        mp_summary = f"{mp_mode} ({mp_mode_pct:.0f}%), avg={mp_mean:.1f}"
        
        print(f"{finger} fingers: {cv_summary:<30} {mp_summary:<30}")
    
    print("\n--- CV SPECIFIC PROBLEMS ---")
    
    # Problem 1: Bias analysis
    all_cv_counts = []
    for finger_data in cv_data['per_finger_results'].values():
        all_cv_counts.extend(finger_data['detected_counts'])
    
    if all_cv_counts:
        cv_distribution = {i: all_cv_counts.count(i) for i in range(6)}
        total = len(all_cv_counts)
        print("\nCV Detection Distribution (all frames):")
        for i in sorted(cv_distribution.keys()):
            pct = cv_distribution[i] / total * 100
            bar = "█" * int(pct / 2)
            print(f"  {i} fingers: {cv_distribution[i]:>4} ({pct:>5.1f}%) {bar}")
        
        most_common = max(cv_distribution.items(), key=lambda x: x[1])
        print(f"\n❌ BIAS: CV detects {most_common[0]} fingers {most_common[1]/total*100:.1f}% of the time")
    
    # Problem 2: False negatives for specific counts
    print("\n--- WORST PERFORMERS ---")
    for finger in sorted([int(k) for k in cv_data['per_finger_results'].keys()]):
        cv_finger_data = cv_data['per_finger_results'][str(finger)]
        accuracy = (cv_finger_data['correct'] / cv_finger_data['samples'] * 100) if cv_finger_data['samples'] > 0 else 0
        
        if accuracy < 50:
            errors = cv_finger_data.get('errors', [])
            error_dist = {}
            for err in errors:
                error_dist[err] = error_dist.get(err, 0) + 1
            
            print(f"\n{finger} fingers: {accuracy:.1f}% accuracy")
            print(f"  Most often detected as:")
            for detected, count in sorted(error_dist.items(), key=lambda x: x[1], reverse=True)[:3]:
                pct = count / len(errors) * 100
                print(f"    {detected} fingers: {count} times ({pct:.1f}%)")
    
    # Problem 3: Variance analysis
    print("\n--- STABILITY ANALYSIS ---")
    for finger in sorted([int(k) for k in cv_data['per_finger_results'].keys()]):
        cv_counts = cv_data['per_finger_results'][str(finger)]['detected_counts']
        mp_counts = mp_data['per_finger_results'][str(finger)]['detected_counts']
        
        if cv_counts and mp_counts:
            cv_std = np.std(cv_counts)
            mp_std = np.std(mp_counts)
            
            if cv_std > 0.5:
                print(f"{finger} fingers: CV std={cv_std:.2f}, MP std={mp_std:.2f} - ❌ CV unstable")
    
    print("\n" + "="*80)
    print("RECOMMENDATIONS:")
    print("="*80)
    
    # Analyze bias
    if most_common[0] in [0, 1] and most_common[1]/total > 0.5:
        print("\n1. ❌ CRITICAL: Strong bias toward detecting 0-1 fingers")
        print("   Problem: Convexity defects not finding enough valleys")
        print("   Solutions:")
        print("   - Check if defects array is empty most of the time")
        print("   - Reduce defect depth threshold further")
        print("   - Rely more on extrema points and distance transform")
        print("   - Add fallback logic when convexity defects fail")
    
    # Check if 0 fingers incorrectly detected
    if '0' in cv_data['per_finger_results']:
        zero_finger_data = cv_data['per_finger_results']['0']
        if zero_finger_data['samples'] > 0 and zero_finger_data['correct'] / zero_finger_data['samples'] < 0.3:
            print("\n2. ❌ Cannot detect closed fist (0 fingers)")
            print("   Problem: Detecting 1 finger instead of 0")
            print("   Solution: Check palm area vs convexity hull ratio")
            print("   - If ratio > 0.95, likely a closed fist")
    
    # Check if 3-5 fingers incorrectly detected  
    high_finger_accuracies = []
    for finger in [3, 4, 5]:
        finger_key = str(finger)
        if finger_key in cv_data['per_finger_results']:
            data = cv_data['per_finger_results'][finger_key]
            if data['samples'] > 0:
                acc = data['correct'] / data['samples']
                high_finger_accuracies.append(acc)
    
    if high_finger_accuracies and np.mean(high_finger_accuracies) < 0.2:
        print("\n3. ❌ Cannot detect 3-5 fingers (open hand)")
        print("   Problem: Open hand detected as 1-2 fingers")
        print("   Solutions:")
        print("   - Check if extrema points method is finding multiple fingertips")
        print("   - Increase weight of distance transform method")
        print("   - Use range detection: if hand area large + multiple peaks = 5 fingers")


def main():
    benchmarks_dir = Path('benchmarks')
    
    # Find latest parallel comparison
    cv_files = sorted(benchmarks_dir.glob('finger_accuracy_cv_*.json'))
    mp_files = sorted(benchmarks_dir.glob('finger_accuracy_mediapipe_*.json'))
    
    if not cv_files or not mp_files:
        print("❌ No benchmark files found")
        return
    
    # Use latest
    cv_file = cv_files[-1]
    mp_file = mp_files[-1]
    
    print(f"\nAnalyzing:")
    print(f"  CV: {cv_file.name}")
    print(f"  MP: {mp_file.name}")
    
    analyze_finger_benchmark(cv_file, mp_file)


if __name__ == '__main__':
    main()
