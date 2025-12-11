"""
Compare finger counting benchmark results
Shows improvement between baseline and optimized versions
"""

import json
import sys
import os
from pathlib import Path


def load_benchmark(filepath):
    """Load benchmark JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)


def analyze_benchmark(data):
    """Extract key metrics from benchmark data"""
    overall_acc = data['overall_accuracy']
    total_samples = data['total_samples']
    total_correct = data['total_correct']
    
    per_finger = {}
    for finger, results in data['per_finger_results'].items():
        samples = results['samples']
        correct = results['correct']
        accuracy = (correct / samples * 100) if samples > 0 else 0
        
        # Calculate most common error
        errors = results.get('errors', [])
        error_dist = {}
        for err in errors:
            error_dist[err] = error_dist.get(err, 0) + 1
        most_common_error = max(error_dist.items(), key=lambda x: x[1])[0] if error_dist else None
        
        per_finger[int(finger)] = {
            'samples': samples,
            'correct': correct,
            'accuracy': accuracy,
            'most_common_error': most_common_error
        }
    
    return {
        'overall_accuracy': overall_acc,
        'total_samples': total_samples,
        'total_correct': total_correct,
        'per_finger': per_finger
    }


def compare_benchmarks(baseline_file, improved_file):
    """Compare two benchmark results"""
    baseline = load_benchmark(baseline_file)
    improved = load_benchmark(improved_file)
    
    baseline_metrics = analyze_benchmark(baseline)
    improved_metrics = analyze_benchmark(improved)
    
    print("\n" + "="*80)
    print("FINGER COUNTING PERFORMANCE COMPARISON")
    print("="*80)
    print(f"Baseline: {Path(baseline_file).name}")
    print(f"Improved: {Path(improved_file).name}")
    print("="*80)
    
    # Overall comparison
    print("\n--- OVERALL ACCURACY ---")
    baseline_acc = baseline_metrics['overall_accuracy']
    improved_acc = improved_metrics['overall_accuracy']
    improvement = improved_acc - baseline_acc
    improvement_pct = (improvement / baseline_acc * 100) if baseline_acc > 0 else 0
    
    print(f"Baseline:    {baseline_acc:.1f}%")
    print(f"Improved:    {improved_acc:.1f}%")
    print(f"Change:      {improvement:+.1f}% ({improvement_pct:+.1f}%)")
    
    if improvement > 0:
        print("✅ IMPROVEMENT")
    elif improvement < 0:
        print("❌ REGRESSION")
    else:
        print("⚠️  NO CHANGE")
    
    # Per-finger comparison
    print("\n--- PER-FINGER ACCURACY COMPARISON ---")
    print(f"{'Fingers':<10} {'Baseline':<12} {'Improved':<12} {'Change':<12} {'Status'}")
    print("-" * 80)
    
    for finger in sorted(baseline_metrics['per_finger'].keys()):
        baseline_data = baseline_metrics['per_finger'][finger]
        improved_data = improved_metrics['per_finger'].get(finger, {'accuracy': 0})
        
        baseline_acc = baseline_data['accuracy']
        improved_acc = improved_data['accuracy']
        change = improved_acc - baseline_acc
        
        status = "✅" if change > 5 else ("❌" if change < -5 else "→")
        
        print(f"{finger:<10} {baseline_acc:>6.1f}%{'':<5} {improved_acc:>6.1f}%{'':<5} {change:>+6.1f}%{'':<5} {status}")
    
    # Most problematic fingers
    print("\n--- MOST PROBLEMATIC FINGERS (IMPROVED) ---")
    problematic = [(f, d['accuracy']) for f, d in improved_metrics['per_finger'].items() if d['accuracy'] < 50]
    problematic.sort(key=lambda x: x[1])
    
    if problematic:
        for finger, acc in problematic[:3]:
            error = improved_metrics['per_finger'][finger]['most_common_error']
            print(f"  {finger} fingers: {acc:.1f}% (often detected as {error})")
    else:
        print("  None! All fingers >50% accuracy")
    
    # Best improvements
    print("\n--- BEST IMPROVEMENTS ---")
    improvements = []
    for finger in baseline_metrics['per_finger'].keys():
        baseline_acc = baseline_metrics['per_finger'][finger]['accuracy']
        improved_acc = improved_metrics['per_finger'].get(finger, {'accuracy': 0})['accuracy']
        change = improved_acc - baseline_acc
        if change > 0:
            improvements.append((finger, change))
    
    improvements.sort(key=lambda x: x[1], reverse=True)
    
    if improvements:
        for finger, change in improvements[:3]:
            print(f"  {finger} fingers: +{change:.1f}%")
    else:
        print("  No improvements detected")
    
    print("\n" + "="*80)


def main():
    benchmarks_dir = Path('benchmarks')
    
    # Find all finger accuracy files
    files = sorted(benchmarks_dir.glob('finger_accuracy_cv_*.json'))
    
    if len(files) < 2:
        print(f"❌ Need at least 2 benchmark files to compare. Found {len(files)}")
        return
    
    print(f"\n📊 Found {len(files)} benchmark files:")
    for i, f in enumerate(files):
        print(f"  {i+1}. {f.name}")
    
    # Compare most recent two
    baseline = files[-2]
    improved = files[-1]
    
    print(f"\n🔍 Comparing:")
    print(f"  Baseline: {baseline.name}")
    print(f"  Improved: {improved.name}")
    
    compare_benchmarks(baseline, improved)


if __name__ == '__main__':
    main()
