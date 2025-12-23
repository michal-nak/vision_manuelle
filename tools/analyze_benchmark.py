"""
Benchmark Analysis and Improvement Tool
Analyzes benchmark results and suggests/applies targeted improvements
"""
import json
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))


class BenchmarkAnalyzer:
    """Analyzes benchmark results and identifies performance issues"""
    
    def __init__(self, benchmark_file):
        self.benchmark_file = Path(benchmark_file)
        self.data = self._load_benchmark()
        self.issues = []
        self.recommendations = []
        
    def _load_benchmark(self):
        """Load benchmark JSON data"""
        with open(self.benchmark_file, 'r') as f:
            return json.load(f)
    
    def analyze(self):
        """Run complete analysis"""
        print("=" * 80)
        print("BENCHMARK ANALYSIS")
        print("=" * 80)
        print(f"File: {self.benchmark_file.name}")
        print(f"Date: {datetime.fromtimestamp(self.benchmark_file.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)
        
        cv_data = self.data.get('CV', {})
        mp_data = self.data.get('MediaPipe', {})
        
        # Analysis categories
        self._analyze_detection_rate(cv_data, mp_data)
        self._analyze_fps_performance(cv_data, mp_data)
        self._analyze_stability(cv_data, mp_data)
        self._analyze_latency(cv_data, mp_data)
        
        # Generate report
        self._generate_report()
        
        return self.recommendations
    
    def _analyze_detection_rate(self, cv, mp):
        """Analyze detection success rate"""
        cv_rate = cv.get('detection_rate', 0)
        mp_rate = mp.get('detection_rate', 0)
        
        print("\n--- DETECTION RATE ANALYSIS ---")
        print(f"CV:        {cv_rate:.1f}%")
        print(f"MediaPipe: {mp_rate:.1f}%")
        print(f"Gap:       {mp_rate - cv_rate:.1f}%")
        
        if cv_rate < 50:
            severity = "CRITICAL"
            self.issues.append({
                'category': 'detection_rate',
                'severity': severity,
                'description': f"CV detection rate is critically low ({cv_rate:.1f}%)",
                'impact': 'High - Most frames fail to detect hand'
            })
            
            # Recommendations
            if cv_rate < 30:
                self.recommendations.append({
                    'priority': 1,
                    'category': 'detection_rate',
                    'action': 'relax_area_constraints',
                    'description': 'Increase MAX_HAND_AREA to allow larger/closer hands',
                    'implementation': 'config.MAX_HAND_AREA: 0.5 → 0.7',
                    'expected_improvement': '+20-30% detection rate'
                })
                
                self.recommendations.append({
                    'priority': 1,
                    'category': 'detection_rate',
                    'action': 'reduce_min_area',
                    'description': 'Lower MIN_HAND_AREA to detect smaller/distant hands',
                    'implementation': 'config.MIN_HAND_AREA: 3000 → 2000',
                    'expected_improvement': '+10-15% detection rate'
                })
            
            self.recommendations.append({
                'priority': 2,
                'category': 'detection_rate',
                'action': 'optimize_color_calibration',
                'description': 'Recalibrate with more generous margins',
                'implementation': 'Increase margin_factor in calibration: 1.5 → 2.0',
                'expected_improvement': '+15-25% detection rate in varied lighting'
            })
            
            self.recommendations.append({
                'priority': 2,
                'category': 'detection_rate',
                'action': 'disable_bg_subtraction_initially',
                'description': 'Background subtraction may be filtering valid hands',
                'implementation': 'Disable for first 10 frames, let user move hand',
                'expected_improvement': '+10-20% detection rate'
            })
    
    def _analyze_fps_performance(self, cv, mp):
        """Analyze FPS performance"""
        cv_fps = cv.get('fps', {}).get('mean', 0)
        mp_fps = mp.get('fps', {}).get('mean', 0)
        
        print("\n--- FPS PERFORMANCE ANALYSIS ---")
        print(f"CV:        {cv_fps:.1f} FPS")
        print(f"MediaPipe: {mp_fps:.1f} FPS")
        print(f"Ratio:     {cv_fps/mp_fps if mp_fps > 0 else 0:.2f}x")
        
        if cv_fps < 15:
            severity = "CRITICAL"
            self.issues.append({
                'category': 'fps',
                'severity': severity,
                'description': f"CV FPS is critically low ({cv_fps:.1f} FPS)",
                'impact': 'High - Unusable for real-time interaction'
            })
            
            self.recommendations.append({
                'priority': 1,
                'category': 'fps',
                'action': 'reduce_morphology_iterations',
                'description': 'Morphological operations are expensive',
                'implementation': 'Reduce morph_iterations: 3 → 2 (closing), 2 → 1 (opening)',
                'expected_improvement': '+50-100% FPS (double speed)'
            })
            
            self.recommendations.append({
                'priority': 1,
                'category': 'fps',
                'action': 'disable_denoising',
                'description': 'Denoising (fastNlMeansDenoisingColored) is very slow',
                'implementation': 'Remove denoising or make optional flag',
                'expected_improvement': '+200-300% FPS (3-4x speed)'
            })
            
            self.recommendations.append({
                'priority': 2,
                'category': 'fps',
                'action': 'optimize_finger_detection',
                'description': 'Running 3 methods every frame is expensive',
                'implementation': 'Use single method or alternate methods per frame',
                'expected_improvement': '+30-50% FPS'
            })
            
            self.recommendations.append({
                'priority': 3,
                'category': 'fps',
                'action': 'reduce_resolution',
                'description': 'Process at lower resolution, upscale results',
                'implementation': 'Resize to 320x240, process, scale back',
                'expected_improvement': '+100-150% FPS (2-2.5x speed)'
            })
    
    def _analyze_stability(self, cv, mp):
        """Analyze finger count stability"""
        cv_stab = cv.get('finger_stability', 0)
        mp_stab = mp.get('finger_stability', 0)
        
        print("\n--- STABILITY ANALYSIS ---")
        print(f"CV Variance:        {cv_stab:.4f}")
        print(f"MediaPipe Variance: {mp_stab:.4f}")
        print(f"Comparison:         {'CV more stable' if cv_stab < mp_stab else 'MediaPipe more stable'}")
        
        if cv_stab > 0.5:
            self.issues.append({
                'category': 'stability',
                'severity': 'MODERATE',
                'description': f"High variance in finger counting ({cv_stab:.3f})",
                'impact': 'Medium - Jittery gesture recognition'
            })
            
            self.recommendations.append({
                'priority': 3,
                'category': 'stability',
                'action': 'increase_temporal_smoothing',
                'description': 'Increase smoothing window',
                'implementation': 'FINGER_COUNT_SMOOTHING: 3 → 5 frames',
                'expected_improvement': '-30-50% variance'
            })
    
    def _analyze_latency(self, cv, mp):
        """Analyze detection latency"""
        cv_lat = cv.get('latency_ms', {}).get('mean', 0)
        mp_lat = mp.get('latency_ms', {}).get('mean', 0)
        
        print("\n--- LATENCY ANALYSIS ---")
        print(f"CV:        {cv_lat:.1f} ms")
        print(f"MediaPipe: {mp_lat:.1f} ms")
        print(f"Difference: +{cv_lat - mp_lat:.1f} ms")
        
        if cv_lat > 100:
            self.issues.append({
                'category': 'latency',
                'severity': 'HIGH',
                'description': f"High processing latency ({cv_lat:.0f}ms)",
                'impact': 'High - Noticeable lag in interactions'
            })
            
            # Latency improvements overlap with FPS improvements
            if not any(r['action'] == 'disable_denoising' for r in self.recommendations):
                self.recommendations.append({
                    'priority': 1,
                    'category': 'latency',
                    'action': 'disable_denoising',
                    'description': 'Denoising adds 200-300ms per frame',
                    'implementation': 'Remove cv2.fastNlMeansDenoisingColored',
                    'expected_improvement': '-250ms latency'
                })
    
    def _generate_report(self):
        """Generate summary report"""
        print("\n" + "=" * 80)
        print("ISSUES SUMMARY")
        print("=" * 80)
        
        if not self.issues:
            print("No critical issues detected!")
        else:
            for i, issue in enumerate(self.issues, 1):
                print(f"\n{i}. [{issue['severity']}] {issue['description']}")
                print(f"   Impact: {issue['impact']}")
        
        print("\n" + "=" * 80)
        print("RECOMMENDATIONS (Priority Order)")
        print("=" * 80)
        
        if not self.recommendations:
            print("No specific recommendations at this time.")
        else:
            # Sort by priority
            sorted_recs = sorted(self.recommendations, key=lambda x: x['priority'])
            
            for i, rec in enumerate(sorted_recs, 1):
                print(f"\n{i}. [Priority {rec['priority']}] {rec['description']}")
                print(f"   Action: {rec['action']}")
                print(f"   Implementation: {rec['implementation']}")
                print(f"   Expected: {rec['expected_improvement']}")
        
        print("\n" + "=" * 80)


class ImprovementApplier:
    """Applies recommended improvements to codebase"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.config_file = self.project_root / 'src' / 'core' / 'config.py'
        self.applied_changes = []
    
    def apply_recommendation(self, recommendation):
        """Apply a specific recommendation"""
        action = recommendation['action']
        
        handlers = {
            'relax_area_constraints': self._apply_relax_area,
            'reduce_min_area': self._apply_reduce_min_area,
            'disable_denoising': self._apply_disable_denoising,
            'reduce_morphology_iterations': self._apply_reduce_morphology,
            'increase_temporal_smoothing': self._apply_increase_smoothing,
            'optimize_color_calibration': self._apply_optimize_calibration
        }
        
        handler = handlers.get(action)
        if handler:
            return handler(recommendation)
        else:
            print(f"No handler for action: {action}")
            return False
    
    def _apply_relax_area(self, rec):
        """Increase MAX_HAND_AREA"""
        # This would modify config.py
        print(f"Would apply: {rec['implementation']}")
        self.applied_changes.append(rec)
        return True
    
    def _apply_reduce_min_area(self, rec):
        """Decrease MIN_HAND_AREA"""
        print(f" Would apply: {rec['implementation']}")
        self.applied_changes.append(rec)
        return True
    
    def _apply_disable_denoising(self, rec):
        """Disable denoising step"""
        print(f"Would apply: {rec['implementation']}")
        self.applied_changes.append(rec)
        return True
    
    def _apply_reduce_morphology(self, rec):
        """Reduce morphology iterations"""
        print(f"Would apply: {rec['implementation']}")
        self.applied_changes.append(rec)
        return True
    
    def _apply_increase_smoothing(self, rec):
        """Increase temporal smoothing"""
        print(f"Would apply: {rec['implementation']}")
        self.applied_changes.append(rec)
        return True
    
    def _apply_optimize_calibration(self, rec):
        """Optimize calibration parameters"""
        print(f"Would apply: {rec['implementation']}")
        self.applied_changes.append(rec)
        return True


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze benchmark results and suggest improvements')
    parser.add_argument('benchmark_file', nargs='?', help='Path to benchmark JSON file')
    parser.add_argument('--latest', action='store_true', help='Use latest benchmark file')
    parser.add_argument('--apply', action='store_true', help='Apply recommended improvements (interactive)')
    
    args = parser.parse_args()
    
    # Find benchmark file
    if args.latest or not args.benchmark_file:
        benchmarks_dir = Path(__file__).parent.parent / 'benchmarks'
        json_files = list(benchmarks_dir.glob('benchmark_results_*.json'))
        
        if not json_files:
            print("No benchmark files found in benchmarks/")
            print("Run: python tools/benchmark_comparison.py")
            return
        
        benchmark_file = max(json_files, key=lambda p: p.stat().st_mtime)
        print(f"Using latest benchmark: {benchmark_file.name}\n")
    else:
        benchmark_file = Path(args.benchmark_file)
        if not benchmark_file.exists():
            print(f"File not found: {benchmark_file}")
            return
    
    # Analyze
    analyzer = BenchmarkAnalyzer(benchmark_file)
    recommendations = analyzer.analyze()
    
    # Apply improvements if requested
    if args.apply and recommendations:
        print("\n" + "=" * 80)
        print("APPLY IMPROVEMENTS")
        print("=" * 80)
        
        applier = ImprovementApplier()
        
        print("\nWhich improvements would you like to apply?")
        print("1. All Priority 1 (Critical)")
        print("2. All Priority 1 & 2")
        print("3. Select individually")
        print("4. Cancel")
        
        choice = input("\nChoice (1-4): ").strip()
        
        if choice == '1':
            to_apply = [r for r in recommendations if r['priority'] == 1]
        elif choice == '2':
            to_apply = [r for r in recommendations if r['priority'] <= 2]
        elif choice == '3':
            to_apply = []
            for i, rec in enumerate(recommendations, 1):
                response = input(f"\nApply: {rec['description']}? (y/n): ").strip().lower()
                if response == 'y':
                    to_apply.append(rec)
        else:
            print("Cancelled.")
            return
        
        if to_apply:
            print(f"\nApplying {len(to_apply)} improvements...")
            for rec in to_apply:
                applier.apply_recommendation(rec)
            
            print(f"\n{len(applier.applied_changes)} changes applied!")
            print("Note: This is a dry-run. Implement changes manually or extend the applier.")
        else:
            print("No improvements selected.")


if __name__ == "__main__":
    main()
