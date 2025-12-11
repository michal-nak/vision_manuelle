# Performance Improvements Applied

**Date:** December 7, 2025  
**Based on:** benchmark_results_20251207_105848.json  
**Tool:** analyze_benchmark.py

## Problems Identified

### Critical Issues (from benchmark analysis):
1. **Detection Rate: 21.7%** (vs MediaPipe 100%)
   - 78.3% of frames failed to detect hand
   
2. **FPS: 2.3** (vs MediaPipe 37.9)
   - 16x slower than baseline
   - Unusable for real-time interaction
   
3. **Latency: 432ms** (vs MediaPipe 27ms)
   - 405ms additional delay per frame

## Applied Improvements (Priority 1)

### 1. Relaxed Area Constraints
**File:** `src/core/config.py`
```python
# Before
MAX_HAND_AREA = 0.5  # 50% of frame

# After
MAX_HAND_AREA = 0.7  # 70% of frame
```
**Expected:** +20-30% detection rate (allows closer hands)

### 2. Reduced Minimum Area
**File:** `src/core/config.py`
```python
# Before
MIN_HAND_AREA = 3000

# After  
MIN_HAND_AREA = 2000
```
**Expected:** +10-15% detection rate (detects smaller/distant hands)

### 3. Disabled Denoising (by default)
**File:** `src/detectors/cv/skin_detection.py`
```python
# Before
denoised = cv2.fastNlMeansDenoisingColored(frame, None, denoise_h, denoise_h, 7, 21)

# After
if enable_denoising:  # Default: False
    denoised = cv2.fastNlMeansDenoisingColored(frame, None, denoise_h, denoise_h, 7, 21)
else:
    denoised = frame
```
**Expected:** +200-300% FPS (3-4x speed), -250ms latency

### 4. Reduced Morphology Iterations
**File:** `src/detectors/cv/config_loader.py`
```python
# Before
'morph_iterations': 2

# After
'morph_iterations': 1
```
**Expected:** +50-100% FPS (2x speed)

## Expected Combined Impact

### Detection Rate
- Before: 21.7%
- Expected: 52-67% (+30-45%)
- Target: Closer to MediaPipe's 100%

### FPS Performance
- Before: 2.3 FPS
- Expected: 9-16 FPS (+250-600%)
- Target: 30+ FPS for real-time use

### Latency
- Before: 432ms
- Expected: 150-200ms (-250-280ms)
- Target: <100ms for responsive interaction

## Next Steps (Priority 2 & 3)

### Still To Implement:
1. **Optimize color calibration** (margin_factor: 1.5 → 2.0)
   - Expected: +15-25% detection in varied lighting
   
2. **Disable background subtraction initially** (first 10 frames)
   - Expected: +10-20% detection rate
   
3. **Optimize finger detection** (alternate methods per frame)
   - Expected: +30-50% FPS
   
4. **Resolution optimization** (320x240 processing)
   - Expected: +100-150% FPS (2-2.5x)

## Validation

Run new benchmark to measure actual improvements:
```bash
python tools/benchmark_comparison.py -s  # Skip calibration
# Select option 2 (Quick test - 20s)
```

Then compare results:
```bash
python tools/analyze_benchmark.py --latest
```

## Rollback

If issues occur, revert changes:
```python
# config.py
MAX_HAND_AREA = 0.5
MIN_HAND_AREA = 3000

# config_loader.py
'morph_iterations': 2

# skin_detection.py
# Remove enable_denoising parameter, always denoise
```
