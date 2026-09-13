"""Compare the revised implementation with the preserved public baseline.

Usage: python implementation_checks.py --repo ../video2xy --out results
All fixtures are procedural. Timings exclude decoding and output devices.
"""
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time
import wave

import cv2
import numpy as np

p = argparse.ArgumentParser()
p.add_argument('--repo', type=Path, required=True)
p.add_argument('--out', type=Path, required=True)
a = p.parse_args()
a.repo = a.repo.resolve()
a.out.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(a.repo))
import video_to_xy_audio as revised

baseline_path = Path(__file__).parent/'baseline'/'video_to_xy_audio.py'
spec = importlib.util.spec_from_file_location('video2xy_baseline', baseline_path)
baseline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(baseline)
record = {
    'baseline_commit': '6a7d1dd8be7622966f6e4d7e5f2dfbdec1001af6',
    'revised_commit': subprocess.check_output(
        ['git', '-C', str(a.repo), 'rev-parse', 'HEAD'], text=True).strip(),
    'source_sha256': {name: hashlib.sha256(path.read_bytes()).hexdigest()
                      for name, path in [('baseline', baseline_path),
                                         ('revised', Path(revised.__file__))]},
    'python': platform.python_version(), 'numpy': np.__version__,
    'opencv': cv2.__version__, 'platform': platform.platform(),
    'processor': platform.processor(),
}

def compare(cases):
    result = {'cases': 0, 'mismatches': 0, 'maximum_absolute_difference': 0.0}
    for points, n in cases:
        old = baseline.resample_closed_contour(points, n)
        new = revised.resample_closed_contour(points, n)
        result['cases'] += 1
        result['mismatches'] += int(not np.array_equal(old, new))
        result['maximum_absolute_difference'] = max(result['maximum_absolute_difference'],
            float(np.max(np.abs(old.astype(np.float64)-new.astype(np.float64)))))
    assert result['mismatches'] == 0
    return result

rng = np.random.default_rng(20260914)
def random_cases():
    for _ in range(3000):
        points = rng.uniform(0, 640, (int(rng.integers(3, 65)), 2)).astype(np.float32)
        if rng.integers(0, 5) == 0:
            points = np.insert(points, 1, points[0], axis=0)
        yield points, int(rng.integers(2, 513))
record['random_resampling'] = {'seed': 20260914, **compare(random_cases())}
cases = []
for width in (1, 10, 100, 1000):
    for height in (0, 1e-8, 1e-6, 1e-4, 1):
        rect = np.array([[0, 0], [width, 0], [width, height], [0, height]], np.float32)
        for points in (rect, rect[::-1], np.insert(rect, 1, rect[0], axis=0)):
            for n in (2, 3, 5, 33, 735):
                cases.append((points, n))
record['structured_resampling'] = compare(cases)
closure_points = np.array([[297, 326], [382, 129], [437, 47], [198, 366],
                           [150, 210], [263, 429], [210, 444], [464, 122]], np.float32)
t = revised.resample_closed_contour(closure_points, 33)
record['closure_residual_px'] = float(np.max(np.abs(t[-1]-t[0])))

def rejected(call):
    try:
        call()
    except ValueError as error:
        return {'rejected': True, 'message': str(error)}
    raise AssertionError('Expected ValueError')

rect = np.array([[20, 20], [100, 20], [100, 80], [20, 80]], np.float32)
old_trace = baseline.build_trace([rect], 320, 240, 16, 1.25, .8, 20, 8)
record['budget_regressions'] = {
    'baseline_infeasible_counts': baseline.allocate_samples(np.array([10., 20.]), 3, 8).tolist(),
    'revised_infeasible': rejected(lambda: revised.allocate_samples(np.array([10., 20.]), 3, 8)),
    'baseline_zero_length_counts': baseline.allocate_samples(np.zeros(2), 10, 2).tolist(),
    'revised_zero_length_counts': revised.allocate_samples(np.zeros(2), 10, 2).tolist(),
    'baseline_collapse_unique_pairs': len(np.unique(old_trace, axis=0)),
    'revised_collapse_case': rejected(lambda: revised.build_trace([rect], 320, 240, 16, 1.25, .8, 20, 8)),
}
old_clock = baseline.FrameSampleClock(8000, 20000)
counts = [old_clock.next_count() for _ in range(5)]
record['clock_regression'] = {'baseline_counts': counts, 'baseline_counter': old_clock.written_total,
    'ideal_total': 2, 'revised': rejected(lambda: revised.FrameSampleClock(8000, 20000))}
args = revised.build_argparser().parse_args(['0', '--play'])
record['camera_default_output'] = revised.resolve_output(args)
assert record['camera_default_output'] is None

# Full command-line export at a rate that is not aligned to scan boundaries.
with tempfile.TemporaryDirectory(prefix='video2xy-revised-checks-') as folder:
    folder = Path(folder)
    source = folder/'static.avi'
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*'MJPG'), 25, (160, 120))
    assert writer.isOpened()
    frame = np.zeros((120, 160, 3), np.uint8)
    cv2.rectangle(frame, (20, 25), (120, 95), (255, 255, 255), 2)
    for _ in range(6):
        writer.write(frame)
    writer.release()
    checks = {}
    for name, module in [('baseline', baseline), ('revised', revised)]:
        for tracking in (True, False):
            target = folder/(name+'.wav')
            command = [sys.executable, module.__file__, str(source), '-o', str(target),
                       '--fps', '25', '--max-contours', '1', '--epsilon', '0.02']
            if not tracking:
                command.append('--no-tracking')
            run = subprocess.run(command, capture_output=True, text=True, timeout=30)
            assert run.returncode == 0, run.stderr
            with wave.open(str(target), 'rb') as wav:
                n = wav.getnframes()
                samples = np.frombuffer(wav.readframes(n), dtype='<i2').reshape(-1, 2)
            expected = np.tile(samples[:735], (15, 1))[:n]
            mismatches = int(np.count_nonzero(np.any(samples != expected, axis=1)))
            checks[f'{name}_tracking_{tracking}'] = {'sample_pairs': n,
                'mismatching_pairs_against_continuous_repetition': mismatches,
                'nonzero': bool(np.any(samples)), 'pcm_sha256': hashlib.sha256(samples.tobytes()).hexdigest()}
            assert n == 10584 and np.any(samples)
            assert (mismatches == 0) if name == 'revised' else (mismatches > 0)
        target = folder/(name+'-invalid.wav')
        run = subprocess.run([sys.executable, module.__file__, str(source), '-o', str(target),
                              '--sobel-ksize', '2'], capture_output=True, text=True, timeout=30)
        checks[name+'_invalid_sobel'] = {'exit_code': run.returncode,
            'output_exists': target.exists(), 'output_bytes': target.stat().st_size if target.exists() else 0,
            'traceback': 'Traceback' in run.stderr, 'usage_error': 'usage:' in run.stderr}
        (a.out/(name+'_invalid_sobel_stderr.txt')).write_text(run.stderr, encoding='utf-8')
    record['export_checks'] = checks

# Ten separate rectangles yield twenty retained edge contours. Fix OpenCV's
# thread count to reduce one source of timing variation, and report it.
cv2.setNumThreads(1)
frame = np.zeros((480, 640, 3), np.uint8)
for row in range(2):
    for col in range(5):
        x = 20+col*120; y = 30+row*210
        cv2.rectangle(frame, (x, y), (x+70, y+80), (255, 255, 255), -1)
clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
_, edges = revised.preprocess_edges(frame, 480, 1.5, 5, 3, 72, 35, 110, 1, clahe_operator=clahe)
contours = revised.extract_contours(edges, 18, 32, .004)
tracker = revised.ContourTracker(); tracker.update(contours, 480, 360)
calls = Counter(); stage = 'tracker'; resample = revised.resample_closed_contour
def counted(points, n):
    calls[stage] += 1
    return resample(points, n)
try:
    revised.resample_closed_contour = counted
    ordered = tracker.update(contours, 480, 360)
    stage = 'build_trace'
    revised.build_trace(ordered, 480, 360, 735, 1.25, .8, 1, 8, preserve_order=True)
finally:
    revised.resample_closed_contour = resample
record['calls_per_matched_frame'] = {'retained_contours': len(contours), **dict(calls)}

def run_stages(module, tracker, cached):
    times = [time.perf_counter_ns()]
    _, edges = module.preprocess_edges(frame, 480, 1.5, 5, 3, 72, 35, 110, 1,
                                      **({'clahe_operator': clahe} if cached else {}))
    times.append(time.perf_counter_ns())
    cs = module.extract_contours(edges, 18, 32, .004); times.append(time.perf_counter_ns())
    cs = tracker.update(cs, 480, 360); times.append(time.perf_counter_ns())
    trace = module.build_trace(cs, 480, 360, 735, 1.25, .8, 1, 8, preserve_order=True)
    times.append(time.perf_counter_ns())
    return trace, (np.diff(times)/1e6).tolist()

modules = {'baseline': baseline, 'revised': revised}
trackers = {name: module.ContourTracker() for name, module in modules.items()}
timings = {name: [] for name in modules}; equality = []
for i in range(110):
    outputs = {}
    for name in (list(modules) if i % 2 == 0 else list(reversed(modules))):
        trace, durations = run_stages(modules[name], trackers[name], name == 'revised')
        outputs[name] = trace
        if i >= 10:
            timings[name].append(durations)
    equality.append(bool(np.array_equal(outputs['baseline'], outputs['revised'])))
assert all(equality)
record['kernel_timing'] = {
    'fixture': '10 nonoverlapping filled rectangles in a 640x480 image, resized to width 480',
    'warmup_frames_per_arm': 10, 'measured_frames_per_arm': 100,
    'opencv_threads': cv2.getNumThreads(),
    'stage_order': ['preprocess_edges', 'extract_contours', 'tracker_update', 'build_trace'],
    'trace_equal_all_110_pairs': all(equality), 'raw_stage_ms': timings,
    'summary_ms': {name: {'mean_stages': np.mean(values, axis=0).tolist(),
                          'mean_total': float(np.mean(np.sum(values, axis=1))),
                          'total_q25_median_q75': np.quantile(np.sum(values, axis=1), [.25, .5, .75]).tolist()}
                   for name, values in timings.items()},
}
(a.out/'implementation_checks.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
summary = dict(record)
summary['kernel_timing'] = {k: v for k, v in record['kernel_timing'].items() if k != 'raw_stage_ms'}
print(json.dumps(summary, indent=2))
