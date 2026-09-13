"""Numerical, scheduling, CLI, and export regressions; no hardware required."""
from contextlib import redirect_stderr
from fractions import Fraction
import io
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import wave

import cv2
import numpy as np

import video_to_xy_audio as v


def rectangle(width=100, height=60):
    return np.array([[20, 20], [20+width, 20],
                     [20+width, 20+height], [20, 20+height]], np.float32)


def scalar_reference(points, n):
    """Original 6a7d1dd resampler, retained as a regression oracle only."""
    p = np.vstack([points, points[0]]).astype(np.float32)
    seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
    total = float(seg.sum())
    if total <= 1e-12:
        return np.repeat(p[:1], n, axis=0)
    cumulative = np.concatenate([[0.0], np.cumsum(seg)])
    targets = np.linspace(0.0, total, n, endpoint=True)
    out = np.empty((n, 2), dtype=np.float32)
    idx = 0
    for k, t in enumerate(targets):
        while idx+1 < len(cumulative)-1 and cumulative[idx+1] < t:
            idx += 1
        denom = cumulative[idx+1] - cumulative[idx]
        u = 0.0 if denom <= 1e-12 else (t-cumulative[idx])/denom
        out[k] = p[idx]*(1.0-u) + p[idx+1]*u
    return out


class NumericalTests(unittest.TestCase):
    def test_frame_clock_prefix_duration(self):
        for fps in (Fraction(24), Fraction(25), Fraction(30), Fraction(30000, 1001)):
            with self.subTest(fps=fps):
                clock = v.FrameSampleClock(44100, float(fps))
                total = 0
                for k in range(1, 5001):
                    n = clock.next_count()
                    self.assertGreaterEqual(n, 1)
                    total += n
                    self.assertLessEqual(abs(Fraction(total)-k*44100/fps), Fraction(1, 2))
                    self.assertEqual(total, clock.written_total)

    def test_frame_clock_rejects_unsupported_rates(self):
        for fps in (0, -1, 20000, 1e-320, float('nan'), float('inf')):
            with self.subTest(fps=fps), self.assertRaises(ValueError):
                v.FrameSampleClock(8000, fps)
        with self.assertRaises(ValueError):
            v.FrameSampleClock(0, 30)
        clock = v.FrameSampleClock(8000, 8000)
        self.assertEqual([clock.next_count() for _ in range(10)], [1]*10)

    def test_allocation_exact_sum_and_hard_minimum(self):
        for m in range(1, 17):
            for minimum in (2, 4, 8):
                for extra in (0, 1, m+3, 100):
                    total = m*minimum + extra
                    for lengths in (np.arange(1, m+1), np.zeros(m)):
                        counts = v.allocate_samples(lengths, total, minimum)
                        self.assertEqual(int(counts.sum()), total)
                        self.assertTrue(np.all(counts >= minimum))
        np.testing.assert_array_equal(v.allocate_samples(np.zeros(2), 10, 2), [5, 5])
        np.testing.assert_array_equal(v.allocate_samples(np.zeros(2), 11, 2), [6, 5])
        np.testing.assert_array_equal(v.allocate_samples(np.array([1, 3]), 10, 2), [4, 6])
        self.assertEqual(len(v.allocate_samples(np.array([]), 0, 2)), 0)

    def test_allocation_rejects_infeasible_or_invalid_inputs(self):
        for lengths, total, minimum in (([10, 20], 3, 8), ([1, 2], 5, 3),
                                       ([], 1, 2), ([0], -1, 2), ([1], 5, 1),
                                       ([-1], 5, 2), ([np.nan], 5, 2),
                                       ([np.inf], 5, 2), ([[1, 2]], 5, 2)):
            with self.subTest(lengths=lengths, total=total), self.assertRaises(ValueError):
                v.allocate_samples(np.array(lengths), total, minimum)

    def test_trace_budget_rejects_collapse_case(self):
        for contours in ([rectangle()], []):
            with self.assertRaises(ValueError):
                v.build_trace(contours, 320, 240, 16, 1.25, .8, 20, 8)
        for jump, minimum in ((0, 8), (1, 3), (1, 16)):
            with self.assertRaises(ValueError):
                v.build_trace([rectangle()], 320, 240, 16, 1.25, .8, jump, minimum)

    def test_feasible_traces_retain_length_and_nontrivial_geometry(self):
        for n in (16, 32, 735):
            for jump in (1, 2, 4):
                for m in (1, 3, 6):
                    contours = [rectangle()+[i*8, i*9] for i in range(m)]
                    trace = v.build_trace(contours, 320, 240, n, 1.25, .8, jump, 8)
                    self.assertEqual(trace.shape, (n, 2))
                    self.assertTrue(np.isfinite(trace).all())
                    self.assertGreater(len(np.unique(trace, axis=0)), 1)
        np.testing.assert_array_equal(v.build_trace([], 320, 240, 16, 1.25, .8, 1, 8),
                                      np.zeros((16, 2)))

    def test_resampler_structured_boundaries_match_original(self):
        for width in (1, 10, 100, 1000):
            for height in (0, 1e-8, 1e-6, 1e-4, 1):
                rect = np.array([[0, 0], [width, 0], [width, height], [0, height]], np.float32)
                for points in (rect, rect[::-1], np.insert(rect, 1, rect[0], axis=0)):
                    for n in (2, 3, 5, 33, 735):
                        np.testing.assert_array_equal(v.resample_closed_contour(points, n),
                                                      scalar_reference(points, n))

    def test_resampler_seeded_regression(self):
        rng = np.random.default_rng(20260914)
        for _ in range(500):
            points = rng.uniform(0, 640, (int(rng.integers(3, 65)), 2)).astype(np.float32)
            if rng.integers(0, 5) == 0:
                points = np.insert(points, 1, points[0], axis=0)
            n = int(rng.integers(2, 513))
            np.testing.assert_array_equal(v.resample_closed_contour(points, n),
                                          scalar_reference(points, n))

    def test_resampler_degenerate_and_invalid_inputs(self):
        points = np.array([[3, 4], [3, 4], [3, 4]], np.float32)
        np.testing.assert_array_equal(v.resample_closed_contour(points, 33),
                                      np.tile([3, 4], (33, 1)))
        for bad in (np.empty((0, 2)), np.array([1, 2]), np.array([[np.nan, 0]])):
            with self.assertRaises(ValueError):
                v.resample_closed_contour(bad, 33)
        with self.assertRaises(ValueError):
            v.resample_closed_contour(points, 1)

    def test_coordinate_mapping_both_aspect_branches(self):
        for w, h, scale_x, scale_y in ((320, 240, 1, .9375), (240, 320, .6, 1)):
            points = np.array([[0, 0], [w-1, h-1]], np.float32)
            expected = .8*np.array([[-scale_x, scale_y], [scale_x, -scale_y]])
            np.testing.assert_allclose(v.pixel_to_scope_xy(points, w, h, 1.25, .8),
                                       expected, atol=1e-7)

    def test_pcm_clipping_and_channel_order(self):
        samples = np.array([[-2, 2], [-1, 1], [0, .5]], np.float32)
        actual = np.frombuffer(v.float_to_pcm16(samples), dtype='<i2').reshape(-1, 2)
        np.testing.assert_array_equal(actual, [[-32767, 32767], [-32767, 32767], [0, 16384]])

    def test_cached_clahe_matches_fresh_objects(self):
        rng = np.random.default_rng(11)
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
        for height in (120, 180, 120):
            frame = rng.integers(0, 256, (height, 160, 3), dtype=np.uint8)
            args = (frame, 240, 1.5, 5, 3, 72, 35, 110, 1)
            fresh = v.preprocess_edges(*args)
            reused = v.preprocess_edges(*args, clahe_operator=clahe)
            for a, b in zip(fresh, reused):
                np.testing.assert_array_equal(a, b)


class SchedulingTests(unittest.TestCase):
    def test_export_retains_phase_across_arbitrary_reads(self):
        repeater = v.TraceRepeater(7)
        trace = np.arange(14, dtype=np.float32).reshape(7, 2)
        repeater.publish(trace)
        output = np.vstack([repeater.read(n) for n in (4, 13, 0, 2, 30)])
        np.testing.assert_array_equal(output, np.tile(trace, (7, 1)))

    def test_export_adopts_only_latest_publication_at_boundary(self):
        repeater = v.TraceRepeater(4)
        old = np.column_stack([np.arange(4), np.arange(4)]).astype(np.float32)
        repeater.publish(old)
        np.testing.assert_array_equal(repeater.read(3), old[:3])
        repeater.publish(old+10)
        repeater.publish(old+20)
        np.testing.assert_array_equal(repeater.read(4), np.vstack([old[3:], old[:3]+20]))

    def test_publication_is_owned_and_validated(self):
        repeater = v.TraceRepeater(4)
        trace = np.ones((4, 2), np.float32)
        repeater.publish(trace)
        trace[:] = 0
        for bad in (np.ones((3, 2)), np.ones((4, 1)), np.full((4, 2), np.nan)):
            with self.assertRaises(ValueError):
                repeater.publish(bad)
        np.testing.assert_array_equal(repeater.read(4), np.ones((4, 2)))
        with self.assertRaises(ValueError):
            repeater.read(-1)


class CommandLineTests(unittest.TestCase):
    def test_invalid_options_fail_before_resources_open(self):
        cases = (['--sobel-ksize', '2'], ['--sample-rate', '8000', '--fps', '20000'],
                 ['--jump-samples', '736'], ['--min-samples-per-contour', '736'],
                 ['--amplitude', 'nan'], ['--trace-hz', 'inf'], ['--width', '-1'],
                 ['--canny-low', '200', '--canny-high', '10'], ['--fps', '1e-320'])
        for flags in cases:
            with self.subTest(flags=flags), patch.object(sys, 'argv', ['video2xy', 'missing.avi', *flags]), \
                    patch.object(v.cv2, 'VideoCapture') as cap, patch.object(v.wave, 'open') as wav, \
                    redirect_stderr(io.StringIO()) as stderr:
                with self.assertRaises(SystemExit) as error:
                    v.main()
                self.assertEqual(error.exception.code, 2)
                self.assertIn('usage:', stderr.getvalue())
                cap.assert_not_called(); wav.assert_not_called()

    def test_camera_output_is_opt_in(self):
        parser = v.build_argparser()
        for arguments, expected in ((['0', '--play'], None), (['0', '-o', 'camera.wav'], 'camera.wav'),
                                    (['input.avi'], 'xy_scope.wav'), (['input.avi', '--no-output'], None),
                                    (['input.avi', '-o', ''], None)):
            self.assertEqual(v.resolve_output(parser.parse_args(arguments)), expected)

    def test_invalid_reported_fps_releases_capture_before_output(self):
        cap = Mock(); cap.isOpened.return_value = True; cap.get.return_value = 50000
        with patch.object(sys, 'argv', ['video2xy', 'input.avi']), \
                patch.object(v.cv2, 'VideoCapture', return_value=cap), \
                patch.object(v.wave, 'open') as wav, redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                v.main()
        cap.release.assert_called_once(); wav.assert_not_called()

    def test_audio_start_failure_closes_resources_before_wav_creation(self):
        cap = Mock(); cap.isOpened.return_value = True
        stream = Mock(); stream.start.side_effect = RuntimeError('simulated start failure')
        sd = SimpleNamespace(OutputStream=Mock(return_value=stream))
        with patch.object(sys, 'argv', ['video2xy', 'input.avi', '--play', '--fps', '30']), \
                patch.dict(sys.modules, {'sounddevice': sd}), \
                patch.object(v.cv2, 'VideoCapture', return_value=cap), patch.object(v.wave, 'open') as wav:
            with self.assertRaisesRegex(RuntimeError, 'simulated'):
                v.main()
        stream.close.assert_called_once(); stream.stop.assert_not_called()
        cap.release.assert_called_once(); wav.assert_not_called()

    def test_invalid_sobel_does_not_overwrite_existing_output(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)/'existing.wav'; output.write_bytes(b'preserve this file')
            run = subprocess.run([sys.executable, v.__file__, 'missing.avi', '-o', str(output),
                                  '--sobel-ksize', '2'], capture_output=True, text=True)
            self.assertEqual(run.returncode, 2)
            self.assertNotIn('Traceback', run.stderr)
            self.assertEqual(output.read_bytes(), b'preserve this file')

    def test_static_video_export_is_continuous_at_25_fps(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder)/'static.avi'
            writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*'MJPG'), 25, (160, 120))
            self.assertTrue(writer.isOpened())
            frame = np.zeros((120, 160, 3), np.uint8)
            cv2.rectangle(frame, (20, 25), (120, 95), (255, 255, 255), 2)
            for _ in range(6):
                writer.write(frame)
            writer.release()
            for flags in ([], ['--no-tracking']):
                output = Path(folder)/'continuous.wav'
                run = subprocess.run([sys.executable, v.__file__, str(source), '-o', str(output),
                                      '--fps', '25', '--max-contours', '1', '--epsilon', '0.02',
                                      *flags], capture_output=True, text=True, timeout=30)
                self.assertEqual(run.returncode, 0, run.stderr)
                with wave.open(str(output), 'rb') as wav:
                    self.assertEqual(wav.getnframes(), 10584)
                    samples = np.frombuffer(wav.readframes(10584), dtype='<i2').reshape(-1, 2)
                expected = np.tile(samples[:735], (15, 1))[:len(samples)]
                self.assertGreater(np.abs(samples).max(), 0)
                np.testing.assert_array_equal(samples, expected)


if __name__ == '__main__':
    unittest.main()
