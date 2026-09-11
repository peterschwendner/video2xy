import subprocess
import sys
import tempfile
import unittest
import wave
from types import SimpleNamespace
from pathlib import Path

import cv2
import numpy as np

from video_to_xy_audio import ContourTracker, LiveTracePlayer, build_trace


def rectangle(x, y, w=30, h=20):
    return np.array([[x, y], [x+w, y], [x+w, y+h], [x, y+h]], np.float32)


def trace(contours):
    return build_trace(contours, 320, 240, 735, 1.25, 0.8, 1, 8,
                       preserve_order=True)


class TrackingTests(unittest.TestCase):
    def test_live_playback_repeats_during_camera_stall(self):
        player = LiveTracePlayer(7)
        scan = np.arange(14, dtype=np.float32).reshape(7, 2)
        player.publish(scan)
        blocks = []
        for size in (4, 13, 2, 30):
            block = np.empty((size, 2), np.float32)
            player.callback(block, size, None, SimpleNamespace(output_underflow=False))
            blocks.append(block)
        np.testing.assert_array_equal(np.vstack(blocks), np.tile(scan, (7, 1)))

    def test_live_update_waits_for_scan_boundary(self):
        player = LiveTracePlayer(7)
        old = np.full((7, 2), 0.25, np.float32)
        new = np.full((7, 2), 0.75, np.float32)
        player.publish(old)
        block = np.empty((3, 2), np.float32)
        status = SimpleNamespace(output_underflow=False)
        player.callback(block, 3, None, status)
        player.publish(new)
        new[:] = 0  # Publishing owns a copy, independent of the producer.
        block = np.empty((12, 2), np.float32)
        player.callback(block, 12, None, SimpleNamespace(output_underflow=True))
        np.testing.assert_array_equal(block[:4], 0.25)
        np.testing.assert_array_equal(block[4:], 0.75)
        self.assertEqual(player.underflows, 1)

    def test_detection_order_start_and_direction_do_not_change_audio(self):
        tracker = ContourTracker()
        a, b = rectangle(20, 20), rectangle(180, 140)
        first = trace(tracker.update([a, b], 320, 240))
        changed = [np.roll(b[::-1], 2, axis=0), np.roll(a[::-1], 1, axis=0)]
        second = trace(tracker.update(changed, 320, 240))
        np.testing.assert_allclose(first, second, atol=1e-6)

    def test_motion_and_vertex_count_change(self):
        tracker = ContourTracker()
        a = rectangle(20, 20)
        first = tracker.update([a], 320, 240)[0]
        subdivided = np.insert(a, 1, (a[0] + a[1]) / 2, axis=0)
        moved = tracker.update([np.roll(subdivided[::-1] + [3, 2], 2, axis=0)],
                               320, 240)[0]
        np.testing.assert_allclose(moved[0], first[0] + [3, 2], atol=1e-5)
        expected = trace([first + [3, 2]])
        np.testing.assert_allclose(trace([moved]), expected, atol=1e-6)

    def test_survivors_keep_order_and_newcomers_follow(self):
        tracker = ContourTracker()
        a, b, c = rectangle(20, 20), rectangle(180, 140), rectangle(270, 30)
        tracker.update([a, b], 320, 240)
        result = tracker.update([c, b], 320, 240)
        self.assertEqual(len(result), 2)
        np.testing.assert_allclose(result[0].min(axis=0), b.min(axis=0))
        np.testing.assert_allclose(result[1].min(axis=0), c.min(axis=0))

    def test_empty_frame_and_resolution_change_reset(self):
        tracker = ContourTracker()
        a, b = rectangle(20, 20), rectangle(180, 140)
        tracker.update([a, b], 320, 240)
        self.assertEqual(tracker.update([], 320, 240), [])
        np.testing.assert_array_equal(tracker.update([b, a], 320, 240)[0], b)
        np.testing.assert_array_equal(tracker.update([a, b], 640, 480)[0], a)

    def test_large_motion_starts_new_track(self):
        tracker = ContourTracker()
        tracker.update([rectangle(20, 20)], 320, 240)
        new = np.roll(rectangle(200, 150)[::-1], 2, axis=0)
        np.testing.assert_array_equal(tracker.update([new], 320, 240)[0], new)

    def test_video_to_wav_with_and_without_tracking(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / 'moving.avi'
            writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*'MJPG'),
                                     30, (160, 120))
            self.assertTrue(writer.isOpened())
            for i in range(6):
                frame = np.zeros((120, 160, 3), np.uint8)
                cv2.rectangle(frame, (20+i, 20), (70+i, 80), (255, 255, 255), 2)
                writer.write(frame)
            writer.release()
            for flags in ([], ['--no-tracking']):
                output = Path(folder) / 'output.wav'
                run = subprocess.run(
                    [sys.executable, str(Path(__file__).with_name('video_to_xy_audio.py')),
                     str(source), '-o', str(output), '--fps', '30', *flags],
                    capture_output=True, text=True, timeout=30)
                self.assertEqual(run.returncode, 0, run.stderr)
                with wave.open(str(output)) as audio:
                    self.assertEqual(audio.getnchannels(), 2)
                    self.assertEqual(audio.getnframes(), 8820)
                    samples = np.frombuffer(audio.readframes(8820), dtype='<i2')
                    self.assertGreater(np.max(np.abs(samples)), 0)


if __name__ == '__main__':
    unittest.main()
