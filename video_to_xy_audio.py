#!/usr/bin/env python3
"""
video_to_xy_audio.py

Convert video frames into a stereo XY oscilloscope signal.

Left audio channel  -> X (oscilloscope Channel I, "black")
Right audio channel -> Y (oscilloscope Channel II, "red")

Image processing pipeline:
    grayscale
    optional CLAHE
    Gaussian blur
    Sobel X/Y gradients
    Sobel magnitude threshold
    Canny edge detector
    morphology closing
    contour extraction
    polygon simplification
    temporal contour matching and stable scan ordering
    arc-length resampling

The resulting XY trajectory is repeated at --trace-hz and encoded as
44.1 kHz stereo audio. It can be written to a WAV file and/or played
through the sound card in real time.

Dependencies:
    pip install numpy opencv-python
Optional for --play:
    pip install sounddevice

Usage:

    For a video file, a good first experiment is:

    python video_to_xy_audio.py input.mp4 -o scope.wav --preview

    Then play scope.wav normally with all sound enhancements/EQ disabled. Left goes to HM507 CH I/X, right to CH II/Y.

    For direct real-time playback:

    python video_to_xy_audio.py input.mp4 --play --preview

    or from a webcam:

    python video_to_xy_audio.py 0 --play --preview --fps 30

    The parameters I would experiment with first are:

    python video_to_xy_audio.py input.mp4 -o scope.wav --preview \
        --trace-hz 60 \
        --sobel-percentile 72 \
        --blur 5 \
        --max-contours 32 \
        --epsilon 0.004 \
        --min-perimeter 18    
"""

from __future__ import annotations

import argparse
import math
import sys
import time
import wave
from pathlib import Path

import cv2
import numpy as np


def parse_source(s: str):
    """Interpret a purely numeric source as a camera index."""
    return int(s) if s.isdigit() else s


class FrameSampleClock:
    """Generate integer frame sample counts without long-term drift."""
    def __init__(self, sample_rate: int, fps: float):
        self.samples_per_frame = float(sample_rate) / float(fps)
        self.target_total = 0.0
        self.written_total = 0

    def next_count(self) -> int:
        self.target_total += self.samples_per_frame
        new_total = int(round(self.target_total))
        n = new_total - self.written_total
        self.written_total = new_total
        return max(n, 1)


def preprocess_edges(
    frame: np.ndarray,
    width: int,
    clahe_clip: float,
    blur_ksize: int,
    sobel_ksize: int,
    sobel_percentile: float,
    canny_low: int,
    canny_high: int,
    close_iters: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Resize frame and create a robust binary edge map."""
    h0, w0 = frame.shape[:2]
    if width > 0 and w0 != width:
        scale = width / float(w0)
        h = max(2, int(round(h0 * scale)))
        frame = cv2.resize(frame, (width, h), interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if clahe_clip > 0:
        clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

    blur_ksize = max(1, int(blur_ksize))
    if blur_ksize % 2 == 0:
        blur_ksize += 1
    if blur_ksize > 1:
        gray_blur = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)
    else:
        gray_blur = gray

    sobel_ksize = int(sobel_ksize)
    if sobel_ksize not in (1, 3, 5, 7):
        raise ValueError("--sobel-ksize must be 1, 3, 5, or 7")

    gx = cv2.Sobel(gray_blur, cv2.CV_32F, 1, 0, ksize=sobel_ksize)
    gy = cv2.Sobel(gray_blur, cv2.CV_32F, 0, 1, ksize=sobel_ksize)
    mag = cv2.magnitude(gx, gy)

    nz = mag[mag > 0]
    if nz.size == 0:
        sobel_mask = np.zeros_like(gray, dtype=np.uint8)
    else:
        threshold = float(np.percentile(nz, np.clip(sobel_percentile, 0, 100)))
        sobel_mask = (mag >= threshold).astype(np.uint8) * 255

    # Canny supplies thin edges; the Sobel mask rejects weak Canny responses.
    canny = cv2.Canny(gray_blur, canny_low, canny_high, L2gradient=True)
    edges = cv2.bitwise_and(canny, sobel_mask)

    if close_iters > 0:
        kernel = np.ones((3, 3), dtype=np.uint8)
        edges = cv2.morphologyEx(
            edges, cv2.MORPH_CLOSE, kernel, iterations=int(close_iters)
        )

    return frame, edges


def extract_contours(
    edges: np.ndarray,
    min_perimeter: float,
    max_contours: int,
    epsilon_frac: float,
) -> list[np.ndarray]:
    """Extract, filter, and simplify image contours."""
    contours, _ = cv2.findContours(
        edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE
    )

    kept: list[tuple[float, np.ndarray]] = []
    for c in contours:
        p = float(cv2.arcLength(c, closed=True))
        if p < min_perimeter:
            continue

        eps = max(0.0, epsilon_frac) * p
        approx = cv2.approxPolyDP(c, epsilon=eps, closed=True)
        pts = approx[:, 0, :].astype(np.float32)

        if len(pts) >= 3:
            kept.append((p, pts))

    kept.sort(key=lambda t: t[0], reverse=True)
    return [pts for _, pts in kept[:max_contours]]


def polyline_length(points: np.ndarray, closed: bool = True) -> float:
    if len(points) < 2:
        return 0.0
    q = np.vstack([points, points[0]]) if closed else points
    return float(np.linalg.norm(np.diff(q, axis=0), axis=1).sum())


def rotate_closed_contour_to_nearest(
    contour: np.ndarray, target: np.ndarray
) -> np.ndarray:
    """Rotate contour vertex order so it begins nearest target."""
    d2 = np.sum((contour - target) ** 2, axis=1)
    i = int(np.argmin(d2))
    return np.vstack([contour[i:], contour[:i]])


def order_contours(contours: list[np.ndarray]) -> list[np.ndarray]:
    """
    Greedy contour ordering.

    Each closed contour is rotated so its starting vertex is as close as
    possible to the end/start point of the preceding contour. This reduces
    bright inter-contour jump lines.
    """
    if not contours:
        return []

    remaining = [c.copy() for c in contours]
    ordered = [remaining.pop(0)]  # largest first
    current = ordered[0][0]

    while remaining:
        best_j = 0
        best_i = 0
        best_d2 = math.inf

        for j, c in enumerate(remaining):
            d2 = np.sum((c - current) ** 2, axis=1)
            i = int(np.argmin(d2))
            val = float(d2[i])
            if val < best_d2:
                best_d2 = val
                best_j = j
                best_i = i

        c = remaining.pop(best_j)
        c = np.vstack([c[best_i:], c[:best_i]])
        ordered.append(c)
        current = c[0]  # contour is closed, so it returns here

    return ordered


def resample_closed_contour(points: np.ndarray, n: int) -> np.ndarray:
    """Sample a closed polygon approximately uniformly in arc length."""
    n = max(int(n), 2)

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
        while idx + 1 < len(cumulative) - 1 and cumulative[idx + 1] < t:
            idx += 1

        denom = cumulative[idx + 1] - cumulative[idx]
        u = 0.0 if denom <= 1e-12 else (t - cumulative[idx]) / denom
        out[k] = p[idx] * (1.0 - u) + p[idx + 1] * u

    return out


class ContourTracker:
    """Match adjacent frames while retaining scan order and contour phase.

    Matching uses position, bounding-box size, and perimeter, with a global
    greedy one-to-one assignment. Missing contours disappear immediately;
    new contours follow surviving tracks. No old geometry is drawn.
    """

    def __init__(self, max_distance: float = 0.08):
        self.max_distance = max_distance
        self.previous: list[np.ndarray] = []
        self.frame_size: tuple[int, int] | None = None

    @staticmethod
    def describe(c: np.ndarray):
        lo, hi = c.min(axis=0), c.max(axis=0)
        return (lo + hi) * 0.5, np.maximum(hi - lo, 1.0), polyline_length(c)

    @staticmethod
    def align(c: np.ndarray, previous: np.ndarray) -> np.ndarray:
        # Project the old starting point onto the translated new polygon.
        # This also handles changes in the number of simplified vertices.
        center, _, _ = ContourTracker.describe(c)
        old_center, _, _ = ContourTracker.describe(previous)
        target = previous[0] + center - old_center
        ends = np.roll(c, -1, axis=0)
        delta = ends - c
        u = np.clip(np.sum((target - c) * delta, axis=1) /
                    np.maximum(np.sum(delta * delta, axis=1), 1e-12), 0, 1)
        projected = c + u[:, None] * delta
        i = int(np.argmin(np.sum((projected - target) ** 2, axis=1)))
        forward = np.vstack([projected[i], c[i + 1:], c[:i + 1]])
        reverse = np.vstack([forward[:1], forward[:0:-1]])
        reference = resample_closed_contour(previous, 33) - old_center
        def error(points):
            return float(np.sum((resample_closed_contour(points, 33)
                                 - center - reference) ** 2))
        return forward if error(forward) <= error(reverse) else reverse

    def update(self, contours: list[np.ndarray], width: int,
               height: int) -> list[np.ndarray]:
        size = (width, height)
        if self.frame_size != size:
            self.previous = []
            self.frame_size = size
        if not contours:
            self.previous = []
            return []
        distance_limit = self.max_distance * math.hypot(width, height)
        old = [self.describe(c) for c in self.previous]
        new = [self.describe(c) for c in contours]
        candidates = []
        for i, (center, extent, perimeter) in enumerate(old):
            for j, (other, other_extent, other_perimeter) in enumerate(new):
                distance = float(np.linalg.norm(center - other))
                size_change = float(np.max(np.abs(np.log(other_extent / extent))))
                length_change = abs(math.log(max(other_perimeter, 1e-6) /
                                             max(perimeter, 1e-6)))
                if (distance <= distance_limit and size_change <= math.log(2)
                        and length_change <= math.log(2)):
                    cost = distance / max(distance_limit, 1e-6)
                    candidates.append((cost + size_change + length_change, i, j))
        matches = {}
        used = set()
        for _, i, j in sorted(candidates):
            if i not in matches and j not in used:
                matches[i] = j
                used.add(j)
        ordered = [self.align(contours[matches[i]], previous)
                   for i, previous in enumerate(self.previous) if i in matches]
        newcomers = [c for j, c in enumerate(contours) if j not in used]
        if ordered:
            # Use the last survivor as the anchor, without reordering survivors.
            ordered.extend(order_contours([ordered[-1], *newcomers])[1:])
        else:
            ordered = order_contours(newcomers)
        self.previous = [c.copy() for c in ordered]
        return ordered


def allocate_samples(
    lengths: np.ndarray,
    total: int,
    min_each: int,
) -> np.ndarray:
    """Allocate exactly total samples among contours."""
    m = len(lengths)
    if m == 0:
        return np.zeros(0, dtype=int)

    min_each = max(2, int(min_each))
    base = np.full(m, min_each, dtype=int)
    remaining = total - int(base.sum())

    if remaining < 0:
        # Caller normally avoids this, but keep behaviour safe.
        base[:] = 2
        remaining = total - int(base.sum())

    if remaining <= 0:
        # Adjust to exact total if possible.
        while base.sum() > total:
            i = int(np.argmax(base))
            if base[i] <= 2:
                break
            base[i] -= 1
        while base.sum() < total:
            base[int(np.argmax(lengths))] += 1
        return base

    weights = lengths / max(float(lengths.sum()), 1e-12)
    raw = weights * remaining
    add = np.floor(raw).astype(int)
    counts = base + add

    leftover = total - int(counts.sum())
    if leftover > 0:
        frac_order = np.argsort(-(raw - add))
        for i in frac_order[:leftover]:
            counts[i] += 1

    return counts


def pixel_to_scope_xy(
    pts: np.ndarray,
    w: int,
    h: int,
    scope_aspect: float,
    amplitude: float,
) -> np.ndarray:
    """
    Convert pixel coordinates to normalized scope coordinates.

    The image aspect ratio is preserved relative to the physical scope
    graticule aspect ratio (10/8 = 1.25 for a typical HM507 display).
    """
    x = 2.0 * pts[:, 0] / max(w - 1, 1) - 1.0
    y = 1.0 - 2.0 * pts[:, 1] / max(h - 1, 1)  # positive Y is upward

    src_aspect = w / float(h)
    scope_aspect = max(float(scope_aspect), 1e-6)

    if src_aspect > scope_aspect:
        y *= scope_aspect / src_aspect
    else:
        x *= src_aspect / scope_aspect

    xy = np.column_stack([x, y]).astype(np.float32)
    xy *= float(amplitude)
    return xy


def build_trace(
    contours: list[np.ndarray],
    width: int,
    height: int,
    n_trace: int,
    scope_aspect: float,
    amplitude: float,
    jump_samples: int,
    min_samples_per_contour: int,
    preserve_order: bool = False,
) -> np.ndarray:
    """Build exactly one periodic XY scan trajectory."""
    if n_trace < 8 or not contours:
        return np.zeros((n_trace, 2), dtype=np.float32)

    if not preserve_order:
        contours = order_contours(contours)

    jump_samples = max(1, int(jump_samples))
    min_samples_per_contour = max(4, int(min_samples_per_contour))

    # There is one jump after every contour, including the final jump back
    # to the first contour's start point.
    max_n_contours = max(
        1, n_trace // (min_samples_per_contour + jump_samples)
    )
    contours = contours[:max_n_contours]

    lengths = np.array(
        [polyline_length(c, closed=True) for c in contours],
        dtype=np.float64,
    )

    visible_budget = n_trace - len(contours) * jump_samples
    if visible_budget < 2 * len(contours):
        contours = contours[:1]
        lengths = lengths[:1]
        visible_budget = n_trace - jump_samples

    counts = allocate_samples(
        lengths, visible_budget, min_samples_per_contour
    )

    starts = [c[0].astype(np.float32) for c in contours]
    pieces: list[np.ndarray] = []

    for i, (c, count) in enumerate(zip(contours, counts)):
        traced = resample_closed_contour(c, int(count))
        pieces.append(traced)

        next_start = starts[(i + 1) % len(starts)]

        # Move to the next contour in as few samples as possible. Without a
        # Z/intensity channel, these transitions cannot be truly blanked.
        # Fast jumps make them relatively faint.
        a = traced[-1]
        jump = np.linspace(
            a, next_start, jump_samples + 1, endpoint=True, dtype=np.float32
        )[1:]
        pieces.append(jump)

    trace_px = np.vstack(pieces)

    # Defensive exact-length correction.
    if len(trace_px) > n_trace:
        trace_px = trace_px[:n_trace]
    elif len(trace_px) < n_trace:
        pad = np.repeat(trace_px[-1:], n_trace - len(trace_px), axis=0)
        trace_px = np.vstack([trace_px, pad])

    return pixel_to_scope_xy(
        trace_px, width, height, scope_aspect, amplitude
    )


class LiveTracePlayer:
    """Repeat the latest scan independently of camera/processing speed.

    Published arrays are immutable after publication. The callback takes a
    reference at a scan boundary, so updates never splice two scan shapes.
    """

    def __init__(self, n_trace: int):
        self.pending = np.zeros((n_trace, 2), dtype=np.float32)
        self.active = self.pending
        self.position = 0
        self.underflows = 0

    def publish(self, trace: np.ndarray):
        self.pending = trace.copy()

    def callback(self, outdata, frames, time_info, status):
        if status.output_underflow:
            self.underflows += 1
        offset = 0
        while offset < frames:
            if self.position == 0:
                self.active = self.pending
            count = min(frames - offset, len(self.active) - self.position)
            outdata[offset:offset + count] = self.active[self.position:self.position + count]
            offset += count
            self.position = (self.position + count) % len(self.active)


def repeat_trace(trace: np.ndarray, n_samples: int) -> np.ndarray:
    """Repeat/truncate a periodic trace to fill one video-frame duration."""
    if len(trace) == 0:
        return np.zeros((n_samples, 2), dtype=np.float32)

    reps = int(math.ceil(n_samples / len(trace)))
    return np.tile(trace, (reps, 1))[:n_samples].astype(np.float32, copy=False)


def float_to_pcm16(xy: np.ndarray) -> bytes:
    x = np.clip(xy, -1.0, 1.0)
    pcm = np.round(x * 32767.0).astype("<i2")
    return pcm.tobytes()


def draw_preview(frame: np.ndarray, edges: np.ndarray, contours: list[np.ndarray]):
    overlay = frame.copy()
    for c in contours:
        cv2.polylines(
            overlay,
            [np.round(c).astype(np.int32)],
            isClosed=True,
            color=(0, 255, 0),
            thickness=1,
            lineType=cv2.LINE_AA,
        )

    cv2.imshow("XY contours", overlay)
    cv2.imshow("Edges", edges)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Convert video contours to stereo XY oscilloscope audio."
    )
    p.add_argument("input", help="Video filename or camera index, e.g. 0")
    p.add_argument(
        "-o", "--output", default="xy_scope.wav",
        help="Output stereo WAV file. Use '' to disable file output."
    )
    p.add_argument("--play", action="store_true",
                   help="Play generated XY audio live via sounddevice.")
    p.add_argument("--device", default=None,
                   help="sounddevice output device name/index.")
    p.add_argument("--sample-rate", type=int, default=44100)
    p.add_argument(
        "--fps", type=float, default=0.0,
        help="Override input FPS. Useful for webcams."
    )
    p.add_argument(
        "--trace-hz", type=float, default=60.0,
        help="How often the complete contour drawing is retraced."
    )
    p.add_argument(
        "--width", type=int, default=480,
        help="Internal video width; smaller is faster/cleaner."
    )
    p.add_argument(
        "--scope-aspect", type=float, default=1.25,
        help="Physical scope screen aspect ratio; HM507 graticule is ~10/8."
    )
    p.add_argument(
        "--amplitude", type=float, default=0.80,
        help="Digital full-scale XY amplitude, 0..1."
    )

    # Image / contour controls
    p.add_argument("--clahe", type=float, default=1.5,
                   help="CLAHE clip limit; 0 disables.")
    p.add_argument("--blur", type=int, default=5,
                   help="Gaussian blur kernel size.")
    p.add_argument("--sobel-ksize", type=int, default=3)
    p.add_argument(
        "--sobel-percentile", type=float, default=72.0,
        help="Keep Sobel responses above this percentile."
    )
    p.add_argument("--canny-low", type=int, default=35)
    p.add_argument("--canny-high", type=int, default=110)
    p.add_argument("--close-iters", type=int, default=1)
    p.add_argument(
        "--min-perimeter", type=float, default=18.0,
        help="Discard contours shorter than this many resized pixels."
    )
    p.add_argument("--max-contours", type=int, default=32)
    p.add_argument("--no-tracking", action="store_true",
                   help="Disable temporal contour tracking for comparison.")
    p.add_argument("--tracking-distance", type=float, default=0.08,
                   help="Maximum contour motion per frame as a fraction of the "
                        "image diagonal (default: 0.08).")
    p.add_argument(
        "--epsilon", type=float, default=0.004,
        help="Polygon simplification epsilon as fraction of contour perimeter."
    )

    # XY scan controls
    p.add_argument(
        "--jump-samples", type=int, default=1,
        help="Samples used between contours; smaller makes jump lines dimmer."
    )
    p.add_argument("--min-samples-per-contour", type=int, default=8)

    p.add_argument("--preview", action="store_true")
    p.add_argument(
        "--max-seconds", type=float, default=0.0,
        help="Stop after this duration; 0 means until video ends/Ctrl-C."
    )
    return p


def main() -> int:
    args = build_argparser().parse_args()

    if not (0.0 < args.amplitude <= 1.0):
        raise ValueError("--amplitude must be in (0, 1]")
    if args.trace_hz <= 0:
        raise ValueError("--trace-hz must be > 0")
    if args.sample_rate < 8000:
        raise ValueError("--sample-rate is implausibly low")
    if not (0 < args.tracking_distance <= 1):
        raise ValueError("--tracking-distance must be in (0, 1]")
    tracker = None if args.no_tracking else ContourTracker(args.tracking_distance)

    source = parse_source(args.input)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Could not open input: {args.input}", file=sys.stderr)
        return 2

    fps = args.fps if args.fps > 0 else float(cap.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps <= 1e-3:
        fps = 30.0
        print("Input FPS unavailable; using 30 fps.", file=sys.stderr)

    n_trace = max(16, int(round(args.sample_rate / args.trace_hz)))
    clock = FrameSampleClock(args.sample_rate, fps)

    wav = None
    if args.output:
        wav = wave.open(str(Path(args.output)), "wb")
        wav.setnchannels(2)
        wav.setsampwidth(2)
        wav.setframerate(args.sample_rate)

    stream = None
    player = LiveTracePlayer(n_trace)
    if args.play:
        try:
            import sounddevice as sd
        except ImportError:
            print(
                "--play requires sounddevice: pip install sounddevice",
                file=sys.stderr,
            )
            return 3

        device = args.device
        if isinstance(device, str) and device.isdigit():
            device = int(device)

        stream = sd.OutputStream(
            samplerate=args.sample_rate,
            channels=2,
            dtype="float32",
            device=device,
            blocksize=0,
            callback=player.callback,
        )
        stream.start()

    print(
        f"Input FPS: {fps:.6g} | audio: {args.sample_rate} Hz stereo | "
        f"trace: {args.trace_hz:.3g} Hz ({n_trace} samples/trace)"
    )

    frame_index = 0
    next_frame_time = time.monotonic()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            processed, edges = preprocess_edges(
                frame=frame,
                width=args.width,
                clahe_clip=args.clahe,
                blur_ksize=args.blur,
                sobel_ksize=args.sobel_ksize,
                sobel_percentile=args.sobel_percentile,
                canny_low=args.canny_low,
                canny_high=args.canny_high,
                close_iters=args.close_iters,
            )
            h, w = edges.shape

            contours = extract_contours(
                edges,
                min_perimeter=args.min_perimeter,
                max_contours=args.max_contours,
                epsilon_frac=args.epsilon,
            )
            if tracker is not None:
                contours = tracker.update(contours, w, h)

            trace = build_trace(
                contours=contours,
                width=w,
                height=h,
                n_trace=n_trace,
                scope_aspect=args.scope_aspect,
                amplitude=args.amplitude,
                jump_samples=args.jump_samples,
                min_samples_per_contour=args.min_samples_per_contour,
                preserve_order=tracker is not None,
            )

            n_frame = clock.next_count()
            audio = repeat_trace(trace, n_frame)

            if wav is not None:
                wav.writeframesraw(float_to_pcm16(audio))

            if stream is not None:
                player.publish(trace)

            if args.preview:
                draw_preview(processed, edges, contours)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break

            frame_index += 1
            if args.max_seconds > 0 and frame_index / fps >= args.max_seconds:
                break
            if stream is not None:
                # Callback playback no longer provides blocking-write pacing.
                # Pace fast sources, but do not try to catch up after a stall.
                next_frame_time += 1.0 / fps
                now = time.monotonic()
                if next_frame_time > now:
                    time.sleep(next_frame_time - now)
                else:
                    next_frame_time = now

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        if wav is not None:
            wav.close()
        if stream is not None:
            stream.stop()
            stream.close()
            if player.underflows:
                print(f"Audio output underruns: {player.underflows}", file=sys.stderr)
        if args.preview:
            cv2.destroyAllWindows()

    print(f"Processed {frame_index} frames.")
    if args.output:
        print(f"Wrote: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
