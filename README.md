# video2xy

Convert raster video or camera frames into stereo XY oscilloscope commands. The left channel carries X and the right channel carries Y. Contours are associated across adjacent frames to stabilize their scan order, starting point, and direction.

## Install

Python 3.11 or later is required. From this directory:

```sh
pip install .
# Optional live audio:
pip install '.[play]'
```

This installs the `video2xy` command. Running `python video_to_xy_audio.py` also works. The runtime dependencies are NumPy and OpenCV; live output additionally uses sounddevice. Preview needs an OpenCV build with GUI support. For headless development and the pinned test environment:

```sh
pip install -r requirements-test.txt
pip install --no-deps -e .
```

Use one OpenCV wheel variant in an environment: the headless test requirements replace the GUI wheel.

## Usage

```sh
video2xy input.mp4 -o scope.wav --preview
video2xy input.mp4 --play --preview --no-output
video2xy 0 --play --preview --fps 30
video2xy 0 --play --fps 30 -o camera.wav
```

File inputs default to `xy_scope.wav`. Camera inputs (purely numeric source strings) write no file unless `-o` is specified. `--no-output` or an empty output argument disables file writing. Select at least one of file output, playback, and preview.

For a video file, controls to explore include:

```sh
video2xy input.mp4 -o scope.wav --preview \
    --trace-hz 60 --sobel-percentile 72 --blur 5 \
    --max-contours 32 --epsilon 0.004 --min-perimeter 18
```

Use an appropriate stereo interface and XY display setup. Digital amplitude is a dimensionless coordinate scale, not a calibrated output voltage. Audio processing, channel gain, coupling, and display response affect the result. Transitions between contours remain visible: two channels provide no independent intensity blanking. An empty detection produces a center command, not a beam-off command.

## Timing and output semantics

At 44,100 sample pairs/s and a requested 60 scans/s, each scan contains 735 pairs. The program reports the realized scan frequency, which can differ from the request after integer rounding.

WAV and live output share `TraceRepeater`. It keeps its position across reads and adopts the latest published geometry only at a scan boundary. At 25 video frames/s, a frame receives 1,764 pairs: after two full scans, the next frame continues at position 294. Geometry updates can still have different boundary coordinates. Faster publications can replace pending geometry before it is used.

WAV duration follows decoded/acquired frame count and the nominal frame rate. It is not a recording of the audio callback or of camera wall time. The WAV and live player have separate scheduler instances, so their output streams can differ. Live output repeats its current scan through producer stalls; file playback pacing does not catch up after a stall. Variable-frame-rate presentation timestamps are not used, and `--max-seconds` measures nominal video duration. No hard real-time or analog fidelity guarantee is made.

`--fps 0` selects automatic metadata timing, with 30 fps as fallback for unavailable/nonfinite metadata or values at most 0.001 fps. A resolved rate must satisfy `0 < fps <= sample_rate`. The frame clock differences rounded cumulative sample totals; finite-precision accumulation is tested over finite sequences, not guaranteed for unlimited durations.

## Processing and sample budget

The pipeline resizes frames, converts to grayscale, optionally applies CLAHE, blurs, intersects Canny edges with a Sobel-magnitude percentile mask, closes gaps, extracts and simplifies contours, associates adjacent frames, and resamples ordered closed paths. The CLI reuses one configured CLAHE object. Vectorized arc-length resampling uses left insertion at cumulative-length boundaries to retain the original scalar implementation's tested float32 results.

Every scan requires `jump_samples >= 1`, `min_samples_per_contour >= 4`, and enough samples for at least one contour plus its jump. The ordered contour list is capped by this joint budget. Allocation honors its minimum and returns exactly the available count; all-zero lengths divide the remainder equally, with input-order ties. Infeasible settings are rejected rather than truncating a malformed drawing. Invalid CLI options are checked before opening capture/output resources; resolved metadata timing is checked before opening the WAV.

Tracking is a greedy geometric heuristic. It does not preserve identities through arbitrary occlusions, crossings, or detector changes, and changes in contour allocation can alter a static scene's waveform. Resampling retains the original floating-point closure residual in some polygons.

## Verification

```sh
python -m unittest discover -v
python tools/run_coverage.py
```

The suite contains 29 tests: the original eight temporal/live/export tests plus clock, allocation, resampling, mapping, PCM, cached CLAHE, scheduler, validation, resource cleanup, and 25 fps export regressions. The 25 fps fixture isolates export timing using a single simplified contour and checks every PCM pair against continuous scan repetition, with tracking both enabled and disabled.

The coverage runner instruments subprocesses through a temporary startup hook, combines their data, and writes `coverage.json`. It does not change the Python installation. GitHub Actions is configured for Ubuntu and Windows on Python 3.12 and 3.13, with pinned test dependencies and subprocess-aware coverage. These jobs run when the changes are published to GitHub; configuring them is not evidence of a hosted CI pass.

## Changes in 0.2.0

- Vectorized resampling with boundary-case regressions; reused CLAHE state.
- Rejected unsupported frame rates and infeasible scan budgets; defined zero-length allocation.
- Validated options before output side effects and managed resource cleanup with `ExitStack`.
- Shared scan-boundary scheduling for phase-continuous WAV and live output.
- Made camera WAV recording explicit and added `--no-output`.
- Removed the unused nearest-vertex rotation helper and added packaging, regression tests, and CI.

## Paper and reproducibility

[Read the paper (PDF)](paper/video2xy.pdf), or browse the [LaTeX source, references, experiments, and recorded results](paper/README.md).

The manuscript documents implementation commit `cc7fe7a82c60362b43f500489821624e65326bf2`, including the 29-test regression suite, resampling compatibility checks, continuous 25 fps export, and a fixed-image processing benchmark. The paper directory includes instructions for reproducing the measurements and compiling the manuscript. AI assistance is disclosed in the paper and below.

## AI assistance

Peter Schwendner used OpenAI Codex for the original project and this revision. User-supplied reviews attributed to Claude informed the changes; their claims were checked against source and executed tests. AI assistance also contributed to the companion manuscript and verification scripts. The human author is responsible for the final implementation and interpretation.
