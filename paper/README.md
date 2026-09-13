# video2xy: revised implementation and reproducibility

Author: Peter Schwendner, Zurich University of Applied Sciences, scwp@zhaw.ch.
Verification date: September 13, 2026.

Read the [paper (PDF)](video2xy.pdf) or its [LaTeX source](video2xy.tex). This manuscript describes implementation commit `cc7fe7a82c60362b43f500489821624e65326bf2`, based on public commit `6a7d1dd8be7622966f6e4d7e5f2dfbdec1001af6`. The implementation had not been pushed to GitHub when the measurements were recorded; the PDF preserves that historical statement. This directory adds the paper and its reproducibility materials without changing the measured implementation. No software licence was selected or added.

References to a companion Git bundle and patch in the PDF and source notes describe the original delivery archive. This repository preserves the cited Git history, source, and measurements directly; no separate bundle is required for the commands below.

## Reproduce from the repository root

Use a fresh Python environment and a full Git checkout. Python 3.12.14 was used for the recorded Windows run. The commands below create a separate checkout at the exact implementation revision cited by the paper, so later changes cannot silently affect the comparison. The pinned OpenCV dependency is headless and does not support GUI preview.

```sh
git worktree add --detach paper/verification-checkout cc7fe7a82c60362b43f500489821624e65326bf2
python -m pip install -r paper/requirements-verification.txt
python -m pip install --no-deps -e paper/verification-checkout
python -m unittest discover -s paper/verification-checkout -v
python paper/verification-checkout/tools/run_coverage.py
python paper/experiments.py --repo paper/verification-checkout --out paper/reproduced-results
python paper/implementation_checks.py --repo paper/verification-checkout --out paper/reproduced-results
```

Run the worktree command once; subsequent runs can reuse that checkout. The scripts record its Git revision, and new results go to the ignored `paper/reproduced-results/` directory, preserving the supplied measurements. SHA-256 hashes in the recorded comparison refer to the original working-file bytes; Git checkout settings can change line endings without changing Python behavior. Install the GUI OpenCV variant instead when using `--preview`; do not install both OpenCV wheel variants in one environment.

## Compile

From this `paper/` directory, run either:

```sh
tectonic video2xy.tex
```

or a normal LaTeX/BibTeX sequence:

```sh
pdflatex video2xy.tex
bibtex video2xy
pdflatex video2xy.tex
pdflatex video2xy.tex
```

The supplied `video2xy.bbl` and vector PDF figures are included for submission workflows. Tectonic 0.17.0 produced the included PDF. It contains 15 A4 pages, five tables, two figures, 18 numbered equations, and 14 references. All pages were rendered and visually checked; the build has no undefined citations/references or overfull/underfull boxes.

## Recorded evidence

- `results/metrics.json`: 200 contour representation trials, translation/subdivision check, four 100,000-frame timing checks, and the raster example.
- `results/permutation.csv` and `leading_contour.csv`: every representation trial and its leading contour identifier.
- `results/implementation_checks.json`: 3,000 random and 300 structured resampling comparisons; original/revised budget, clock, CLI, and 25 fps export cases; resampling call counts; all raw processing-stage timings.
- `results/upstream_tests.txt`: all 29 tests run by `experiments.py`; the historical filename does not mean only the original eight tests were run.
- `results/coverage_run.txt` and `coverage.json`: successful 29-test run with parent/subprocess instrumentation, 434 of 485 executable lines covered (89.5%). Branch coverage was not collected.
- `baseline/video_to_xy_audio.py`: unchanged original module read from the public baseline commit, used by the comparison script.
- `source_notes.md`: evidence scope and verified bibliography notes.

The 25 fps test uses one simplified contour (`--max-contours 1 --epsilon 0.02`) and checks all 10,584 PCM sample pairs against repetition of the first 735-pair scan. The revised output has zero mismatches; the original has 7,056. This isolates scheduling and is not a general static-scene invariance claim.

The benchmark uses ten fixed rectangles, 20 extracted contours, one OpenCV thread, ten warm-up iterations and 100 measured iterations per arm, with alternating order. Mean processing time is 26.89 ms for the original and 19.07 ms for the revision. It excludes decoding, capture, WAV writing, preview, pacing, and audio callbacks. It does not isolate individual changes or measure sustained real-time performance. Repeat timings will vary. All 110 paired scan arrays match exactly.

The original three-rectangle trial data reproduced unchanged, including all conditional ranges. Numerical compatibility is a finite observation in the recorded environment, not a proof for every input or dependency version. Camera/audio hardware and GUI preview were not exercised; GitHub Actions was configured but not run remotely.

The manuscript explicitly discloses Codex assistance in implementation, tests, experiments and writing, and the influence of two user-supplied reviews attributed to Claude. Reported numerical results came from local executions.
