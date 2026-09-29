# TactiCal — Tactile-Assisted Hand-Eye Calibration

TactiCal is a hand-eye calibration system for a fixed-camera robotic setup that uses
a **DIGIT tactile sensor** to provide physical ground truth in addition to (or instead
of) purely vision-based calibration. The central research claim is **robustness under
visual degradation**: when camera input is blurred, over/under-exposed, noisy, or
partially occluded, a physical touch from the DIGIT gel still yields a trustworthy
position, whereas a vision-only calibration has no such fallback.

This repository contains the three core Python scripts of the pipeline. It builds on
prior work by Hamza Manzoor, developed at TU Dresden's LASR Lab.

## Hardware

| Component | Detail |
|---|---|
| Robot arm | UFACTORY xArm7 (`IP 192.168.1.222` by default) |
| Tactile sensor | DIGIT gel tactile sensor, mounted at the end-effector |
| Camera | Fixed Intel RealSense RGB-D camera |
| Fiducial marker | ArUco `DICT_5X5_1000`, ID 20 (main) / ID 25 (secondary), 0.1 m edge length |

## Software dependencies

```
opencv-python
numpy
scipy
pyrealsense2
xarm-python-sdk   # xarm.wrapper.XArmAPI
digit-interface   # digit_interface.Digit
```

Install with:

```bash
pip install opencv-python numpy scipy pyrealsense2 xarm-python-sdk digit-interface
```

The xArm SDK and DIGIT interface additionally require the vendor drivers/firmware for
the physical devices (xArm controller network access, DIGIT USB permissions) to be
set up on the host machine — this repo only contains the calibration/experiment logic.

## Files

### `TactiCal.py`

The core library and entry point for calibration. Provides:

- **`DigitContactDetector`** — adaptive contact detection on the DIGIT gel image.
  Captures a noise baseline while hovering (no contact), then flags contact via an
  adaptive `mean + sigma·std` threshold on deformation area, with a sanity check that
  rejects unreachable thresholds.
- **`HandEyeCalibrator`** — the main calibration class (subclassed by both other
  scripts in this repo). Handles:
  - ArUco marker pose estimation (`solvePnP`, `IPPE_SQUARE`).
  - Guided/autonomous multi-sample touch collection.
  - **Kabsch–Umeyama** rigid registration (translations only) to solve for
    `T_base_cam` from paired camera-observed and robot-touched points.
  - **Leave-one-out (LOO) cross-validation** (`loo_cross_validation`) as the honest
    generalization-accuracy metric, as opposed to optimistic in-sample fit RMSE.
  - **Corner-press DIGIT-center calibration** (`calibrate_digit_center_at_corner`) —
    presses the gel onto a known, flat marker corner to find the true offset between
    the DIGIT gel center and the robot's registered tool-center-point (`digit_center_offset_mm`),
    persisted to disk and auto-loaded/auto-applied on every subsequent touch.
  - **Zero-parallax visual correction** (`visual_xy_correction`) — at the moment of
    contact the tool tip lies exactly on the marker plane, so a detected touch
    footprint can be backprojected to a base-frame XY correction.
  - **`touch_marker_converged`** — closed-loop touch-and-correct loop that repeats
    touch + visual correction until the residual is below tolerance, used to
    establish an independent physical ground truth (shared by the other two scripts).
  - Safety mechanisms: Z-plausibility gating before any descent, controller
    fault-clearing, hover-baseline retry with settle time, and unconditional arm
    retract after every touch.
- Also supports **classical (vision-only) hand-eye calibration** via
  `cv2.calibrateHandEye` (Tsai / Park / Horaud / Daniilidis) as the comparison
  baseline.
- Run directly (`python TactiCal.py`) to perform an interactive touch-based
  calibration session (and, if configured, the corner-press DIGIT-center
  calibration and/or the classical baseline).

### `unified_evaluation_voVSvwt.py`

Compares the **touch-based calibration** against the **classical (vision-only)
calibration**, head-to-head, on identical test points, using a shared, closed-loop
tactile-derived ground truth (`touch_marker_converged`) so neither calibration is
scored against a truth that favors it. Loads the most recent calibration `.npz`
files for both methods and reports per-point and aggregate lateral/Z error for each,
plus a breakdown across all four classical methods against the tactile ground truth.

Run with `python unified_evaluation_voVSvwt.py` (auto-discovers the latest
`hand_eye_calib_*.npz` and `classical_handeye_*.npz` files, or accepts explicit
paths).

### `visual_degradation_experiment_groundtruth.py`

The main experiment script. For a configurable sweep of synthetic visual
degradation conditions (`CONDITIONS`, in `__main__`) — exposure, Gaussian blur,
Gaussian noise, and rectangular occlusion — it:

1. Establishes an independent ground-truth position for the marker via
   `touch_marker_converged` (`establish_reference`).
2. Applies each synthetic degradation to the live camera feed
   (`make_blur`, `make_noise`, `make_occlusion`) and measures marker
   detection rate and lateral/Z prediction error vs. ground truth.
3. Optionally (`WITH_RECOVERY = True`) performs a **tactile recovery touch**
   under degraded vision (`recover_touch`), and compares the recovered physical
   touch position against ground truth — this is the "green line" in the resulting
   plots, contrasted against the "red line" (pure-vision error under degradation).
4. Saves a timestamped CSV of all trial results plus a summary matplotlib figure
   (`save_and_plot`).

Supports both `calibration_mode='touch'` and `calibration_mode='classical'`, so the
same degradation sweep can be run against either calibration for a fair comparison.

Default degradation levels (edit the `CONDITIONS` list in `__main__` to change):

| Family | Levels |
|---|---|
| exposure | 1100, 800, 500, 300 (gain=64) |
| blur (kernel px) | 5, 11, 17, 23 |
| noise (Gaussian σ) | 8, 16, 28, 40 |
| occlusion (fraction) | 0.01, 0.02, 0.05, 0.10 |

Run with `python visual_degradation_experiment_groundtruth.py`.

## Typical workflow

```bash
# 1. Touch-based hand-eye calibration (produces hand_eye_calib_*.npz)
#    Optionally run the corner-press DIGIT-center calibration first/within this step.
python TactiCal.py

# 2. Classical vision-only calibration baseline (produces classical_handeye_*.npz)
#    (invoked from within TactiCal.py / its CLI options — see __main__ for flags)

# 3. Head-to-head comparison on shared, tactile-verified ground truth
python unified_evaluation_voVSvwt.py

# 4. Visual-degradation robustness sweep, with optional tactile recovery
python visual_degradation_experiment_groundtruth.py
```

Each run writes timestamped output files (`.npz` calibration files, `.csv` trial
logs, `reference_truth_*.txt`, summary `.png` plots) into the working directory.

## Notes

- Code comments and docstrings are in English. Some user-facing runtime
  prompts/messages (`print`/`input`) are bilingual (English / 中文) for lab use.
- Hardware IP/serial values (e.g. the xArm IP) are placeholders and should be
  updated for your setup.
