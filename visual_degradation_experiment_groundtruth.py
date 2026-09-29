#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visual Degradation Experiment
==============================

Purpose
-------
Demonstrate the core value of "tactile verification" over "vision only":
when the camera input degrades (low light, defocus, noise, occlusion),
the marker position error from a pure-vision prediction grows, while a
single physical touch can pull it back to ground truth. This is a
differentiating advantage that structurally cannot be matched by
pure-vision methods such as EasyHeC / Li 2024.

Experiment design (Design A, safe, runnable in a single day)
--------------------------------------------------------------
1. Detect the marker once under "good vision", and (optionally) use a
   touch to obtain the **ground truth** p_base_true of the marker
   center in the base frame. The touch is performed only this one
   time, and the arm does not move afterward, so it is safe.
2. Apply degradation conditions one by one; for each condition:
     - Repeat detection N times -> compute the detection success rate
       and the pose jitter std;
     - Use the calibration matrix T to predict the (degraded) p_cam as
       base-frame coordinates;
     - Lateral error = ||pred_xy - p_base_true_xy||  <-- the main metric.
3. Restore the camera, save the CSV, plot the results.

Degradation methods (all reproducible, all suitable for the paper's tables)
-----------------------------------------------------------------------------
  A) Exposure / gain — real hardware, set via the RealSense option API
  B) Gaussian blur   — simulates defocus, controlled by kernel size
  C) Gaussian noise  — simulates low-light sensor noise, controlled by sigma
  D) Occlusion       — covers a fraction of the marker from one side;
                        the detection success rate drops off a cliff

Two run modes
-------------
  tactile_ground_truth=True (default, requires the robot arm):
     Ground truth comes from a single touch; the lateral error is the
     absolute value of "how far off you'd be if you trusted vision".
  tactile_ground_truth=False (pure software, arm does not move, good
     for getting the pipeline running first):
     Uses the "good vision" prediction as the reference and measures
     the drift "additionally" introduced by the degradation.

Usage
-----
    python3 visual_degradation_experiment.py
Adjust TACTILE_GROUND_TRUTH / the individual degradation levels in the
main entry point as needed.
"""

import os
import csv
import glob
import time

import cv2
import numpy as np
import pyrealsense2 as rs

from TactiCal import HandEyeCalibrator


# ===========================================================================
# Degradation function factory: returns a callable img -> img
# ===========================================================================
def make_blur(ksize):
    k = int(ksize) | 1  # force odd
    if k <= 1:
        return None

    def f(img):
        return cv2.GaussianBlur(img, (k, k), 0)
    return f


def make_noise(sigma, seed=0):
    if sigma <= 0:
        return None
    rng = np.random.default_rng(seed)

    def f(img):
        n = rng.normal(0, sigma, img.shape)
        return np.clip(img.astype(np.float32) + n, 0, 255).astype(np.uint8)
    return f


def make_occlusion(bbox, frac, side='left', value=0):
    """
    bbox : (x0, y0, x1, y1) the marker's bounding box under good vision
           (pixels).
    frac : fraction (0~1) of the marker covered along the `side` direction.
    ArUco detection is binary — covering any corner usually causes outright
    detection failure, which is itself a result.
    """
    if frac <= 0 or bbox is None:
        return None
    x0, y0, x1, y1 = [int(v) for v in bbox]
    w, h = x1 - x0, y1 - y0

    if side == 'left':
        rect = (x0, y0, x0 + int(w * frac), y1)
    elif side == 'right':
        rect = (x1 - int(w * frac), y0, x1, y1)
    elif side == 'top':
        rect = (x0, y0, x1, y0 + int(h * frac))
    else:  # bottom
        rect = (x0, y1 - int(h * frac), x1, y1)

    def f(img):
        out = img.copy()
        cv2.rectangle(out, (rect[0], rect[1]), (rect[2], rect[3]),
                      (value, value, value), -1)
        return out
    return f


# ===========================================================================
class DegradationRig(HandEyeCalibrator):
    """
    Adds on top of HandEyeCalibrator: synthetic degradation injection +
    exposure control + the experiment loop.
    By overriding _color_image, the parent class's get_marker_pose /
    detection pipeline automatically picks up the degraded image, with
    no changes needed to any detection code in TactiCal.
    """

    def __init__(self, *args, calibration_mode='touch',
                 calibration_file=None, **kwargs):
        """
        calibration_mode:
          'touch'     -> use the tactile calibration (T_base_cam from
                         hand_eye_calib_*.npz)
          'classical' -> use the classical calibration (T_base_cam_best
                         from classical_handeye_*.npz)
        calibration_file: specifies a particular file; None auto-selects
                         the newest one for that class.

        In classical mode, tactile recovery (the green line) is
        automatically disabled — the classical method has no tactile
        correction, it is pure vision, so only the red line applies.
        """
        self.calibration_mode = calibration_mode
        self._calib_file_override = calibration_file
        super().__init__(*args, **kwargs)
        self._synthetic = None          # current synthetic degradation function (blur/noise/occlusion)
        self._marker_bbox = None        # marker bounding box under good vision
        self._clean_marker_info = None  # clean marker geometry when undegraded
        self._rgb_sensor = self._find_rgb_sensor()

        # Load the calibration matrix per mode, overriding whatever
        # tactile version the parent class may have loaded
        loaded = self._load_calibration_by_mode()
        if loaded is not None:
            self.T_base_cam = loaded
        elif self.T_base_cam is None and self.T_base_cam_guess is not None:
            self.T_base_cam = self.T_base_cam_guess

        # For the touch that establishes the reference ground truth,
        # regardless of the current calibration_mode, positioning must
        # always use the tactile calibration (more accurate, and the
        # calibration currently being evaluated must not be allowed to
        # define where its own ground truth is — this is the same class
        # of fairness bug seen earlier in unified_evaluation). In
        # classical mode self.T_base_cam has already been swapped to the
        # classical calibration, so a separate tactile version is loaded
        # here, used only for that one reference-truth touch.
        if self.calibration_mode == 'classical':
            self._T_touch_for_ref = self._load_T_for_reference()
        else:
            self._T_touch_for_ref = self.T_base_cam

        print(f"\n[Degradation experiment / 退化实验] Using calibration / 使用标定: {self.calibration_mode}")

    # ------------------------------------------------------------------
    def _load_T_for_reference(self):
        """
        Dedicated to establishing the reference ground truth: regardless
        of calibration_mode, always loads the tactile calibration
        (T_base_cam from hand_eye_calib_*.npz), used to position the
        single touch that establishes ground truth.
        """
        if self._calib_file_override and self.calibration_mode == 'touch':
            path = self._calib_file_override
        else:
            files = sorted(glob.glob(os.path.join(os.getcwd(),
                                                   "hand_eye_calib_*.npz")),
                           key=os.path.getmtime)
            if not files:
                print("WARNING: no hand_eye_calib_*.npz found for "
                      "reference-truth positioning — falling back to the "
                      "calibration_mode-selected matrix (may be less "
                      "accurate for this one touch). / "
                      "WARNING: 找不到hand_eye_calib_*.npz用于定位参照"
                      "真值——退回用calibration_mode选中的那份标定"
                      "（这一次触碰的定位精度可能受影响）。")
                return self.T_base_cam
            path = files[-1]
        try:
            data = np.load(path, allow_pickle=True)
            T = data['T_base_cam']
            print(f"Loaded touch calibration for reference-truth "
                  f"positioning / 加载触觉版标定用于定位参照真值: {path}")
            return T
        except Exception as exc:
            print(f"WARNING: failed to load touch calibration for "
                  f"reference positioning ({exc}); falling back to the "
                  f"calibration_mode-selected matrix. / "
                  f"WARNING: 加载触觉版标定失败({exc})，退回用"
                  f"calibration_mode选中的那份标定。")
            return self.T_base_cam

    # ------------------------------------------------------------------
    def _load_calibration_by_mode(self):
        """Load the corresponding calibration matrix for touch / classical mode."""
        if self.calibration_mode == 'classical':
            pattern = "classical_handeye_*.npz"
            # Key point: converting p_cam (camera frame) to the base frame
            # requires the matrix in the "camera->base" direction, i.e.
            # T_cam2base_best. T_base_cam_best must NOT be used (it is the
            # inverse, the opposite direction, and using it for prediction
            # would be off by hundreds of mm). This has the same root cause
            # as the earlier bug in the evaluation harness.
            field = "T_cam2base_best"
        else:
            pattern = "hand_eye_calib_*.npz"
            field = "T_base_cam"

        if self._calib_file_override:
            path = self._calib_file_override
        else:
            files = sorted(glob.glob(os.path.join(os.getcwd(), pattern)),
                           key=os.path.getmtime)
            if not files:
                print(f"WARNING: could not find {pattern}, cannot load "
                      f"{self.calibration_mode} calibration. / "
                      f"WARNING: 找不到 {pattern}，无法加载 "
                      f"{self.calibration_mode} 标定。")
                return None
            path = files[-1]

        try:
            data = np.load(path, allow_pickle=True)
            if field not in data.files:
                print(f"WARNING: field '{field}' not found in {path}. "
                      f"Available: {data.files} / "
                      f"字段'{field}'不在{path}。可用: {data.files}")
                return None
            T = data[field]
            print(f"Loaded {self.calibration_mode} calibration "
                  f"('{field}') from: {path}")
            return T
        except Exception as exc:
            print(f"Failed to load calibration / 加载标定失败: {exc}")
            return None

    # ------------------------------------------------------------------
    def _load_any_calibration(self):
        files = sorted(glob.glob(os.path.join(os.getcwd(),
                                              "hand_eye_calib_*.npz")),
                       key=os.path.getmtime)
        if not files:
            print("WARNING: no hand_eye_calib_*.npz found — the experiment "
                  "cannot convert p_cam to the base frame. Please run a "
                  "calibration first. / "
                  "WARNING: 没有找到任何 hand_eye_calib_*.npz，"
                  "实验无法把 p_cam 转换到 base 系。请先完成一次标定。")
            return None
        T = np.load(files[-1])["T_base_cam"]
        print(f"Loaded calibration for experiment: {files[-1]}")
        return T

    # ------------------------------------------------------------------
    # Route all detection through the degradation injection
    # ------------------------------------------------------------------
    def _color_image(self, timeout_ms=3000):
        img = super()._color_image(timeout_ms)
        if img is not None and self._synthetic is not None:
            img = self._synthetic(img)
        return img

    # ------------------------------------------------------------------
    # RealSense exposure / gain control
    # ------------------------------------------------------------------
    def _find_rgb_sensor(self):
        try:
            dev = self.pipeline.get_active_profile().get_device()
            for s in dev.query_sensors():
                name = s.get_info(rs.camera_info.name) \
                    if s.supports(rs.camera_info.name) else ""
                if 'RGB' in name or 'Color' in name:
                    return s
        except Exception as exc:
            print("Could not access the RGB sensor:", exc)
        return None

    def set_exposure(self, exposure_us=None, gain=None, flush=12):
        """exposure_us=None -> restore auto exposure. Returns the actual (exp, gain) applied."""
        s = self._rgb_sensor
        if s is None:
            return None, None
        try:
            if exposure_us is None:
                s.set_option(rs.option.enable_auto_exposure, 1)
            else:
                s.set_option(rs.option.enable_auto_exposure, 0)
                rng = s.get_option_range(rs.option.exposure)
                val = float(np.clip(exposure_us, rng.min, rng.max))
                s.set_option(rs.option.exposure, val)
                if gain is not None:
                    grng = s.get_option_range(rs.option.gain)
                    s.set_option(rs.option.gain,
                                 float(np.clip(gain, grng.min, grng.max)))
        except Exception as exc:
            print("Exposure/gain set failed:", exc)

        for _ in range(flush):  # wait for the setting to take effect
            try:
                self.pipeline.wait_for_frames(2000)
            except RuntimeError:
                pass

        try:
            act_exp = (None if exposure_us is None
                       else s.get_option(rs.option.exposure))
            act_gain = s.get_option(rs.option.gain)
            return act_exp, act_gain
        except Exception:
            return exposure_us, gain

    def read_camera_state(self):
        """
        Reads back the camera's **currently actually in effect**
        exposure/gain/auto-exposure switch.
        The real exposure value can also be read under auto exposure —
        used to record the capture conditions when establishing ground
        truth.
        Returns a dict or None.
        """
        s = self._rgb_sensor
        if s is None:
            return None
        state = {}
        try:
            state['auto_exposure'] = bool(
                s.get_option(rs.option.enable_auto_exposure))
        except Exception:
            state['auto_exposure'] = None
        try:
            state['exposure'] = float(s.get_option(rs.option.exposure))
        except Exception:
            state['exposure'] = None
        try:
            state['gain'] = float(s.get_option(rs.option.gain))
        except Exception:
            state['gain'] = None
        return state

    # ------------------------------------------------------------------
    # One vision measurement: returns (detection success rate, p_cam median(3,), per-axis std_mm(3,))
    # ------------------------------------------------------------------
    def measure_vision(self, n=15, show=True):
        ts, ok = [], 0
        last_vis = None
        for _ in range(n):
            img = self._color_image()
            if img is None:
                continue
            gray = np.ascontiguousarray(
                cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), dtype=np.uint8)
            pts = self._detect_once(gray)
            if pts is None:
                if show:
                    last_vis = img.copy()
                    cv2.putText(last_vis, "DETECTION FAILED", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    cv2.imshow("Degradation view", last_vis)
                    cv2.waitKey(1)
                continue
            rvec, tvec = self._solve_marker_pose(pts)
            if rvec is None:
                continue
            ok += 1
            ts.append(tvec)
            if show:
                last_vis = img.copy()
                cv2.polylines(last_vis, [pts.astype(np.int32)], True,
                              (0, 255, 0), 2)
                cv2.imshow("Degradation view", last_vis)
                cv2.waitKey(1)

        rate = ok / max(n, 1)
        if not ts:
            return rate, None, None
        arr = np.array(ts)
        return rate, np.median(arr, axis=0), np.std(arr, axis=0) * 1000.0

    # ------------------------------------------------------------------
    # Establish the reference: detection under good vision + (optional) touch ground truth
    # ------------------------------------------------------------------
    def establish_reference(self, tactile_ground_truth=True):
        self._synthetic = None
        self.set_exposure(None)  # auto
        input("\nPlace the marker, ensure good lighting/focus, then press "
              "Enter to establish the reference... / "
              "把marker放好，确保光照/对焦良好，然后回车建立参照...")

        # Record the camera's **actual** exposure/gain at the moment the
        # truth is established (readable even under auto exposure)
        cam_state = self.read_camera_state()

        info = self.get_marker_pose()  # clean detection with confirmation
        if info is None:
            print("Reference detection failed. / 参照检测失败。")
            return None

        # Record the marker bounding box, used by the occlusion experiment
        c = info['corners_px']
        self._marker_bbox = (c[:, 0].min(), c[:, 1].min(),
                             c[:, 0].max(), c[:, 1].max())

        # Save the clean marker geometry from the undegraded detection;
        # the recovery experiment's correction needs it
        # (correction relies on marker-plane back-projection, so the
        # plane geometry must be accurate — only the clean detection can
        # be used)
        self._clean_marker_info = info

        p_cam_clean = info['p_cam']
        pred_clean = self.T_base_cam[:3, :3] @ p_cam_clean + \
            self.T_base_cam[:3, 3]
        pred_clean_mm = pred_clean * 1000.0

        dxy_history = []  # convergence record; stays empty in non-tactile mode or on touch failure

        if tactile_ground_truth:
            print("\nEstablishing the base-frame ground truth via closed-loop "
                  "visual correction (the arm will not move again after "
                  "this)... / "
                  "用闭环视觉纠偏自动建立base系真值"
                  "（之后机械臂不再移动）...")
            # For this touch that establishes ground truth, positioning
            # must use the tactile calibration, not the one selected by
            # calibration_mode (in classical mode that one is the
            # classical calibration, whose residual is an order of
            # magnitude coarser than the tactile version; using it to
            # position ground truth would easily get rejected for an
            # unreasonable Z prediction, or press into empty space —
            # this is exactly why the ground truth ended up empty last
            # time in classical mode and fell back to clean_vision).
            # Switch back immediately after the touch; which calibration
            # is used to predict for the subsequent 17 degradation
            # conditions is unaffected.
            saved_T_for_sweep = self.T_base_cam
            self.T_base_cam = self._T_touch_for_ref
            try:
                contact_pose, dxy_history = self.touch_marker_converged(
                    info, tol_mm=0.2, max_iters=5)
            finally:
                self.T_base_cam = saved_T_for_sweep
            if contact_pose is None:
                print("Touch failed, falling back to 'pure-vision reference' mode. / "
                      "触碰失败，退回到'纯视觉参照'模式。")
                p_base_true_mm = pred_clean_mm
                truth_source = "clean_vision"
                # Must also park at a safe height when the touch fails —
                # previously this branch was empty, leaving the arm
                # stranded at whatever hover position goto_xy_safely
                # left it at, unmanaged, potentially blocking the
                # camera's view of the marker for every subsequent
                # degradation condition.
                try:
                    cur = self._get_pose()
                    if cur is not None:
                        self._move_to([cur[0], cur[1],
                                       max(self.TRAVEL_Z_MM, cur[2] + 60.0)])
                        print(f"  Parked at safe height / 已停靠到安全高度 "
                              f"Z={max(self.TRAVEL_Z_MM, cur[2] + 60.0):.1f}mm.")
                except Exception as exc:
                    print(f"  Failed to park at safe height / 停靠安全高度失败：{exc}")
            else:
                p_base_true_mm = np.asarray(contact_pose[:3], float)
                truth_source = "tactile"
                # Raise to a safe height so it doesn't block the camera
                safe = np.asarray(contact_pose, float).copy()
                safe[2] = max(self.TRAVEL_Z_MM, safe[2] + 120.0)
                self._return_to_pose(safe)
        else:
            p_base_true_mm = pred_clean_mm
            truth_source = "clean_vision"

        cal_err = (np.linalg.norm(pred_clean_mm[:2] - p_base_true_mm[:2])
                   if truth_source == "tactile" else 0.0)

        # Camera acquisition conditions (truth is measured under "no
        # degradation": blur=0, noise=0, occlusion=0, exposure=auto;
        # what's read back below is the actual value at that moment
        # under auto exposure)
        exp = cam_state.get('exposure') if cam_state else None
        gain = cam_state.get('gain') if cam_state else None
        auto = cam_state.get('auto_exposure') if cam_state else None

        # ------- print the full truth values -------
        print("\n" + "=" * 58)
        print("REFERENCE (truth) ESTABLISHED")
        print("=" * 58)
        print(f"  truth source           : {truth_source}")
        print(f"  p_base_true X (mm)     : {p_base_true_mm[0]:.3f}")
        print(f"  p_base_true Y (mm)     : {p_base_true_mm[1]:.3f}")
        print(f"  p_base_true Z (mm)     : {p_base_true_mm[2]:.3f}")
        print(f"  p_cam (clean, m)       : "
              f"{np.round(p_cam_clean, 4).tolist()}")
        print(f"  clean-vision pred (mm) : {np.round(pred_clean_mm, 3).tolist()}")
        if truth_source == "tactile":
            print(f"  calib lateral err@good : {cal_err:.3f} mm")
            print(f"  ref-touch iterations   : {len(dxy_history)}  "
                  f"dXY history (mm): "
                  f"{[round(v, 3) for v in dxy_history]}")
        print("  --- camera acquisition conditions of the truth ---")
        print(f"  auto exposure          : {auto}")
        print(f"  exposure (actual)      : "
              f"{exp:.1f}" if exp is not None else "  exposure (actual)      : n/a")
        print(f"  gain (actual)          : "
              f"{gain:.1f}" if gain is not None else "  gain (actual)          : n/a")
        print("  blur                   : 0 (none)")
        print("  noise sigma            : 0 (none)")
        print("  occlusion              : 0 (none)")
        print("=" * 58)

        # ------- save to a standalone file for later reference -------
        ts = time.strftime("%Y%m%d_%H%M%S")
        ref_path = f"reference_truth_{ts}.txt"
        try:
            with open(ref_path, "w") as fh:
                fh.write("TactiCal visual-degradation experiment - REFERENCE (truth)\n")
                fh.write(f"timestamp            : {ts}\n")
                fh.write(f"truth_source         : {truth_source}\n")
                fh.write(f"p_base_true_x_mm     : {p_base_true_mm[0]:.4f}\n")
                fh.write(f"p_base_true_y_mm     : {p_base_true_mm[1]:.4f}\n")
                fh.write(f"p_base_true_z_mm     : {p_base_true_mm[2]:.4f}\n")
                fh.write(f"p_cam_clean_m        : {p_cam_clean.tolist()}\n")
                fh.write(f"clean_pred_mm        : {pred_clean_mm.tolist()}\n")
                fh.write(f"calib_lateral_err_mm : {cal_err:.4f}\n")
                fh.write(f"auto_exposure        : {auto}\n")
                fh.write(f"exposure_actual      : {exp}\n")
                fh.write(f"gain_actual          : {gain}\n")
                fh.write("blur                 : 0\n")
                fh.write("noise_sigma          : 0\n")
                fh.write("occlusion            : 0\n")
                fh.write(f"ref_touch_iters      : {len(dxy_history)}\n")
                fh.write(f"ref_touch_dxy_mm     : "
                         f"{[round(v, 3) for v in dxy_history]}\n")
            print(f"Reference saved to '{ref_path}'")
        except Exception as exc:
            print("Could not save reference file:", exc)

        return {
            'p_base_true_mm': p_base_true_mm,
            'pred_clean_mm': pred_clean_mm,
            'truth_source': truth_source,
            'truth_exposure': exp,
            'truth_gain': gain,
            'truth_auto_exposure': auto,
        }

    # ------------------------------------------------------------------
    # Main experiment
    # ------------------------------------------------------------------
    # Tactile recovery: a real touch + correction under degraded vision,
    # producing the "post-recovery error" (green line)
    # ------------------------------------------------------------------
    def recover_touch(self, p_cam_degraded, p_true_mm, pred_clean_mm,
                      max_lateral_mm=30.0, max_z_mm=20.0):
        """
        Given the p_cam produced by current (degraded) vision, has the
        arm perform one real touch and correction, and returns a dict:
          recovered_lateral_mm  residual lateral error after correction (<-- green line)
          touched               whether contact was made successfully
          skipped               whether the touch was skipped for exceeding the safety bound
          pre_correction_mm     the vision landing error before correction (should be ~ the red line, used as a self-check)

        Safety logic:
          1) If the degraded prediction is more than max_lateral_mm from
             the clean ground-truth XY, this is judged as "vision is off
             by too much, pressing down would be risky" and the touch is
             skipped (this is itself a strong result: once vision is off
             by a lot, even a safe touch becomes impossible, and only
             tactile/clean reference can serve as a backstop).
          1b) The marker never moves throughout, so Z should theoretically
             be constant; if the predicted Z deviates from the known
             clean Z by more than max_z_mm, this vision estimate is
             likewise judged untrustworthy and the press is skipped —
             this specifically guards against the case where "XY looks
             fine but the Z prediction is wildly off, causing overshoot
             or a collision" (which the XY check alone would not catch).
          2) The descent reuses descend_to_contact's z_floor safety limit.
          3) Correction uses the clean marker geometry from the undegraded
             detection (plane back-projection must be accurate).
          4) **Regardless of how far the process gets or whether an
             exception occurs, before the function returns the end
             effector is always parked at TRAVEL_Z_MM, the truly safe
             height** (not just a small lift off the contact surface),
             to ensure the camera's view of the marker isn't blocked
             for the next degradation condition. The small intermediate
             lift (RETRACT_MM) is only to allow a safe lateral move
             during correction/the second touch, not the final parked
             position.
        """
        result = {'recovered_lateral_mm': None, 'touched': False,
                  'skipped': False, 'pre_correction_mm': None}

        if self._clean_marker_info is None or self.T_base_cam is None:
            result['skipped'] = True
            return result

        pred_mm = (self.T_base_cam[:3, :3] @ p_cam_degraded +
                   self.T_base_cam[:3, 3]) * 1000.0

        # --- Safety check 1: skip the press if the prediction is too far from ground truth (XY) ---
        lat_from_truth = float(np.linalg.norm(pred_mm[:2] - p_true_mm[:2]))
        if lat_from_truth > max_lateral_mm:
            print(f"  [recover] Vision prediction is {lat_from_truth:.1f}mm from "
                  f"truth > {max_lateral_mm}mm — skipping the touch for "
                  f"safety. / "
                  f"视觉预测离真值{lat_from_truth:.1f}mm"
                  f">{max_lateral_mm}mm，为安全跳过下压。")
            result['skipped'] = True
            return result

        # --- Safety check 1b: also skip if the predicted Z is too far
        #     from the known clean Z (the marker never moves on its own,
        #     so Z is theoretically constant; a large deviation means
        #     this vision prediction is untrustworthy) ---
        z_from_clean = float(abs(pred_mm[2] - pred_clean_mm[2]))
        if z_from_clean > max_z_mm:
            print(f"  [recover] Predicted Z deviates from the clean reference by "
                  f"{z_from_clean:.1f}mm > {max_z_mm}mm — skipping the "
                  f"touch for safety. / "
                  f"视觉预测Z偏离干净参照"
                  f"{z_from_clean:.1f}mm>{max_z_mm}mm，为安全跳过下压。")
            result['skipped'] = True
            return result

        # Turn off the degradation injection; both correction and the
        # touch happen under the real (clean) image
        saved_syn = self._synthetic
        self._synthetic = None
        self.set_exposure(None)
        try:
            # Move above the degraded prediction point — pred_mm is
            # "where the DIGIT center should be" (if T_base_cam itself
            # was calibrated with corrected samples); the TCP's target
            # position must subtract the current correction offset,
            # otherwise descend_to_contact would add it a second time
            # when recording (this is exactly the root cause of that
            # constant bias exactly equal to the correction offset seen
            # earlier in unified_evaluation).
            move_target = pred_mm.copy()
            move_target[:2] -= self.digit_center_offset_mm
            if not self.goto_xy_safely(move_target):
                print("  [recover] Could not reach the hover point, skipping. / "
                      "无法到达悬停点，跳过。")
                result['skipped'] = True
                return result

            # Descend to contact (using real contact detection, backed by the safety floor)
            contact_pose, _, _ = self.descend_to_contact(pred_mm[2])
            if contact_pose is None:
                print("  [recover] No contact (vision error caused an empty press). / "
                      "未接触（视觉偏差导致压空）。")
                result['touched'] = False
                return result

            result['touched'] = True

            # Grab a frame (zero parallax, while still at contact height) -> lift slightly out of the way, then correct
            img = self._color_image()
            self._retract(self.RETRACT_MM)

            if img is None:
                print("  [recover] No camera frame, cannot correct. / "
                      "无相机帧，无法纠偏。")
                return result

            dxy, action = self.visual_xy_correction(
                img, self._clean_marker_info, self.T_base_cam)
            # Pre-correction error: changed to the distance between the
            # first touch's physical position (already with the DIGIT
            # center correction applied) and the independent ground
            # truth p_true_mm — this uses the same logic as the green
            # line, no longer relying on the approximation of the visual
            # difference-blob centroid. The dxy computed by
            # visual_xy_correction is now used only to decide "which
            # direction to correct and by how much" and is no longer
            # reported as an accuracy figure — but the correction
            # magnitude itself still needs its own variable for the
            # safety-valve check, and must not be replaced by pre_corr.
            corr_mag = float(np.linalg.norm(dxy))
            pre_corr = float(np.linalg.norm(
                np.asarray(contact_pose[:2], float) - p_true_mm[:2]))
            result['pre_correction_mm'] = pre_corr
            print(f"  [recover] Pre-correction landing error / 纠偏前落点误差 "
                  f"|dXY| = {pre_corr:.2f} mm")

            if action == 'abort':
                return result

            # Apply the correction and touch again
            if action == 'apply' and corr_mag < 60.0:
                self._move_relative([dxy[0], dxy[1], 0.0],
                                    speed=self.MOVE_SPEED)
                cp2, _, _ = self.descend_to_contact(contact_pose[2])
                self._retract(self.RETRACT_MM)
                if cp2 is not None:
                    # Green line: distance between the second touch's
                    # physical position (already with the DIGIT center
                    # correction applied) and the independent ground
                    # truth p_true_mm — a pure coordinate comparison, no
                    # longer dependent on whether the visual-correction
                    # step was clicked by a human or auto-detected; who
                    # decides "which way to correct" doesn't affect the
                    # objectivity of this number.
                    result['recovered_lateral_mm'] = float(
                        np.linalg.norm(
                            np.asarray(cp2[:2], float) - p_true_mm[:2]))
                else:
                    result['recovered_lateral_mm'] = None
                    result['touched'] = False
            else:
                # The operator accepted, or the correction was too large
                # and rejected: there is no second touch, so the
                # residual error is the distance between the first
                # touch's physical position and ground truth.
                result['recovered_lateral_mm'] = float(
                    np.linalg.norm(
                        np.asarray(contact_pose[:2], float) - p_true_mm[:2]))

            rec = result['recovered_lateral_mm']
            if rec is not None:
                print(f"  [recover] Post-correction residual error / 纠偏后残余误差 "
                      f"= {rec:.2f} mm  <-- green line / 绿线")
            return result

        finally:
            self._synthetic = saved_syn
            # Unconditionally park at a safe height: regardless of which
            # branch above was taken (abort/accept/apply succeeded/apply
            # rejected/pressed empty/no image), and regardless of
            # whether an exception occurred partway through, as long as
            # a descent ever happened, raise the end effector to
            # TRAVEL_Z_MM, the truly safe height (not the small
            # RETRACT_MM lift used mid-correction), before returning, to
            # ensure the camera's view of the marker isn't blocked for
            # the next degradation condition.
            try:
                cur = self._get_pose()
                if cur is not None:
                    self._move_to(
                        [cur[0], cur[1],
                         max(self.TRAVEL_Z_MM, cur[2] + 60.0)])
            except Exception:
                pass

    # ------------------------------------------------------------------
    def run(self, conditions, ref, n_detect=15, with_recovery=False,
            recover_families=('exposure', 'blur')):
        """
        conditions: list of dict, each dict describing one degradation
        level, e.g.
          {'family':'blur',      'level':11}
          {'family':'noise',     'level':20}
          {'family':'occlusion', 'level':0.3, 'side':'left'}
          {'family':'exposure',  'level':600, 'gain':64}
          {'family':'clean',     'level':0}
        """
        p_true = ref['p_base_true_mm']
        pred_clean = ref['pred_clean_mm']
        rows = []

        for k, cond in enumerate(conditions):
            fam = cond['family']
            lvl = cond.get('level', 0)
            print("\n" + "=" * 58)
            print(f"[{k+1}/{len(conditions)}] condition: {fam} = {lvl}")
            print("=" * 58)

            act_exp, act_gain = None, None
            self._synthetic = None
            self.set_exposure(None)

            if fam == 'clean':
                pass
            elif fam == 'blur':
                self._synthetic = make_blur(lvl)
            elif fam == 'noise':
                self._synthetic = make_noise(lvl, seed=k)
            elif fam == 'occlusion':
                self._synthetic = make_occlusion(
                    self._marker_bbox, lvl, cond.get('side', 'left'))
            elif fam == 'exposure':
                act_exp, act_gain = self.set_exposure(
                    lvl, cond.get('gain', None))
            else:
                print("unknown family, skipped.")
                continue

            rate, p_cam, std_mm = self.measure_vision(n=n_detect)

            row = {
                'family': fam, 'level': lvl,
                'side': cond.get('side', ''),
                'exposure_actual': act_exp if act_exp is not None else '',
                'gain_actual': act_gain if act_gain is not None else '',
                'detect_rate': round(rate, 3),
                'jitter_x_mm': '', 'jitter_y_mm': '', 'jitter_z_mm': '',
                'pred_x_mm': '', 'pred_y_mm': '', 'pred_z_mm': '',
                'lateral_err_vs_truth_mm': '',
                'z_err_vs_truth_mm': '',
                'lateral_drift_vs_clean_mm': '',
                'recovered_lateral_mm': '',   # green line: residual error after tactile correction
                'recover_touched': '',
                'recover_skipped': '',
            }

            if p_cam is None:
                print(f"  detection rate {rate:.2f} -> no valid detection, this "
                      f"condition is recorded as failed. / "
                      f"检测率{rate:.2f} -> 无有效检测，本档记为失败。")
                rows.append(row)
                continue

            pred = (self.T_base_cam[:3, :3] @ p_cam +
                    self.T_base_cam[:3, 3]) * 1000.0
            lat_truth = float(np.linalg.norm(pred[:2] - p_true[:2]))
            z_truth = float(pred[2] - p_true[2])
            lat_clean = float(np.linalg.norm(pred[:2] - pred_clean[:2]))

            row.update({
                'jitter_x_mm': round(float(std_mm[0]), 3),
                'jitter_y_mm': round(float(std_mm[1]), 3),
                'jitter_z_mm': round(float(std_mm[2]), 3),
                'pred_x_mm': round(float(pred[0]), 2),
                'pred_y_mm': round(float(pred[1]), 2),
                'pred_z_mm': round(float(pred[2]), 2),
                'lateral_err_vs_truth_mm': round(lat_truth, 3),
                'z_err_vs_truth_mm': round(z_truth, 3),
                'lateral_drift_vs_clean_mm': round(lat_clean, 3),
            })
            print(f"  detect rate      : {rate:.2f}")
            print(f"  pose jitter (mm) : {np.round(std_mm, 2)}")
            print(f"  lateral err vs TRUTH : {lat_truth:.2f} mm  "
                  f"<-- pure-vision error / 纯视觉误差")
            print(f"  lateral drift vs clean: {lat_clean:.2f} mm")

            # ---- Tactile recovery (optional): real touch + correction, producing the green line ----
            if with_recovery and fam in recover_families:
                print("  --- tactile recovery ---")
                rec = self.recover_touch(p_cam, p_true, pred_clean)
                row['recovered_lateral_mm'] = (
                    round(rec['recovered_lateral_mm'], 3)
                    if rec['recovered_lateral_mm'] is not None else '')
                row['recover_touched'] = int(rec['touched'])
                row['recover_skipped'] = int(rec['skipped'])

            rows.append(row)

        # Restore the camera
        self._synthetic = None
        self.set_exposure(None)
        try:
            cv2.destroyWindow("Degradation view")
        except cv2.error:
            pass

        return rows

    # ------------------------------------------------------------------
    def save_and_plot(self, rows, ref, tag=None):
        ts = time.strftime("%Y%m%d_%H%M%S")
        tag = tag or ts
        csv_path = f"visual_degradation_{tag}.csv"
        fields = list(rows[0].keys())
        with open(csv_path, "w", newline="") as fh:
            # Write the full truth values and acquisition conditions at
            # the top first (prefixed with # for human readability;
            # when reading with pandas, add comment='#' to skip it)
            t = ref['p_base_true_mm']
            fh.write(f"# TRUTH source={ref['truth_source']} "
                     f"p_base_true_mm=({t[0]:.3f},{t[1]:.3f},{t[2]:.3f}) "
                     f"exposure={ref.get('truth_exposure')} "
                     f"gain={ref.get('truth_gain')} "
                     f"auto_exposure={ref.get('truth_auto_exposure')} "
                     f"blur=0 noise=0 occlusion=0\n")
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
        print(f"\nCSV saved to '{csv_path}'")

        try:
            self._plot(rows, ref, tag)
        except Exception as exc:
            print("Plotting skipped:", exc)
        return csv_path

    def _plot(self, rows, ref, tag):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        families = [f for f in ['exposure', 'blur', 'noise', 'occlusion']
                    if any(r['family'] == f for r in rows)]
        if not families:
            return

        fig, axes = plt.subplots(2, len(families),
                                 figsize=(5 * len(families), 8),
                                 squeeze=False)

        for j, fam in enumerate(families):
            sub = [r for r in rows if r['family'] == fam]
            x = [float(r['level']) for r in sub]
            err = [r['lateral_err_vs_truth_mm']
                   if r['lateral_err_vs_truth_mm'] != '' else np.nan
                   for r in sub]
            rate = [r['detect_rate'] for r in sub]

            ax = axes[0][j]
            ax.plot(x, err, 'o-', color='tab:red',
                    label='pure vision (red)')

            # Overlay the tactile-recovery green line (if data exists)
            rec_x, rec_y = [], []
            for r in sub:
                v = r.get('recovered_lateral_mm', '')
                if v != '' and v is not None:
                    rec_x.append(float(r['level']))
                    rec_y.append(float(v))
            if rec_x:
                ax.plot(rec_x, rec_y, '^-', color='tab:green',
                        label='after tactile touch (green)')
                ax.legend(fontsize=8)

            ax.set_title(f"{fam}: pure-vision vs tactile-recovered error")
            ax.set_xlabel(fam + " level")
            ax.set_ylabel("lateral error vs tactile truth (mm)")
            ax.grid(True, alpha=0.3)

            ax2 = axes[1][j]
            ax2.plot(x, rate, 's-', color='tab:blue')
            ax2.set_ylim(-0.05, 1.05)
            ax2.set_title(f"{fam}: marker detection rate")
            ax2.set_xlabel(fam + " level")
            ax2.set_ylabel("detection success rate")
            ax2.grid(True, alpha=0.3)

        fig.suptitle("Vision degrades, tactile stays the anchor "
                     f"(truth source: {ref['truth_source']})",
                     fontsize=13)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        png = f"visual_degradation_{tag}.png"
        fig.savefig(png, dpi=130)
        print(f"Plot saved to '{png}'")


# ===========================================================================
if __name__ == "__main__":
    # ===== Calibration mode: decides which calibration matrix is used for prediction =====
    #   'touch'     -> tactile version (hand_eye_calib_*.npz)
    #   'classical' -> classical version (classical_handeye_*.npz)
    # Running it twice (once with touch, once with classical) gives two
    # degradation curves to compare.
    CALIBRATION_MODE = 'touch'      # or 'classical'
    CALIBRATION_FILE = None         # specify a particular file; None = auto-pick the newest

    # True  -> requires the robot arm, touches once to get ground truth (recommended, error is an absolute value)
    # False -> the arm never moves, only measures the drift introduced by degradation (use this to first get the pipeline running)
    TACTILE_GROUND_TRUTH = True

    N_DETECT = 15  # how many frames to detect per degradation level

    # Tactile recovery: performs a real touch and correction after each
    # change in degradation condition. Only meaningful for the tactile
    # version (the classical version is pure vision with no tactile
    # correction, so it is automatically disabled in classical mode).
    # Now covers all four degradation families: exposure, blur, noise,
    # occlusion — as long as the camera can still detect the marker at
    # that level (p_cam is not None), a touch will be performed. Levels
    # where detection fails outright (e.g. occlusion too large) have no
    # p_cam available for pressing down anyway, and are skipped
    # automatically.
    WITH_RECOVERY = True
    RECOVER_FAMILIES = ('exposure', 'blur', 'noise', 'occlusion')

    # The classical version has no tactile recovery, force the green line off
    if CALIBRATION_MODE == 'classical':
        WITH_RECOVERY = False
        print("[Classical mode / 经典模式] Tactile recovery (green line) "
              "disabled — pure vision has no tactile correction. / "
              "触觉恢复(绿线)已关闭 —— 纯视觉没有触觉纠偏。")

    # Degradation levels. Free to add/remove. Meaning of 'level' per family:
    #   exposure : exposure value (smaller = darker), optional gain
    #   blur     : Gaussian kernel size (pixels, automatically rounded to odd)
    #   noise    : Gaussian noise sigma (gray levels)
    #   occlusion: fraction of marker covered 0~1, side=left/right/top/bottom
    CONDITIONS = [
        {'family': 'clean',     'level': 0},

        {'family': 'exposure',  'level': 1100, 'gain': 64},
        {'family': 'exposure',  'level': 800,  'gain': 64},
        {'family': 'exposure',  'level': 500,  'gain': 64},
        {'family': 'exposure',  'level': 300,  'gain': 64},

        {'family': 'blur',      'level': 5},
        {'family': 'blur',      'level': 11},
        {'family': 'blur',      'level': 17},
        {'family': 'blur',      'level': 23},

        {'family': 'noise',     'level': 8},
        {'family': 'noise',     'level': 16},
        {'family': 'noise',     'level': 28},
        {'family': 'noise',     'level': 40},

        {'family': 'occlusion', 'level': 0.01, 'side': 'left'},
        {'family': 'occlusion', 'level': 0.02, 'side': 'left'},
        {'family': 'occlusion', 'level': 0.05, 'side': 'left'},
        {'family': 'occlusion', 'level': 0.10, 'side': 'left'},
    ]

    rig = None
    try:
        rig = DegradationRig(
            arm_ip='192.168.1.222',
            marker_size=0.1,
            marker_id=20,
            dry_run=False,
            use_previous_calibration=True,
            calibration_mode=CALIBRATION_MODE,
            calibration_file=CALIBRATION_FILE,
        )

        if rig.T_base_cam is None:
            raise SystemExit("No calibration available — please run the "
                          "matching calibration first. / "
                          "没有可用标定，先完成对应标定再来。")

        ref = rig.establish_reference(
            tactile_ground_truth=TACTILE_GROUND_TRUTH)
        if ref is None:
            raise SystemExit("Reference establishment failed. / 参照建立失败。")

        rows = rig.run(CONDITIONS, ref, n_detect=N_DETECT,
                       with_recovery=WITH_RECOVERY,
                       recover_families=RECOVER_FAMILIES)
        # Tag the output filename with the mode so two runs don't overwrite each other
        rig.save_and_plot(rows, ref,
                          tag=f"{CALIBRATION_MODE}_"
                              f"{time.strftime('%Y%m%d_%H%M%S')}")

        print("\n=== Summary (lateral error vs truth) ===")
        for r in rows:
            e = r['lateral_err_vs_truth_mm']
            print(f"  {r['family']:10s} {str(r['level']):>6}  "
                  f"rate={r['detect_rate']:.2f}  "
                  f"err={e if e != '' else 'FAIL':>8}")

        print("\nDone. For the paper's main figure, use the top-row "
              "'lateral error' curves. / "
              "完成。论文主图用第一行的'lateral error'曲线：")
        print("As vision degrades, its error grows, while the touch-derived "
              "truth stays fixed — this is the robustness pure vision "
              "alone cannot provide. / "
              "视觉越差误差越大，而触碰得到的真值始终不变 —— "
              "这就是纯视觉方法给不了的鲁棒性。")

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    except Exception as exc:
        print(f"\nError: {exc}")
        import traceback
        traceback.print_exc()
    finally:
        if rig is not None:
            try:
                rig.set_exposure(None)  # be sure to restore auto exposure
            except Exception:
                pass
            rig.shutdown()
