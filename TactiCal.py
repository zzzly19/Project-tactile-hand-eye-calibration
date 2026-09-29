#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TactiCal v2 - Tactile-Assisted Hand-Eye Calibration
====================================================

Main changes relative to v1 (each one addresses a problem found in real testing):

[Z stability]
 1. The slow-down zone is now anchored to the "predicted contact height" instead of the
    "maximum travel distance". In v1, max_down=100mm and SLOW_ZONE=15mm meant the first 85mm
    were all taken in big 4mm steps, while the predicted contact point was at 60mm -- i.e. the
    step that actually hits the marker was guaranteed to be one of the big 4mm steps. That was
    the direct cause of "too much force".
 2. The contact criterion changed from "mean absolute difference over the whole image" to
    "contact area fraction after removing the global drift". The mean difference gets pushed up
    overall by LED drift / gel rebound, so v1's threshold max(0.84, noise_mean + 1*noise_std)
    sat almost right on the noise floor and triggered false positives easily -- that was the
    cause of "retracting before actually touching".
 3. After contact is detected, the arm doesn't stop immediately; it keeps pressing in 0.15mm
    steps until a fixed contact area fraction (PRESS_TARGET_AREA_PCT) is reached, giving a
    repeatable indentation amount -> repeatable Z.
 4. A fresh baseline is captured right before entering the slow zone, to remove sensor drift
    accumulated during the move.

[XY centering]
 5. During manual jogging, a **live camera feed** is now shown alongside the DIGIT view, with
    the marker center drawn as a crosshair. v1's jog only looked at the DIGIT window, which is
    blank until contact -- effectively jogging blind.
 6. Added a "post-contact visual correction" closed loop: at contact, grab one camera frame ->
    retract -> the operator clicks the tool's actual landing point in a zoomed window ->
    back-project onto the marker plane -> convert to a base-frame XY correction -> press down
    again. 1-2 iterations typically get within 1-2mm.
 7. Allows using the previous calibration result for **coarse positioning** (this was commented
    out in v1), but it must still go through the visual correction in step 6, otherwise the new
    calibration would just copy the old one (see _warn_fixed_point).
 8. XY moves now go through three segments -- "lift -> translate -> descend" -- instead of
    sweeping diagonally across the marker.

[Diagnostics]
 9. Scaled Umeyama is used as a diagnostic: if the recovered scale deviates noticeably from 1,
    it indicates marker_size was entered wrong or the camera intrinsic focal length is off.
10. Leave-one-out cross-validation (LOO), which gives a much more honest accuracy estimate than
    the fit residual RMSE.

Author's note: the contact pose TOUCH_* must stay fixed for the entire run, so that the
flange->gel offset is a constant that can be absorbed into the translation solution t.
"""

import os
import glob
import time

import cv2
import numpy as np
import pyrealsense2 as rs
from scipy.spatial.transform import Rotation as R
from xarm.wrapper import XArmAPI
from digit_interface import Digit


# ===========================================================================
# ArUco robust-lighting parameters (shared by TactiCal and the classic
# calibration script, so the comparison between them is fair)
# ===========================================================================
def _apply_robust_aruco_params(p):
    """
    Tune the ArUco detector parameters to be more tolerant of uneven lighting,
    glare, and low contrast. Every attribute is wrapped in try/except so this
    stays compatible across OpenCV versions (whichever attribute is missing
    is simply skipped).

    The core idea is adaptive thresholding: instead of one global threshold,
    each local window adapts on its own, so the checkerboard cells can still
    be separated even when half the image is bright and half is dark, or
    there's a glare spot.
    """
    def setattr_safe(name, value):
        try:
            setattr(p, name, value)
        except Exception:
            pass

    # Adaptive threshold window: sweep from small to large sizes to adapt to
    # lighting unevenness at different scales
    setattr_safe('adaptiveThreshWinSizeMin', 3)
    setattr_safe('adaptiveThreshWinSizeMax', 43)   # default is 23; increased to handle wide-range lighting gradients
    setattr_safe('adaptiveThreshWinSizeStep', 8)
    setattr_safe('adaptiveThreshConstant', 7)

    # Widen the marker size range so slight contour changes from lighting don't cause misses
    setattr_safe('minMarkerPerimeterRate', 0.02)
    setattr_safe('maxMarkerPerimeterRate', 4.0)

    # Relax the polygon approximation accuracy to tolerate edges blurred by glare/defocus
    setattr_safe('polygonalApproxAccuracyRate', 0.05)

    # Lower the minimum corner distance requirement so corners survive under low contrast
    setattr_safe('minCornerDistanceRate', 0.05)
    setattr_safe('minMarkerDistanceRate', 0.05)

    # Shrink the border-ignore margin so markers near the image edge can still be detected
    setattr_safe('minDistanceToBorder', 1)

    # Sub-pixel corner refinement (fallback in case the above attributes failed to set)
    try:
        p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        p.cornerRefinementWinSize = 5
        p.cornerRefinementMaxIterations = 40
        p.cornerRefinementMinAccuracy = 0.01
    except Exception:
        pass

    # False-detection rejection: relax the error tolerance a bit so real markers
    # aren't discarded as noise under weak lighting
    setattr_safe('maxErroneousBitsInBorderRate', 0.4)
    setattr_safe('errorCorrectionRate', 0.6)


# ===========================================================================
# Point-set registration
# ===========================================================================
def kabsch_umeyama(points_src, points_dst, with_scale=False):
    """
    Solve for the rigid transform (R, t) that best maps R @ p_src + t to p_dst.
    When with_scale=True, also return the scale s (diagnostic only -- the
    calibration itself does not use scale).
    """
    P = np.asarray(points_src, dtype=float)
    Q = np.asarray(points_dst, dtype=float)
    if P.shape != Q.shape or P.shape[0] < 3:
        raise ValueError("Need at least 3 matched 3D point pairs.")

    cp, cq = P.mean(axis=0), Q.mean(axis=0)
    X, Y = P - cp, Q - cq

    H = X.T @ Y
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    R_mat = Vt.T @ D @ U.T

    if with_scale:
        var_p = float(np.sum(X ** 2))
        scale = float(np.sum(S * np.array([1.0, 1.0, d]))) / max(var_p, 1e-12)
        t_vec = cq - scale * (R_mat @ cp)
        return R_mat, t_vec, scale

    t_vec = cq - R_mat @ cp
    return R_mat, t_vec


def check_point_geometry(points):
    """Check whether the sampled geometry is degenerate (collinear / coplanar)."""
    P = np.asarray(points, dtype=float)
    sv = np.linalg.svd(P - P.mean(axis=0), compute_uv=False) * 1000.0
    print(f"Sample geometry singular values (mm): "
          f"{sv[0]:.1f}, {sv[1]:.1f}, {sv[2]:.1f}")
    if sv[1] < 30.0:
        print("  WARNING: 采样点接近共线，请在 X/Y 上拉开距离。")
    if sv[2] < 10.0:
        print("  NOTE: 采样点接近共面。建议 1~2 次把 marker 垫高 "
              "(书/盒子, 30~80mm)，Z 方向和绕 X/Y 的旋转才有约束。")


def loo_cross_validation(points_cam, points_base):
    """
    Leave-one-out cross-validation: solve with N-1 points, predict the Nth point.
    This is an honest estimate of "can it align at a brand-new position",
    which is far more meaningful than the fit residual RMSE (fitting 6 DOF
    with 4 points is bound to make the residual look artificially small).
    """
    n = len(points_cam)
    if n < 4:
        print("LOO cross-validation needs >= 4 samples. Skipped.")
        return None

    errs = []
    for i in range(n):
        idx = [j for j in range(n) if j != i]
        try:
            R_i, t_i = kabsch_umeyama(points_cam[idx], points_base[idx])
        except np.linalg.LinAlgError:
            continue
        pred = R_i @ points_cam[i] + t_i
        errs.append((pred - points_base[i]) * 1000.0)

    if not errs:
        return None
    errs = np.array(errs)
    norms = np.linalg.norm(errs, axis=1)
    lateral = np.linalg.norm(errs[:, :2], axis=1)
    print("\nLeave-one-out cross-validation (真实泛化精度):")
    for i, (e, nn) in enumerate(zip(errs, norms)):
        print(f"  hold-out {i+1}: dX={e[0]:+7.2f} dY={e[1]:+7.2f} "
              f"dZ={e[2]:+7.2f}  |e|={nn:6.2f} mm")
    print(f"  LOO mean |e|      : {norms.mean():.2f} mm")
    print(f"  LOO mean XY error : {lateral.mean():.2f} mm  <-- 手眼精度")
    return float(norms.mean())


# ===========================================================================
# DIGIT contact detection
# ===========================================================================
class DigitContactDetector:
    """
    Area-based contact criterion.

    Compared to v1's mean(absdiff):
      - First subtract the median difference over the whole image, to remove
        global brightness drift caused by LED / temperature changes;
      - Only count the area fraction of "significantly changed pixels", so
        small localized contacts are still detected cleanly;
      - A morphological opening removes salt-and-pepper noise;
      - Also returns the centroid of the contact patch, which can tell
        whether the press landed on a marker edge.
    """

    def __init__(self, digit,
                 pix_diff=8,            # per-pixel grayscale difference threshold
                 blur=5,
                 roi_margin=0.08,       # crop off 8% of the image border (vignette/distortion region)
                 area_on_pct=1.0,       # lower bound on contact area fraction (%)
                 sigma=6.0,             # adaptive threshold = mean + sigma*std
                 press_target_pct=4.0,  # keep pressing until this area fraction is reached
                 max_sane_threshold_pct=25.0):  # baseline is considered unusable above this threshold
        self.digit = digit
        self.pix_diff = pix_diff
        self.blur = blur if blur % 2 == 1 else blur + 1
        self.roi_margin = roi_margin
        self.area_on_pct = area_on_pct
        self.sigma = sigma
        self.press_target_pct = press_target_pct
        self.max_sane_threshold_pct = max_sane_threshold_pct

        self.baseline = None
        self.thr_on = area_on_pct
        self.window = "DIGIT with Crosshair"

    # ------------------------------------------------------------------
    def _grab(self, tries=6):
        for _ in range(tries):
            f = self.digit.get_frame()
            if f is not None:
                return f
            time.sleep(0.02)
        return None

    def _prep(self, frame):
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        g = cv2.GaussianBlur(g, (self.blur, self.blur), 0)
        h, w = g.shape
        m = self.roi_margin
        y0, y1 = int(h * m), int(h * (1 - m))
        x0, x1 = int(w * m), int(w * (1 - m))
        return g[y0:y1, x0:x1].astype(np.int16), (x0, y0)

    def _analyse(self, gray_roi):
        """Returns (area_pct, mask, centroid_or_None)."""
        d = gray_roi - self.baseline
        d = d - np.median(d)                      # remove global drift
        mask = (np.abs(d) > self.pix_diff).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                                np.ones((3, 3), np.uint8))
        area_pct = 100.0 * float(mask.sum()) / mask.size

        centroid = None
        if mask.sum() > 20:
            ys, xs = np.nonzero(mask)
            centroid = (float(xs.mean()), float(ys.mean()))
        return area_pct, mask, centroid

    # ------------------------------------------------------------------
    def capture_baseline(self, num_frames=30, show=True, status="baseline"):
        """Capture the baseline while known to be out of contact; also estimate noise and set the threshold."""
        grays = []
        for _ in range(num_frames):
            f = self._grab()
            if f is None:
                continue
            if show:
                self.show(f, status_text=f"{status}\n(no contact expected)")
            g, _ = self._prep(f)
            grays.append(g)
            time.sleep(0.02)

        if len(grays) < 10:
            print("DIGIT baseline failed: not enough frames.")
            return False

        self.baseline = np.median(np.stack(grays, axis=0), axis=0)
        areas = [self._analyse(g)[0] for g in grays]
        mean, std = float(np.mean(areas)), float(np.std(areas))
        self.thr_on = max(self.area_on_pct, mean + self.sigma * std)
        print(f"  DIGIT baseline: noise area {mean:.3f}% +/- {std:.3f}%  "
              f"-> contact threshold {self.thr_on:.3f}%")
        if mean > 0.8:
            print("  WARNING: 无接触时噪声面积偏大，凝胶可能已经受力，"
                  "或者 baseline 采集时机械臂还在动。")
        if self.thr_on > self.max_sane_threshold_pct:
            print(f"  REJECTED: 阈值 {self.thr_on:.1f}% 超过合理上限 "
                  f"{self.max_sane_threshold_pct:.1f}%，接触面积"
                  "永远不可能触发这个阈值——基线本身不可用（凝胶受力/"
                  "振动未平息），拒绝继续下压。")
            return False
        return True

    def measure(self, status=""):
        """Returns (area_pct, centroid, frame)."""
        f = self._grab()
        if f is None or self.baseline is None:
            return 0.0, None, None
        g, off = self._prep(f)
        area, mask, c = self._analyse(g)
        self.show(f, status_text=status, mask=mask, mask_off=off,
                  area=area)
        return area, c, f

    def contact_offset_mm(self, centroid, gel_mm=(18.6, 14.3)):
        """
        Offset of the contact patch centroid (or any pixel point, e.g. a
        detected corner vertex) relative to the gel center (mm, gel frame).
        gel_mm is the effective sensing area size of the DIGIT; update it
        to match your actual calibration.
        """
        if centroid is None or self.baseline is None:
            return None
        h, w = self.baseline.shape
        du = (centroid[0] - w / 2.0) / w * gel_mm[0]
        dv = (centroid[1] - h / 2.0) / h * gel_mm[1]
        return np.array([du, dv])

    def detect_corner(self, mask, margin_px=4,
                      angle_tol_deg=35.0):
        """
        From the deformation mask, find the marker corner's vertex position
        and the angular deviation of its two edges relative to the
        horizontal/vertical axes -- this is specifically for asymmetric
        contact where the press lands on a marker corner, which is a
        different use case from the centroid used in _analyse for a press
        centered on the flat area.

        Method: take the largest contour, simplify it into a polygon with
        approxPolyDP, and find the vertex closest to a right angle (90 deg)
        that isn't sitting right on the crop boundary; that vertex is taken
        to be the marker corner's position in the image.

        Returns (corner_px, edge_deviation_deg, confidence):
          corner_px           (x, y) pixel coordinates; None if no clear corner was found
          edge_deviation_deg  average angular deviation (degrees) of the two
                              edges relative to the crosshair's horizontal/
                              vertical lines; for manual/downstream orientation
                              judgement only -- orientation is not auto-corrected here
          confidence          0-1; closer to 1 means the contour looks more
                              like a clean right angle; 0.0 if none was found
        """
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, None, 0.0
        c = max(contours, key=cv2.contourArea)
        if cv2.contourArea(c) < 50:
            return None, None, 0.0

        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True).reshape(-1, 2)
        h, w = mask.shape
        n = len(approx)

        best, best_score = None, -1.0
        for i in range(n):
            p_prev = approx[(i - 1) % n].astype(float)
            p_cur = approx[i].astype(float)
            p_next = approx[(i + 1) % n].astype(float)
            if not (margin_px < p_cur[0] < w - margin_px
                   and margin_px < p_cur[1] < h - margin_px):
                continue  # too close to the crop boundary, likely the ROI edge rather than a real marker edge
            v1, v2 = p_prev - p_cur, p_next - p_cur
            n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
            if n1 < 5 or n2 < 5:
                continue
            cos_ang = np.dot(v1, v2) / (n1 * n2)
            ang = np.degrees(np.arccos(np.clip(cos_ang, -1, 1)))
            score = -abs(ang - 90.0)  # closer to 90 degrees scores higher
            if score > best_score:
                best_score = score
                best = (p_cur, v1 / n1, v2 / n2, ang)

        if best is None or -best_score > angle_tol_deg:
            return None, None, 0.0

        p_cur, u1, u2, ang = best

        def axis_dev(u):
            deg = np.degrees(np.arctan2(u[1], u[0])) % 90.0
            return min(deg, 90.0 - deg)

        edge_dev = (axis_dev(u1) + axis_dev(u2)) / 2.0
        confidence = max(0.0, 1.0 - abs(ang - 90.0) / angle_tol_deg)
        return (float(p_cur[0]), float(p_cur[1])), float(edge_dev), \
            float(confidence)

    # ------------------------------------------------------------------
    def show(self, frame=None, status_text="", mask=None, mask_off=(0, 0),
             area=None):
        if frame is None:
            frame = self._grab()
        if frame is None:
            return None

        vis = frame.copy()
        h, w = vis.shape[:2]
        cx, cy = w // 2, h // 2

        # No longer overlaying any mask-based visualization (color blobs/contour
        # lines) -- the gel's own color/texture change under pressure is
        # already the most direct information, and any overlay just makes it
        # harder to see the real indentation shape. Only the crosshair is
        # kept as an alignment reference.
        cv2.line(vis, (0, cy), (w - 1, cy), (0, 255, 0), 1)
        cv2.line(vis, (cx, 0), (cx, h - 1), (0, 255, 0), 1)
        cv2.circle(vis, (cx, cy), 4, (0, 0, 255), -1)

        lines = status_text.split("\n") if status_text else []
        if area is not None:
            lines.append(f"area {area:5.2f}%  thr {self.thr_on:5.2f}%")
        for i, line in enumerate(lines):
            cv2.putText(vis, line, (6, 16 + 16 * i),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1)

        cv2.imshow(self.window, vis)
        cv2.waitKey(1)
        return frame

    def close_window(self):
        try:
            cv2.destroyWindow(self.window)
        except cv2.error:
            pass


# ===========================================================================
class HandEyeCalibrator:

    # ------------------------------------------------------------------
    # Tunable parameters
    # ------------------------------------------------------------------
    # Fixed contact pose (must not change for the entire run)
    TOUCH_ROLL = -179.797708
    TOUCH_PITCH = 0.320627
    TOUCH_YAW = 48.432867

    # ---- Motion ----
    TRAVEL_Z_MM = 180.0          # safe height for XY translation moves
    APPROACH_HEIGHT_MM = 40.0    # hover height above the predicted contact point
    PRE_CONTACT_MARGIN_MM = 12.0 # switch to slow speed once this close to the predicted contact point
    MAX_EXTRA_DOWN_MM = 25.0     # how much further than predicted the arm is allowed to press down
    MIN_SAFE_Z_ABS_MM = -5.0     # absolute lower bound in the base frame
    MAX_TARGET_DISTANCE_MM = 850.0
    RETRACT_MM = 15.0            # lift height used during visual correction

    # When lightly touching the marker, base-frame Z should be roughly around
    # this value -- used to sanity-check "is this predicted Z plausible" before
    # actually pressing down, so a bad prediction is rejected up front instead
    # of only being discovered after moving there.
    EXPECTED_CONTACT_Z_MM = 2.5
    CONTACT_Z_SANITY_TOL_MM = 5.0

    # When calibrating on a corner press, the deformation right after crossing
    # the contact threshold is too shallow to show a clean right-angle contour;
    # press a bit further to this area fraction before running corner analysis.
    # This is slightly higher than PRESS_TARGET_AREA_PCT (4%, used for flat-face
    # presses) but still far short of "pressed all the way through/flat".
    CORNER_READ_AREA_PCT = 10.0

    DOWN_STEP_FAST_MM = 3.0
    DOWN_STEP_SLOW_MM = 0.3
    DOWN_STEP_PRESS_MM = 0.15
    MAX_INDENT_MM = 2.0          # max additional press depth after contact is detected

    MOVE_SPEED = 60
    DESCEND_SPEED_FAST = 15
    DESCEND_SPEED_SLOW = 3

    # ---- Contact ----
    CONTACT_PIX_DIFF = 8
    CONTACT_AREA_ON_PCT = 1.0
    CONTACT_SIGMA = 6.0
    PRESS_TARGET_AREA_PCT = 4.0
    CONTACT_CONFIRM_FRAMES = 3

    # ---- Visual correction ----
    ALIGN_MAX_ITERS = 3
    ALIGN_TOL_MM = 1.5

    # ------------------------------------------------------------------
    def __init__(self, arm_ip='192.168.1.222', marker_size=0.1,
                 marker_id=20, dry_run=False,
                 use_previous_calibration=True,
                 calib_file='camera_calibration.npz'):
        self.dry_run = dry_run
        self.marker_id = marker_id
        self.marker_size = marker_size

        # ---------------- xArm ----------------
        self.arm = XArmAPI(arm_ip)
        self.arm.connect()
        self.arm.clean_warn()
        self.arm.clean_error()
        self.arm.motion_enable(True)
        self.arm.set_mode(0)
        self.arm.set_state(0)
        time.sleep(0.5)
        self._report_tcp_offset()

        # ---------------- DIGIT ----------------
        self.digit = Digit("D20978")
        self.digit.connect()
        self.digit.set_fps(30)
        self.contact = DigitContactDetector(
            self.digit,
            pix_diff=self.CONTACT_PIX_DIFF,
            area_on_pct=self.CONTACT_AREA_ON_PCT,
            sigma=self.CONTACT_SIGMA,
            press_target_pct=self.PRESS_TARGET_AREA_PCT,
        )

        # ---------------- RealSense ----------------
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        self.pipeline.start(config)

        cd = np.load(calib_file)
        self.camera_matrix = cd['mtx']
        self.dist_coeffs = cd['dist']
        print("Camera matrix:\n", np.round(self.camera_matrix, 2))

        # ---------------- ArUco ----------------
        self.detector = None
        try:
            self.aruco_dict = cv2.aruco.getPredefinedDictionary(
                cv2.aruco.DICT_5X5_1000)
            self.parameters = cv2.aruco.DetectorParameters()
            self.detector = cv2.aruco.ArucoDetector(self.aruco_dict,
                                                    self.parameters)
        except AttributeError:
            self.aruco_dict = cv2.aruco.Dictionary_get(
                cv2.aruco.DICT_5X5_1000)
            self.parameters = cv2.aruco.DetectorParameters_create()
        try:
            self.parameters.cornerRefinementMethod = \
                cv2.aruco.CORNER_REFINE_SUBPIX
        except AttributeError:
            pass
        # More robust lighting/glare tolerance parameters (kept consistent
        # with the classic calibration script, so the comparison is fair)
        _apply_robust_aruco_params(self.parameters)

        # ---------------- State ----------------
        self.T_base_cam = None
        self.T_base_cam_guess = (self._load_previous_calibration()
                                 if use_previous_calibration else None)
        # Systematic offset between "the DIGIT's actual center" and "its
        # registered TCP value", measured via the corner-press calibration
        # (base-frame XY, mm). This is a fact about the DIGIT hardware itself
        # and is independent of "whether to use the old T_base_cam as the
        # initial guess" -- regardless of whether use_previous_calibration is
        # False (e.g. UnifiedEvaluator deliberately sets it False and loads
        # T_touch/T_classical manually itself), as long as the DIGIT center
        # has been calibrated before, it should be loaded and applied
        # unconditionally. Defaults to 0 when no calibration/save file is
        # found, matching the previous behavior exactly.
        self.digit_center_offset_mm = self._load_digit_center_offset()
        self.applied_corrections_mm = []   # used for the fixed-point check
        self._click = None

    # ==================================================================
    def _report_tcp_offset(self):
        """
        Different versions of the xArm-Python-SDK expose the TCP offset
        differently: newer versions have the arm.tcp_offset attribute, while
        only older versions have get_tcp_offset(). Failing to read it doesn't
        affect calibration -- it just means one less diagnostic line.
        """
        off = None
        try:
            off = getattr(self.arm, 'tcp_offset', None)
            if off is None and hasattr(self.arm, 'get_tcp_offset'):
                ret = self.arm.get_tcp_offset()
                off = ret[1] if isinstance(ret, (list, tuple)) and \
                    len(ret) == 2 else ret
        except Exception as exc:
            print("Could not read TCP offset:", exc)

        if off is None:
            print("TCP offset: 本版本 SDK 未提供读取接口（不影响标定）。")
        else:
            print(f"xArm TCP offset: {np.round(np.asarray(off, float), 3)}")
        print("  NOTE: TCP offset 在整个标定+使用期间必须保持不变，"
              "否则记录的 p_base 含义会变。")

    def _load_previous_calibration(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        files = sorted(
            {p for d in (os.getcwd(), script_dir)
             for p in glob.glob(os.path.join(d, "hand_eye_calib_*.npz"))},
            key=os.path.getmtime)
        if not files:
            print("No previous calibration found -> 第一次采样用手动 jog。")
            return None
        try:
            guess = np.load(files[-1])["T_base_cam"]
            print(f"Loaded T_base_cam_guess from: {files[-1]}")
            print(np.array2string(guess, precision=5, suppress_small=True))
            return guess
        except Exception as exc:
            print("Failed to load previous calibration:", exc)
            return None

    def _load_digit_center_offset(self):
        """
        Load the DIGIT center offset saved by the last run of
        calibrate_digit_center_at_corner. If the saved file was measured
        using a calibration noticeably different from the current
        T_base_cam_guess, it becomes unclear how much of that offset is
        "the DIGIT's real mechanical offset" versus "error from that
        particular calibration run" -- so a warning is printed recommending
        a fresh corner calibration, but the offset is still loaded and used
        (better than no correction at all, just with a question mark over
        its precision).
        """
        script_dir = os.path.dirname(os.path.abspath(__file__))
        files = sorted(
            {p for d in (os.getcwd(), script_dir)
             for p in glob.glob(os.path.join(d, "digit_center_offset_*.npz"))},
            key=os.path.getmtime)
        if not files:
            print("No digit-center offset found -> 暂不对触碰位置做"
                  "任何修正（等同于以前的行为）。")
            return np.zeros(2)
        try:
            data = np.load(files[-1])
            offset = np.asarray(data['offset_mm'], float)
            print(f"Loaded digit_center_offset from: {files[-1]}  "
                  f"offset=({offset[0]:+.3f}, {offset[1]:+.3f}) mm")
            if self.T_base_cam_guess is not None and \
                    'T_base_cam_used' in data:
                saved_T = np.asarray(data['T_base_cam_used'], float)
                drift = float(np.linalg.norm(
                    saved_T[:3, 3] - self.T_base_cam_guess[:3, 3]) * 1000.0)
                if drift > 2.0:
                    print(f"  WARNING: 当前标定和测这个偏移时用的标定，"
                          f"平移部分相差 {drift:.1f}mm——这个偏移可能已"
                          "经不准了，建议重新跑一次"
                          "calibrate_digit_center_at_corner。")
            return offset
        except Exception as exc:
            print("Failed to load digit_center_offset:", exc)
            return np.zeros(2)

    def _save_digit_center_offset(self, offset_mm):
        ts = time.strftime("%Y%m%d_%H%M%S")
        fname = f"digit_center_offset_{ts}.npz"
        T_used = (self.T_base_cam if self.T_base_cam is not None
                 else self.T_base_cam_guess)
        np.savez(fname, offset_mm=np.asarray(offset_mm, float),
                 T_base_cam_used=T_used if T_used is not None
                 else np.eye(4))
        print(f"digit_center_offset 已存到 '{fname}'，之后每次启动会"
              "自动加载并应用到所有触碰上。")

    # ==================================================================
    # Camera / marker
    # ==================================================================
    def _color_image(self, timeout_ms=3000):
        try:
            frames = self.pipeline.wait_for_frames(timeout_ms)
        except RuntimeError:
            return None
        cf = frames.get_color_frame()
        if not cf:
            return None
        return np.asanyarray(cf.get_data())

    def _color_image_nonblocking(self, last=None):
        frames = self.pipeline.poll_for_frames()
        if frames:
            cf = frames.get_color_frame()
            if cf:
                return np.asanyarray(cf.get_data())
        return last

    def _detect_once(self, gray, marker_id=None):
        target_id = self.marker_id if marker_id is None else marker_id
        if self.detector is not None:
            corners, ids, _ = self.detector.detectMarkers(gray)
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(
                gray, self.aruco_dict, parameters=self.parameters)
        if ids is None:
            return None
        matches = np.where(ids.flatten() == target_id)[0]
        if len(matches) == 0:
            return None
        pts = corners[int(matches[0])].reshape(-1, 2).astype(np.float32)
        term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001)
        cv2.cornerSubPix(gray, pts, (5, 5), (-1, -1), term)
        return pts

    def _solve_marker_pose(self, image_corners):
        half = self.marker_size / 2.0
        obj = np.array([[-half, half, 0.0], [half, half, 0.0],
                        [half, -half, 0.0], [-half, -half, 0.0]],
                       dtype=np.float32)
        flags = getattr(cv2, "SOLVEPNP_IPPE_SQUARE", cv2.SOLVEPNP_ITERATIVE)
        ok, rvec, tvec = cv2.solvePnP(
            obj, image_corners.reshape(-1, 1, 2).astype(np.float32),
            self.camera_matrix, self.dist_coeffs, flags=flags)
        if not ok:
            return None, None
        return rvec.reshape(3), tvec.reshape(3)

    def marker_corner_p_cam(self, marker_info, corner_index):
        """
        From the info returned by get_marker_pose(), compute the 3D
        coordinates (meters) of one of the four corners in the camera
        frame -- used as input for calibrate_digit_center_at_corner.

        corner_index follows the standard ArUco corner ordering (matching
        the marker local frame defined in _solve_marker_pose):
          0 = top-left (-half, +half, 0)
          1 = top-right (+half, +half, 0)
          2 = bottom-right (+half, -half, 0)
          3 = bottom-left (-half, -half, 0)

        This uses marker_info['T_cam_marker'] (the stable multi-frame
        averaged pose), not a single-frame back-projection of the detected
        pixel corners, so it's more stable.
        """
        half = self.marker_size / 2.0
        obj_corners = np.array([
            [-half, half, 0.0],
            [half, half, 0.0],
            [half, -half, 0.0],
            [-half, -half, 0.0],
        ], dtype=np.float64)
        if not (0 <= corner_index < 4):
            raise ValueError("corner_index 必须是 0/1/2/3。")
        T = marker_info['T_cam_marker']
        return T[:3, :3] @ obj_corners[corner_index] + T[:3, 3]

    def project(self, p_cam):
        """Camera-frame 3D point -> pixel."""
        img, _ = cv2.projectPoints(np.asarray(p_cam, float).reshape(1, 1, 3),
                                   np.zeros(3), np.zeros(3),
                                   self.camera_matrix, self.dist_coeffs)
        return img.reshape(2)

    def backproject_to_plane(self, px, plane_point, plane_normal):
        """
        Pixel -> ray -> intersection with the marker plane (camera frame, meters).
        The core of the correction: at the moment of contact, the tool tip is
        exactly on the marker plane, so there is zero parallax.
        """
        pts = np.array([[[float(px[0]), float(px[1])]]], dtype=np.float64)
        und = cv2.undistortPoints(pts, self.camera_matrix, self.dist_coeffs)
        ray = np.array([und[0, 0, 0], und[0, 0, 1], 1.0])
        denom = float(plane_normal @ ray)
        if abs(denom) < 1e-9:
            return None
        s = float(plane_normal @ np.asarray(plane_point, float)) / denom
        return s * ray

    def get_marker_pose(self, require_confirmation=True,
                        num_average=12, max_attempts=150, marker_id=None):
        """
        Returns a dict:
          p_cam        marker center (camera frame, m)
          rvec, normal marker plane pose / normal
          corners_px   the four corner pixels
          centre_px    center pixel
          clean_image  a clean image with no arm occlusion (used later for diff-based prompts)
        When marker_id is not passed, self.marker_id is used (keeps the
        calibration flow unchanged).
        """
        target_id = self.marker_id if marker_id is None else marker_id
        while True:
            samples_t, samples_r = [], []
            display, last_corners = None, None

            for attempt in range(max_attempts):
                img = self._color_image()
                if img is None:
                    continue
                gray = np.ascontiguousarray(
                    cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), dtype=np.uint8)
                pts = self._detect_once(gray, marker_id=target_id)
                if pts is None:
                    if require_confirmation:
                        prev = img.copy()
                        cv2.putText(prev, f"Looking for marker "
                                    f"{target_id} "
                                    f"({attempt+1}/{max_attempts})",
                                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.7, (0, 0, 255), 2)
                        cv2.imshow("Marker Detection", prev)
                        if (cv2.waitKey(1) & 0xFF) == ord('q'):
                            cv2.destroyWindow("Marker Detection")
                            return None
                    continue

                rvec, tvec = self._solve_marker_pose(pts)
                if rvec is None:
                    continue
                samples_t.append(tvec)
                samples_r.append(rvec)
                last_corners, display = pts, img
                if len(samples_t) >= num_average:
                    break

            if len(samples_t) < max(4, num_average // 3):
                print("Marker detection failed (not enough valid frames).")
                try:
                    cv2.destroyWindow("Marker Detection")
                except cv2.error:
                    pass
                return None

            arr = np.array(samples_t)
            p_cam = np.median(arr, axis=0)
            spread_mm = np.std(arr, axis=0) * 1000.0
            rvec_med = np.median(np.array(samples_r), axis=0)
            R_cm = cv2.Rodrigues(rvec_med)[0]

            T_cam_marker = np.eye(4)
            T_cam_marker[:3, :3] = R_cm
            T_cam_marker[:3, 3] = p_cam

            info = {
                'p_cam': p_cam,
                'rvec': rvec_med,
                'normal': R_cm[:, 2],
                'T_cam_marker': T_cam_marker,
                'corners_px': last_corners,
                'centre_px': self.project(p_cam),
                'clean_image': display.copy(),
            }

            print("\n=== Marker Detection ===")
            print(f"Marker ID        : {target_id}")
            print(f"Frames averaged  : {len(samples_t)}")
            print(f"Centre in cam (m): X={p_cam[0]:+.4f} "
                  f"Y={p_cam[1]:+.4f} Z={p_cam[2]:+.4f}")
            print(f"Per-axis std (mm): {spread_mm[0]:.2f}, "
                  f"{spread_mm[1]:.2f}, {spread_mm[2]:.2f}")
            if np.max(spread_mm) > 2.0:
                print("  WARNING: 检测噪声 >2mm，检查光照/反光/对焦。")

            # Side-length self-check: the corner spacing implied by the solved pose should equal marker_size
            side = np.mean([np.linalg.norm(last_corners[i] -
                                           last_corners[(i + 1) % 4])
                            for i in range(4)])
            mm_per_px = self.marker_size * 1000.0 / max(side, 1e-6)
            print(f"Marker apparent side: {side:.1f} px "
                  f"({mm_per_px:.3f} mm/px at the marker plane)")

            if not require_confirmation:
                return info

            vis = display.copy()
            cv2.polylines(vis, [last_corners.astype(np.int32)], True,
                          (0, 255, 0), 2)
            c = info['centre_px'].astype(int)
            cv2.drawMarker(vis, tuple(c), (0, 0, 255),
                           cv2.MARKER_CROSS, 24, 2)
            cv2.drawFrameAxes(vis, self.camera_matrix, self.dist_coeffs,
                              rvec_med, p_cam, self.marker_size / 2.0)
            cv2.putText(vis, "a=accept  r=retake  q=cancel", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("Marker Detection", vis)

            decision = None
            while decision is None:
                k = cv2.waitKey(0) & 0xFF
                if k in (ord('a'), ord('r'), ord('q')):
                    decision = k
                else:
                    print("Press 'a', 'r' or 'q'.")
            cv2.destroyWindow("Marker Detection")

            if decision == ord('a'):
                return info
            if decision == ord('q'):
                return None
            print("Retaking...")

    # ==================================================================
    # Basic robot motion
    # ==================================================================
    def _get_pose(self):
        code, pose = self.arm.get_position()
        if code == 0:
            return np.asarray(pose, dtype=float)

        print("Failed to read xArm pose, code:", code)
        # A nonzero code doesn't necessarily mean an "error state" -- it can
        # also mean the status-reporting connection itself dropped (the
        # "[report-socket] socket read timeout" case is exactly this).
        # clean_error/set_state are useless on an already-disconnected
        # connection, so we need to reconnect first. This path only triggers
        # on an actual read failure; if the reconnect succeeds, this read's
        # pose is returned, otherwise None is returned as before -- this
        # does not change the existing failure-handling logic upstream.
        try:
            if not getattr(self.arm, 'connected', True):
                print("  [reconnect] 检测到连接已断开，尝试重连...")
                self.arm.connect()
                time.sleep(0.5)
            self.arm.clean_warn()
            self.arm.clean_error()
            self.arm.motion_enable(True)
            self.arm.set_mode(0)
            self.arm.set_state(0)
            time.sleep(0.3)
            code, pose = self.arm.get_position()
            if code == 0:
                print("  [reconnect] 重连/清状态后读位姿成功，继续。")
                return np.asarray(pose, dtype=float)
            print(f"  [reconnect] 重试后仍然失败，code={code}。")
        except Exception as exc:
            print(f"  [reconnect] 重连尝试本身出错：{exc}")
        return None

    def verify_touch_orientation(self, tolerance_deg=0.5, verbose=True):
        pose = self._get_pose()
        if pose is None:
            return False
        target = np.array([self.TOUCH_ROLL, self.TOUCH_PITCH, self.TOUCH_YAW])
        delta = (pose[3:6] - target + 180.0) % 360.0 - 180.0
        worst = float(np.max(np.abs(delta)))
        if verbose:
            print(f"Touch orientation deviation (deg): "
                  f"{np.round(delta, 4)}  max {worst:.4f}")
        if worst > tolerance_deg:
            print(f"  WARNING: 姿态偏差 {worst:.3f} deg > {tolerance_deg}, "
                  "法兰->凝胶偏置不再是常量，标定会被系统性拉偏。")
            return False
        return True

    def _clear_fault(self, context=""):
        """
        Call this after any move returns a nonzero error code: it pulls the
        xArm back from an error/stopped state into a normal controllable
        state, so that one hardware fault (e.g. torque protection) doesn't
        silently break every subsequent conditional move. This does not
        retry the current action -- it only guarantees the next call can
        issue commands normally.
        """
        if self.dry_run:
            return
        try:
            self.arm.clean_warn()
            self.arm.clean_error()
            self.arm.motion_enable(True)
            self.arm.set_mode(0)
            self.arm.set_state(0)
            time.sleep(0.3)
            if context:
                print(f"  [fault-clear] 已清除机械臂错误状态"
                      f"（触发处：{context}），可以继续下一步。")
        except Exception as exc:
            print(f"  [fault-clear] 清错误状态时又抛了异常：{exc}")

    def _move_to(self, xyz_mm, wait=True, speed=None):
        if self.dry_run:
            print(f"[DRY_RUN] move to {np.round(xyz_mm, 2)}")
            return 0
        return self.arm.set_position(
            x=float(xyz_mm[0]), y=float(xyz_mm[1]), z=float(xyz_mm[2]),
            roll=self.TOUCH_ROLL, pitch=self.TOUCH_PITCH, yaw=self.TOUCH_YAW,
            speed=speed or self.MOVE_SPEED, mvacc=500, wait=wait)

    def _move_relative(self, dxyz_mm, speed=None, wait=True):
        if self.dry_run:
            print(f"[DRY_RUN] move by {np.round(dxyz_mm, 2)}")
            return 0
        return self.arm.set_position(
            x=float(dxyz_mm[0]), y=float(dxyz_mm[1]), z=float(dxyz_mm[2]),
            relative=True, speed=speed or self.DESCEND_SPEED_SLOW,
            mvacc=200, wait=wait)

    def _return_to_pose(self, pose):
        if self.dry_run:
            print(f"[DRY_RUN] return to {np.round(pose[:3], 2)}")
            return 0
        ret = self.arm.set_position(
            x=float(pose[0]), y=float(pose[1]), z=float(pose[2]),
            roll=float(pose[3]), pitch=float(pose[4]), yaw=float(pose[5]),
            speed=self.MOVE_SPEED, mvacc=500, wait=True)
        if ret != 0:
            print("Warning: failed to return to pose:", ret)
        return ret

    def goto_xy_safely(self, target_xyz_mm):
        """
        Lift -> translate -> descend to the hover point.
        Never sweeps diagonally across the marker (a free-standing marker
        would shift position on the slightest touch).
        """
        cur = self._get_pose()
        if cur is None:
            return False
        travel_z = max(self.TRAVEL_Z_MM, cur[2],
                       target_xyz_mm[2] + self.APPROACH_HEIGHT_MM)
        hover = [target_xyz_mm[0], target_xyz_mm[1],
                 target_xyz_mm[2] + self.APPROACH_HEIGHT_MM]

        if self._move_to([cur[0], cur[1], travel_z]) != 0:
            return False
        if self._move_to([hover[0], hover[1], travel_z]) != 0:
            return False
        if self._move_to(hover) != 0:
            return False
        return True

    # ==================================================================
    # Descend to contact
    # ==================================================================
    def descend_to_contact(self, predicted_contact_z_mm=None):
        """
        Three-stage descent:
          1) Fast stage (3mm steps), down to predicted contact height + PRE_CONTACT_MARGIN;
          2) Re-capture the baseline, then a slow stage (0.3mm steps) to find first contact;
          3) Press stage (0.15mm steps) pressing to PRESS_TARGET_AREA_PCT,
             so the indentation amount is consistent every time and the recorded Z is repeatable.
        Returns (contact_pose(6,), start_pose(6,), info) or (None, start_pose, info)
        """
        info = {'area': 0.0, 'centroid': None, 'down_total': 0.0,
                'edge_contact': False}

        # Defensive fault clear: in case the previous call (or an earlier one)
        # left behind an uncleared fault state, clear it before reading the
        # pose, so the whole descent isn't planned from an untrustworthy
        # start_pose. Clearing faults is a harmless no-op on an arm that has
        # no fault, so it doesn't affect the normal flow.
        self._clear_fault()

        start_pose = self._get_pose()
        if start_pose is None:
            return None, None, info

        self.verify_touch_orientation()

        z_start = start_pose[2]
        had_prediction = predicted_contact_z_mm is not None
        if predicted_contact_z_mm is None:
            predicted_contact_z_mm = z_start - self.APPROACH_HEIGHT_MM

        # Before pressing down, sanity-check this number itself -- when
        # lightly touching the marker, Z should be near EXPECTED_CONTACT_Z_MM;
        # too large a deviation means this vision prediction itself is
        # untrustworthy, so it's rejected outright instead of only being
        # discovered after actually moving there. The rough estimate used in
        # manual jog mode (had_prediction=False) skips this check.
        if had_prediction:
            z_dev = abs(predicted_contact_z_mm - self.EXPECTED_CONTACT_Z_MM)
            if z_dev > self.CONTACT_Z_SANITY_TOL_MM:
                print(f"\nREJECTED: predicted contact Z="
                      f"{predicted_contact_z_mm:.2f}mm，偏离正常范围 "
                      f"{self.EXPECTED_CONTACT_Z_MM}"
                      f"±{self.CONTACT_Z_SANITY_TOL_MM}mm 达 {z_dev:.1f}mm，"
                      "这次视觉预测大概率不可信，拒绝下压。")
                return None, start_pose, info

        z_slow = predicted_contact_z_mm + self.PRE_CONTACT_MARGIN_MM
        z_floor = max(self.MIN_SAFE_Z_ABS_MM,
                      predicted_contact_z_mm - self.MAX_EXTRA_DOWN_MM)

        print(f"\nContact search: start Z={z_start:.2f}  "
              f"predicted contact Z={predicted_contact_z_mm:.2f}  "
              f"slow from Z={z_slow:.2f}  floor Z={z_floor:.2f}")

        # Just arrived at the hover position, mechanical vibration may not
        # have settled yet, so wait a bit before capturing the baseline;
        # if it fails the first time, wait longer and retry instead of
        # declaring failure right away -- the earlier case of "aborting
        # without pressing down at all" happened because this jumped to a
        # conclusion too quickly.
        time.sleep(0.3)
        if not self.contact.capture_baseline(status="baseline @ hover"):
            print("  悬停基线不可用，等待振动平息后重采一次...")
            time.sleep(0.6)
            if not self.contact.capture_baseline(
                    status="baseline @ hover (retry)"):
                print("  重采仍然不可用，放弃这次接触尝试。")
                return None, start_pose, info

        # ---------- 1) Fast stage ----------
        while True:
            cur = self._get_pose()
            if cur is None:
                return None, start_pose, info
            if cur[2] <= z_slow + 1e-3 or cur[2] <= z_floor + 1e-3:
                break
            step = min(self.DOWN_STEP_FAST_MM, cur[2] - z_slow)
            if self._move_relative([0, 0, -step],
                                   speed=self.DESCEND_SPEED_FAST) != 0:
                print("Robot error during fast descent.")
                self._clear_fault("fast descent")
                return None, start_pose, info
            info['down_total'] += step
            area, _, _ = self.contact.measure(
                status=f"fast descent  Z={cur[2]-step:.1f}")
            if area > self.contact.thr_on:
                print(f"  Early contact during fast phase "
                      f"(area {area:.2f}%). 预测高度可能偏低。")
                break

        # ---------- 2) Slow stage ----------
        time.sleep(0.3)
        print("  Re-capturing DIGIT baseline just above the surface...")
        if not self.contact.capture_baseline(num_frames=25,
                                             status="baseline @ pre-contact"):
            print("  预接触基线不可用，等待振动平息后重采一次...")
            time.sleep(0.6)
            if not self.contact.capture_baseline(
                    num_frames=25,
                    status="baseline @ pre-contact (retry)"):
                print("  重采仍然不可用，放弃这次接触尝试。")
                return None, start_pose, info

        contact_pose = None
        while True:
            cur = self._get_pose()
            if cur is None:
                break
            if cur[2] - self.DOWN_STEP_SLOW_MM < z_floor:
                print(f"  Safety stop: reached floor Z={z_floor:.2f} mm "
                      "without contact.")
                break

            if self._move_relative([0, 0, -self.DOWN_STEP_SLOW_MM],
                                   speed=self.DESCEND_SPEED_SLOW) != 0:
                print("Robot error during slow descent.")
                self._clear_fault("slow descent")
                break
            info['down_total'] += self.DOWN_STEP_SLOW_MM
            time.sleep(0.05)

            area, cen, _ = self.contact.measure(
                status=f"searching  Z={cur[2]-self.DOWN_STEP_SLOW_MM:.2f}")
            if area <= self.contact.thr_on:
                continue

            confirms = [area]
            for _ in range(self.CONTACT_CONFIRM_FRAMES - 1):
                time.sleep(0.05)
                confirms.append(self.contact.measure(
                    status="confirming contact")[0])
            print(f"  contact candidate: {[round(v, 2) for v in confirms]} "
                  f"(thr {self.contact.thr_on:.2f})")
            if all(v > self.contact.thr_on for v in confirms):
                contact_pose = self._get_pose()
                print("  >>> First contact detected.")
                break
            print("  transient ignored.")

        if contact_pose is None:
            print("Contact search finished WITHOUT contact.")
            return None, start_pose, info

        # ---------- 3) Press stage (keeps the indentation amount consistent) ----------
        indent = 0.0
        area, cen, _ = self.contact.measure(status="pressing")
        while (area < self.PRESS_TARGET_AREA_PCT
               and indent < self.MAX_INDENT_MM):
            cur = self._get_pose()
            if cur is None or cur[2] - self.DOWN_STEP_PRESS_MM < z_floor:
                break
            if self._move_relative([0, 0, -self.DOWN_STEP_PRESS_MM],
                                   speed=self.DESCEND_SPEED_SLOW) != 0:
                print("Robot error during press phase.")
                self._clear_fault("press phase")
                break
            indent += self.DOWN_STEP_PRESS_MM
            info['down_total'] += self.DOWN_STEP_PRESS_MM
            time.sleep(0.05)
            area, cen, _ = self.contact.measure(
                status=f"pressing {indent:.2f}mm")

        contact_pose = self._get_pose()
        info['area'] = area
        info['centroid'] = cen
        info['indent'] = indent

        # Apply the systematic correction "DIGIT's real center relative to
        # TCP" obtained from the corner-press calibration -- after this
        # step, contact_pose records the point that the DIGIT's actual
        # physical center touched, not the raw, uncorrected TCP position
        # reported by the arm itself. When digit_center_offset_mm=0 (the
        # default), this line is a no-op, matching the previous behavior
        # exactly.
        if contact_pose is not None and np.any(self.digit_center_offset_mm):
            contact_pose = contact_pose.copy()
            contact_pose[0] += self.digit_center_offset_mm[0]
            contact_pose[1] += self.digit_center_offset_mm[1]
            info['digit_center_corrected'] = True

        off = self.contact.contact_offset_mm(cen)
        if off is not None:
            print(f"  contact blob offset on gel: "
                  f"{off[0]:+.1f}, {off[1]:+.1f} mm")
            if np.linalg.norm(off) > 4.0 and area < 1.5 * \
                    self.PRESS_TARGET_AREA_PCT:
                info['edge_contact'] = True
                print("  WARNING: 接触斑块明显偏心且面积不足 —— "
                      "很可能压在 marker 边缘或斜面上，建议纠偏后重压。")

        print(f"  Final: area={area:.2f}%  indent={indent:.2f}mm  "
              f"Z={contact_pose[2]:.3f}mm  total down={info['down_total']:.1f}mm")
        if np.any(self.digit_center_offset_mm):
            print(f"  (已应用 DIGIT 中心修正: "
                  f"{self.digit_center_offset_mm[0]:+.3f}, "
                  f"{self.digit_center_offset_mm[1]:+.3f} mm)")
        return contact_pose, start_pose, info

    # ==================================================================
    # Visual correction (the core new feature)
    # ==================================================================
    def _on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self._click = (x, y)

    def _auto_tip_hint(self, image_now, marker_info):
        """
        Compute a diff between the "clean image" and the "image at contact"
        near the marker, to give an automatic hint of the tool landing point
        (only a hint -- the operator's click is what's actually used).
        """
        try:
            g0 = cv2.cvtColor(marker_info['clean_image'], cv2.COLOR_BGR2GRAY)
            g1 = cv2.cvtColor(image_now, cv2.COLOR_BGR2GRAY)
            corners = marker_info['corners_px']
            x0, y0 = corners.min(axis=0)
            x1, y1 = corners.max(axis=0)
            pad = 0.35 * max(x1 - x0, y1 - y0)
            x0, y0 = int(max(0, x0 - pad)), int(max(0, y0 - pad))
            x1 = int(min(g0.shape[1] - 1, x1 + pad))
            y1 = int(min(g0.shape[0] - 1, y1 + pad))

            d = cv2.absdiff(g0[y0:y1, x0:x1], g1[y0:y1, x0:x1])
            d = cv2.GaussianBlur(d, (7, 7), 0)
            _, m = cv2.threshold(d, 35, 255, cv2.THRESH_BINARY)
            m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
            cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
            if not cnts:
                return None
            c = max(cnts, key=cv2.contourArea)
            if cv2.contourArea(c) < 200:
                return None
            M = cv2.moments(c)
            return (x0 + M['m10'] / M['m00'], y0 + M['m01'] / M['m00'])
        except Exception:
            return None

    def visual_xy_correction(self, image_at_contact, marker_info, T_use):
        """
        At the moment of contact, the tool landing point lies exactly on the
        marker plane (zero parallax). The operator clicks the contact
        position between the tool and the marker surface in a zoomed
        window; the program back-projects it onto the marker plane and
        computes the base-frame XY correction.

        Returns (dxy_mm(2,), action), where action is one of
        {'apply','accept','abort'}
        """
        corners = marker_info['corners_px']
        centre_px = marker_info['centre_px']
        p_cam = marker_info['p_cam']
        normal = marker_info['normal']

        x0, y0 = corners.min(axis=0)
        x1, y1 = corners.max(axis=0)
        pad = 0.55 * max(x1 - x0, y1 - y0)
        X0 = int(max(0, x0 - pad)); Y0 = int(max(0, y0 - pad))
        X1 = int(min(image_at_contact.shape[1], x1 + pad))
        Y1 = int(min(image_at_contact.shape[0], y1 + pad))
        roi = image_at_contact[Y0:Y1, X0:X1].copy()
        if roi.size == 0:
            return np.zeros(2), 'accept'

        scale = max(1.0, min(3.0, 900.0 / max(roi.shape[1], 1)))
        big = cv2.resize(roi, None, fx=scale, fy=scale,
                         interpolation=cv2.INTER_CUBIC)

        def to_big(px):
            return (int((px[0] - X0) * scale), int((px[1] - Y0) * scale))

        def to_full(px):
            return (px[0] / scale + X0, px[1] / scale + Y0)

        hint = self._auto_tip_hint(image_at_contact, marker_info)

        win = "XY alignment  (click the tool footprint)"
        cv2.namedWindow(win)
        self._click = None
        cv2.setMouseCallback(win, self._on_mouse)

        chosen = hint
        while True:
            vis = big.copy()
            cv2.drawMarker(vis, to_big(centre_px), (0, 255, 0),
                           cv2.MARKER_CROSS, 40, 2)
            cv2.circle(vis, to_big(centre_px), 4, (0, 255, 0), -1)
            if hint is not None:
                cv2.circle(vis, to_big(hint), 9, (255, 0, 255), 2)
            if chosen is not None:
                cv2.drawMarker(vis, to_big(chosen), (0, 0, 255),
                               cv2.MARKER_TILTED_CROSS, 30, 2)
                cv2.arrowedLine(vis, to_big(chosen), to_big(centre_px),
                                (0, 200, 255), 2, tipLength=0.15)

            txt = ["GREEN = marker centre   RED = where the tool landed",
                   "click the tool/gel footprint on the marker surface",
                   "y = apply correction   a = accept as is   "
                   "r = retouch   q = abort"]
            if chosen is not None:
                Xc = self.backproject_to_plane(chosen, p_cam, normal)
                if Xc is not None:
                    e_cam = p_cam - Xc
                    d = 1000.0 * (T_use[:3, :3] @ e_cam)
                    txt.append(f"correction: dX={d[0]:+.1f} dY={d[1]:+.1f} mm"
                               f"   (|dXY|={np.linalg.norm(d[:2]):.1f} mm)")
            for i, line in enumerate(txt):
                cv2.putText(vis, line, (8, 22 + 22 * i),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)

            cv2.imshow(win, vis)
            k = cv2.waitKey(30) & 0xFF

            if self._click is not None:
                chosen = to_full(self._click)
                self._click = None
                continue

            if k == ord('y') and chosen is not None:
                Xc = self.backproject_to_plane(chosen, p_cam, normal)
                cv2.destroyWindow(win)
                if Xc is None:
                    return np.zeros(2), 'accept'
                e_cam = p_cam - Xc
                d = 1000.0 * (T_use[:3, :3] @ e_cam)
                return d[:2], 'apply'
            if k == ord('a'):
                cv2.destroyWindow(win)
                return np.zeros(2), 'accept'
            if k == ord('r'):
                cv2.destroyWindow(win)
                return np.zeros(2), 'retouch'
            if k == ord('q'):
                cv2.destroyWindow(win)
                return np.zeros(2), 'abort'

    # ==================================================================
    # Manual jog (with live camera crosshair)
    # ==================================================================
    def jog_to_marker(self, marker_info=None):
        """
        w/s = +X/-X   a/d = +Y/-Y   r/f = +Z/-Z
        1/2/3 = step size 10 / 1 / 0.2 mm
        SPACE = auto-descend to find contact     x/ESC = abort
        The green crosshair in the "JOG - camera view" window is the marker
        center -- move the tool's landing point onto the crosshair, then
        press space.
        """
        step_mm = 5.0
        print("\n--- MANUAL JOG ---")
        print("  w/s=+X/-X  a/d=+Y/-Y  r/f=+Z/-Z  1/2/3=step 10/1/0.2mm")
        print("  SPACE=auto descend   x=abort")
        print("  看 'JOG - camera view' 窗口：把工具落点对准绿色十字。")

        cur = self._get_pose()
        if cur is None:
            return None, None, {}
        self._move_to(cur[:3])

        last_img = None
        cam_win = "JOG - camera view"
        while True:
            last_img = self._color_image_nonblocking(last_img)
            if last_img is not None and marker_info is not None:
                vis = last_img.copy()
                c = marker_info['centre_px'].astype(int)
                cv2.drawMarker(vis, tuple(c), (0, 255, 0),
                               cv2.MARKER_CROSS, 40, 2)
                cv2.polylines(vis,
                              [marker_info['corners_px'].astype(np.int32)],
                              True, (0, 255, 0), 1)
                pose = self._get_pose()
                if pose is not None:
                    cv2.putText(vis, f"X{pose[0]:7.1f} Y{pose[1]:7.1f} "
                                     f"Z{pose[2]:7.1f}  step={step_mm}mm",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, (0, 255, 255), 2)
                cv2.imshow(cam_win, cv2.resize(vis, (960, 540)))

            pose = self._get_pose()
            pos_txt = ("pose n/a" if pose is None else
                       f"X{pose[0]:6.1f} Y{pose[1]:6.1f} Z{pose[2]:6.1f}")
            self.contact.show(status_text=f"JOG step={step_mm}mm\n{pos_txt}\n"
                                          "SPACE=touch  x=abort")

            key = cv2.waitKey(30) & 0xFF
            if key == 255:
                continue

            delta = np.zeros(3)
            if key == ord('w'):
                delta[0] = step_mm
            elif key == ord('s'):
                delta[0] = -step_mm
            elif key == ord('a'):
                delta[1] = step_mm
            elif key == ord('d'):
                delta[1] = -step_mm
            elif key == ord('r'):
                delta[2] = step_mm
            elif key == ord('f'):
                delta[2] = -step_mm
            elif key == ord('1'):
                step_mm = 10.0
                continue
            elif key == ord('2'):
                step_mm = 1.0
                continue
            elif key == ord('3'):
                step_mm = 0.2
                continue
            elif key == ord(' '):
                try:
                    cv2.destroyWindow(cam_win)
                except cv2.error:
                    pass
                print("Starting automatic descent...")
                return self.descend_to_contact()
            elif key in (ord('x'), 27):
                try:
                    cv2.destroyWindow(cam_win)
                except cv2.error:
                    pass
                print("Aborted by user.")
                return None, None, {}
            else:
                continue

            if delta[2] < 0 and pose is not None and \
                    pose[2] + delta[2] < self.MIN_SAFE_Z_ABS_MM:
                print("Jog blocked: below minimum safe Z.")
                continue
            self._move_relative(delta, speed=30)

    # ==================================================================
    # DIGIT real-center calibration: press on a marker corner, first
    # converge automatically, then fine-tune manually
    # ==================================================================
    def calibrate_digit_center_at_corner(self, corner_p_cam,
                                         max_auto_iters=3,
                                         pos_tol_mm=0.3,
                                         gel_mm=(18.6, 14.3)):
        """
        Use a corner of the marker to calibrate the offset between the
        DIGIT gel's true physical center and its current registered TCP
        value. Principle: when lightly pressing on a corner, half the gel
        touches the marker (deforms) and half doesn't (no deformation or
        very shallow), so the deformation contour is an asymmetric
        right-angle shape. The position of this "corner" in the image,
        together with the direction of its two edges, gives both the
        positional deviation and the orientation deviation at once.

        Stage 1 (fully automatic, up to max_auto_iters times):
          light press -> detect the corner vertex from the deformation mask
          -> compute the vertex's pixel offset relative to the crosshair
          center (converted to mm) -> move to correct -> repeat until
          |offset| converges to within pos_tol_mm, or iterations run out.

        Stage 2 (manual fine-tuning):
          After the automatic stage, the arm stays at its current lightly-
          pressed position, and the DIGIT window keeps refreshing (contour +
          crosshair overlaid live). You fine-tune xy yourself using the xArm
          Studio manual mode panel (you can also fine-tune the angle, if you
          want the marker's two edges to line up more closely with the
          crosshair's horizontal/vertical lines); once you're satisfied with
          the alignment, come back to this terminal and press Enter to confirm.

        Note: this step **only auto-corrects position (XY), never
        orientation automatically** -- because automatically changing
        TOUCH_YAW would affect the pose assumption for every subsequent
        contact, which is too risky. So the angular deviation of the two
        edges relative to the crosshair is only printed as a number for your
        reference; it's up to you to decide whether to fix it via manual
        rotation in stage 2, or by physically re-seating the marker.

        Returns a dict:
          digit_center_pose      the arm's actual pose at the moment of manual confirmation
          auto_converged         whether stage 1 converged within max_auto_iters
          final_edge_deviation_deg  the edge angle deviation measured on the
                                     last stage-1 iteration (None if no corner
                                     could be identified)
          digit_center_offset_mm  the total offset between the final
                                   confirmed position and the initial
                                   predicted position (base-frame XY, mm) --
                                   this is the DIGIT real-center correction
                                   produced by this calibration.

        This offset is immediately written to self.digit_center_offset_mm
        and saved to disk (filename digit_center_offset_<timestamp>.npz).
        **From this moment on, any contact_pose returned by
        descend_to_contact during this session automatically has this
        correction added** -- whether it's the main calibration sampling,
        the ground-truth reference for degradation experiments, or a
        recovery touch, all go through this single correction uniformly, so
        it doesn't need to be handled separately in each place. It's also
        automatically loaded from disk the next time the program starts.

        Note: this offset is measured using the T_base_cam (or guess) in
        effect at the time of this call. If the hand-eye calibration is
        redone afterward (T_base_cam changes), it becomes unclear how much
        of the offset was "the DIGIT's real mechanical offset" versus "the
        residual error of that particular calibration" -- it's recommended
        to rerun this function every time the hand-eye calibration is redone.
        """
        # When running this function standalone (e.g. calibration hasn't
        # run yet and only a previously saved result exists),
        # self.T_base_cam will be None, in which case we fall back to the
        # loaded guess, consistent with how touch_marker_at handles it --
        # otherwise this would just crash.
        T_use = (self.T_base_cam if self.T_base_cam is not None
                 else self.T_base_cam_guess)
        if T_use is None:
            print("没有可用的标定（既没有本次求解结果，也没有加载到"
                  "上次的存档），无法把角点坐标换算到 base 系，中止。")
            return {'digit_center_pose': None, 'auto_converged': False,
                    'final_edge_deviation_deg': None}

        pred_mm = (T_use[:3, :3] @ corner_p_cam + T_use[:3, 3]) * 1000.0
        pred_xy_initial = pred_mm[:2].copy()  # used to compute the final total correction

        print("\n=== 阶段一：系统自动逼近角点 ===")
        auto_converged = False
        last_edge_dev = None

        for i in range(max_auto_iters):
            # pred_mm comes from T_use; if T_use itself was calibrated from
            # corrected samples, then pred_mm predicts "where the DIGIT
            # center should be", so the current known correction must be
            # subtracted when moving the TCP, to avoid double-counting with
            # the +d applied inside descend_to_contact (this is exactly the
            # correction we're re-calibrating and possibly tightening here,
            # so the "current" value is used, not the new value this run
            # will ultimately produce).
            move_target = pred_mm.copy()
            move_target[:2] -= self.digit_center_offset_mm
            if not self.goto_xy_safely(move_target):
                print(f"  第{i+1}次：无法到达悬停点，中止自动阶段。")
                break

            self._clear_fault()
            start = self._get_pose()
            if start is None:
                print("  无法读取位姿，中止自动阶段。")
                break

            # This step was missing before: without a baseline, measure()
            # just returns empty and the DIGIT window won't even pop up, so
            # the analysis below would never get anything. Capture one while
            # still hovering and not touching anything; re-capture if it's
            # too noisy (same hover-baseline logic as in descend_to_contact).
            time.sleep(0.3)
            if not self.contact.capture_baseline(
                    status=f"corner-calib baseline 第{i+1}次"):
                print(f"  第{i+1}次：悬停基线不可用，等待振动平息后"
                      "重采一次...")
                time.sleep(0.6)
                if not self.contact.capture_baseline(
                        status=f"corner-calib baseline 第{i+1}次(retry)"):
                    print(f"  第{i+1}次：基线仍不可用，跳过这次迭代。")
                    continue

            # Light press: no longer guessing a fixed height and diving
            # straight to it (a miscalculation there would raise the target
            # height by one contact distance and never actually press down)
            # -- instead, descend in small steps, checking for contact at
            # each step against the just-captured baseline threshold, and
            # stop as soon as "contact detected". This is the truly reliable
            # way to lightly touch a corner.
            z_floor = pred_mm[2] - self.MAX_EXTRA_DOWN_MM
            touched = False
            frame = None
            for _ in range(200):
                cur = self._get_pose()
                if cur is None or \
                        cur[2] - self.DOWN_STEP_SLOW_MM < z_floor:
                    break
                if self._move_relative(
                        [0, 0, -self.DOWN_STEP_SLOW_MM],
                        speed=self.DESCEND_SPEED_SLOW) != 0:
                    self._clear_fault("corner-calib 下降")
                    break
                time.sleep(0.03)
                area, _, frame = self.contact.measure(
                    status=f"corner-calib 逼近 第{i+1}次")
                if area > self.contact.thr_on:
                    touched = True
                    break

            # Right at the moment the threshold is crossed, the deformation
            # is often just a small blurry smudge, not yet enough to show a
            # clear right-angle shape -- this was the main reason for the
            # recurring "contour doesn't look like a clean right angle"
            # issue. Press a few more steps here, to bring the area up to
            # an intermediate level where the shape is clearly visible but
            # not yet fully pressed flat (CORNER_READ_AREA_PCT), and run
            # the corner analysis on that frame instead.
            if touched:
                for _ in range(30):
                    if area >= self.CORNER_READ_AREA_PCT:
                        break
                    cur = self._get_pose()
                    if cur is None or \
                            cur[2] - self.DOWN_STEP_PRESS_MM < z_floor:
                        break
                    if self._move_relative(
                            [0, 0, -self.DOWN_STEP_PRESS_MM],
                            speed=self.DESCEND_SPEED_SLOW) != 0:
                        self._clear_fault("corner-calib 补压")
                        break
                    time.sleep(0.03)
                    area, _, frame = self.contact.measure(
                        status=f"corner-calib 补压 第{i+1}次")

            self._retract(self.RETRACT_MM)

            if not touched:
                print(f"  第{i+1}次：下降到安全下限仍未检测到接触，"
                      "跳过这次迭代（可能没对准marker角点，或者 "
                      "MAX_EXTRA_DOWN_MM 需要调大）。")
                continue

            if frame is None or self.contact.baseline is None:
                print(f"  第{i+1}次：没拿到图像或基线未就绪，跳过。")
                continue

            g, _ = self.contact._prep(frame)
            _, mask, _ = self.contact._analyse(g)
            corner_px, edge_dev, conf = self.contact.detect_corner(mask)
            last_edge_dev = edge_dev

            if corner_px is None:
                print(f"  第{i+1}次：轮廓不像清晰的直角（置信度太低），"
                      "可能没压在角点附近，跳过这次迭代。")
                continue

            offset_mm = self.contact.contact_offset_mm(corner_px,
                                                        gel_mm=gel_mm)
            err = float(np.linalg.norm(offset_mm))
            print(f"  第{i+1}次：顶点偏移 = ({offset_mm[0]:+.2f}, "
                  f"{offset_mm[1]:+.2f}) mm  |err|={err:.2f}mm  "
                  f"边缘角度偏差={edge_dev:.1f}°  置信度={conf:.2f}")

            if err < pos_tol_mm:
                print(f"  收敛（< {pos_tol_mm}mm）。")
                auto_converged = True
                break

            self._move_relative([offset_mm[0], offset_mm[1], 0],
                                speed=self.MOVE_SPEED)
            pred_mm = np.array([pred_mm[0] + offset_mm[0],
                                pred_mm[1] + offset_mm[1],
                                pred_mm[2]])
        else:
            print(f"  自动阶段跑满 {max_auto_iters} 次仍未收敛，"
                  "剩余偏差交给人工阶段处理。")

        print("\n=== 阶段二：人工微调 ===")
        print("  机械臂已停在当前轻压位置，DIGIT窗口会持续刷新。用 "
              "xArm Studio 的手动模式面板自己微调，对齐满意后回到"
              "这个终端窗口、按回车确认。")

        # Also need to capture a baseline while hovering, not touching
        # anything, first — otherwise the upcoming measure() call will
        # just return empty and the DIGIT window won't pop up at all.
        time.sleep(0.3)
        if not self.contact.capture_baseline(status="人工阶段基线"):
            print("  悬停基线不可用，等待振动平息后重采一次...")
            time.sleep(0.6)
            if not self.contact.capture_baseline(
                    status="人工阶段基线(retry)"):
                print("  基线仍然不可用——DIGIT窗口可能显示不出来，"
                      "建议检查凝胶状态/重启后再试，或者直接用 "
                      "xArm Studio 手动操作、不依赖这个窗口做判断。")

        # Likewise, do not add EXPECTED_CONTACT_Z_MM again here — pred_mm[2]
        # already approximates the light-touch height, so use it directly
        # as the starting point for manual fine-tuning; if it feels like
        # it hasn't touched down, jog further down in Z with xArm Studio.
        # Also subtract the current offset correction, for the same
        # reason as in phase one.
        move_target2 = pred_mm.copy()
        move_target2[:2] -= self.digit_center_offset_mm
        self._move_to(move_target2, speed=self.DESCEND_SPEED_SLOW)
        while True:
            self.contact.measure(status="人工微调中，对齐后按回车确认")
            if cv2.waitKey(200) & 0xFF in (13, 10):  # Enter
                break

        final_pose = self._get_pose()
        self._retract(self.RETRACT_MM)
        self.contact.close_window()

        print(f"\n人工确认完成，最终位姿：{np.round(final_pose, 3)}")

        # Total correction = final confirmed position - initial (pure
        # vision + calibration) predicted position; this captures both
        # stage 1's automatic correction and stage 2's manual fine-tuning
        # in one shot, no separate accumulation needed. This is the
        # systematic offset of "DIGIT's real center relative to its
        # registered TCP value" -- it takes effect immediately on this
        # calibrator instance (every descend_to_contact for the rest of
        # this session will automatically carry it) and is also saved to
        # disk, to be loaded automatically on the next startup.
        digit_center_offset_mm = final_pose[:2] - pred_xy_initial
        self.digit_center_offset_mm = digit_center_offset_mm
        self._save_digit_center_offset(digit_center_offset_mm)
        print(f"DIGIT 中心偏移 = ({digit_center_offset_mm[0]:+.3f}, "
              f"{digit_center_offset_mm[1]:+.3f}) mm —— 已立即生效，"
              "本次会话之后所有触碰都会自动修正。")

        return {
            'digit_center_pose': final_pose,
            'auto_converged': auto_converged,
            'final_edge_deviation_deg': last_edge_dev,
            'digit_center_offset_mm': digit_center_offset_mm,
        }

    # ==================================================================
    # Closed-loop converged touch: repeat "touch -> visual correction"
    # until convergence, to establish an independent, trustworthy ground-
    # truth point -- used for the degradation experiment's reference
    # ground truth and every unified_evaluation test point, instead of a
    # single touch (a single touch has no XY search, so its recorded
    # coordinate is always exactly the commanded predicted coordinate and
    # cannot measure real accuracy).
    # ==================================================================
    def touch_marker_converged(self, info, tol_mm=0.2, max_iters=5):
        """
        Returns (contact_pose, dxy_history)
          contact_pose : the end-effector pose of the last touch after convergence
          dxy_history  : |dXY| (mm) for each iteration -- the convergence curve, for reports
        """
        p_cam = info['p_cam']
        pred_mm = (self.T_base_cam[:3, :3] @ p_cam +
                   self.T_base_cam[:3, 3]) * 1000.0
        z_hint = pred_mm[2]

        dxy_history = []
        contact_pose = None

        for i in range(max_iters):
            move_target = pred_mm.copy()
            move_target[:2] -= self.digit_center_offset_mm
            if not self.goto_xy_safely(move_target):
                print(f"  [ref-touch] Attempt {i+1}: could not reach the "
                      f"hover point, aborting. / "
                      f"第{i+1}次：无法到达悬停点，中止。")
                break

            contact_pose, _, _ = self.descend_to_contact(z_hint)
            if contact_pose is None:
                print(f"  [ref-touch] Attempt {i+1}: no contact, empty "
                      f"press, aborting. / "
                      f"第{i+1}次：未接触，压空，中止。")
                break

            img = self._color_image()
            self._retract(self.RETRACT_MM)
            if img is None:
                print(f"  [ref-touch] Attempt {i+1}: no camera frame, "
                      f"cannot correct, aborting. / "
                      f"第{i+1}次：无相机帧，无法纠偏，中止。")
                break

            dxy, action = self.visual_xy_correction(
                img, info, self.T_base_cam)
            err = float(np.linalg.norm(dxy))
            dxy_history.append(err)
            print(f"  [ref-touch] Attempt {i+1} / 第{i+1}次: "
                  f"|dXY| = {err:.3f} mm")

            if err < tol_mm:
                print(f"  [ref-touch] Converged (< {tol_mm}mm) — using "
                      f"attempt {i+1} as ground truth. / "
                      f"收敛（<{tol_mm}mm），用第{i+1}次接触作为"
                      f"ground truth。")
                break

            if action == 'abort':
                print(f"  [ref-touch] Attempt {i+1}: the suggested "
                      f"correction was too large and was rejected, "
                      f"aborting. / "
                      f"第{i+1}次：纠偏建议的修正量过大被拒绝，中止。")
                break

            self._move_relative([dxy[0], dxy[1], 0.0],
                                speed=self.MOVE_SPEED)
            pred_mm = np.array([pred_mm[0] + dxy[0],
                                pred_mm[1] + dxy[1],
                                pred_mm[2]])
            z_hint = contact_pose[2]
        else:
            print(f"  [ref-touch] WARNING: did not converge to within "
                  f"{tol_mm}mm after {max_iters} iterations (last |dXY|="
                  f"{dxy_history[-1]:.3f}mm). Using the last attempt "
                  f"anyway — check the calibration or hardware. / "
                  f"警告：{max_iters}次迭代后仍未收敛到{tol_mm}mm以内"
                  f"（最后一次|dXY|={dxy_history[-1]:.3f}mm）。仍使用"
                  f"最后一次接触，建议检查标定或硬件。")

        return contact_pose, dxy_history

    # ==================================================================
    # One complete "look - align - press" cycle
    # ==================================================================
    def touch_marker_at(self, marker_info, phase="sample"):
        """
        1) Coarsely position above the marker using the existing calibration (or guess);
        2) Descend to contact;
        3) At the moment of contact, grab a camera frame -> retract -> visual correction -> press down again;
        4) Repeat until the correction < ALIGN_TOL_MM.
        Returns (contact_pose(6,), predicted_xyz_mm(3,)) or (None, pred)
        """
        p_cam = marker_info['p_cam']
        T_use = (self.T_base_cam if self.T_base_cam is not None
                 else self.T_base_cam_guess)

        # ---------- Coarse positioning ----------
        predicted = None
        if T_use is not None:
            p_base = T_use[:3, :3] @ p_cam + T_use[:3, 3]
            predicted = p_base * 1000.0
            start = self._get_pose()
            if start is None:
                return None, predicted
            dist = float(np.linalg.norm(predicted - start[:3]))

            print(f"\n=== Touch marker: {phase} ===")
            print("Current  (mm):", np.round(start[:3], 2))
            print("Predicted(mm):", np.round(predicted, 2))
            print(f"Travel distance: {dist:.1f} mm")

            if predicted[2] < self.MIN_SAFE_Z_ABS_MM:
                print("REJECTED: 预测接触 Z 低于安全下限，标定很可能是错的。")
                return None, predicted
            if dist > self.MAX_TARGET_DISTANCE_MM:
                print("REJECTED: 目标太远，标定符号可能反了。")
                return None, predicted
            if input("Proceed? [y/N]: ").strip().lower() != 'y':
                return None, predicted
            # predicted is "where the DIGIT's real center should be" (solved
            # from correction-adjusted samples during calibration), but
            # goto_xy_safely moves the TCP, and the TCP's target is
            # predicted minus this correction -- without subtracting it,
            # descend_to_contact would add the correction again when
            # recording, i.e. it would get applied twice. predicted itself
            # is left unchanged and is still the "target" used for
            # comparison/printing.
            move_target = predicted.copy()
            move_target[:2] -= self.digit_center_offset_mm
            if not self.goto_xy_safely(move_target):
                print("Failed to reach hover pose.")
                return None, predicted
        else:
            print("\n没有可用的标定，进入手动 jog 模式。")
            cp, sp, info = self.jog_to_marker(marker_info)
            if cp is None:
                return None, None
            # Also run a visual correction pass in manual mode
            img = self._color_image()
            self._retract(self.RETRACT_MM)
            if img is not None and self.T_base_cam_guess is not None:
                pass  # no T available to convert direction, skip correction
            return cp, None

        # ---------- Contact + correction iterations ----------
        predicted_z = predicted[2]
        total_corr = np.zeros(2)
        contact_pose = None

        for it in range(self.ALIGN_MAX_ITERS):
            print(f"\n--- touch attempt {it+1}/{self.ALIGN_MAX_ITERS} ---")
            cp, sp, info = self.descend_to_contact(predicted_z)
            if cp is None:
                # No contact: most likely the Z prediction was too high, or it simply missed the marker
                if sp is not None:
                    self._return_to_pose(sp)
                print("No contact this attempt.")
                if it == self.ALIGN_MAX_ITERS - 1:
                    break
                predicted_z -= 5.0
                continue

            contact_pose = cp
            predicted_z = cp[2]        # use the measured surface height next time -- fast and safe

            img_contact = self._color_image()
            self._retract(self.RETRACT_MM)

            # When ALIGN_MAX_ITERS<=1 (this is what test_calibration/
            # test_calibration_two_markers use), the caller's intent is
            # already "measure raw accuracy once, no iterative correction",
            # and they separately pop up a dedicated correction window later
            # just to read the number. If this code still went through the
            # usual "popup -> apply correction -> move the arm" here, the
            # arm would move without a second touch to verify it, and the
            # recorded coordinate wouldn't use the result of that move
            # anyway -- it would be a pointless move plus an equally
            # pointless popup. So this is skipped here, and this touch is
            # returned as-is.
            if self.ALIGN_MAX_ITERS <= 1:
                print("ALIGN_MAX_ITERS<=1: skipping in-loop correction "
                      "(caller will measure separately). / "
                      "ALIGN_MAX_ITERS<=1：跳过本轮纠偏弹窗"
                      "（调用方会单独测量）。")
                break

            if img_contact is None:
                print("No camera frame for correction; accepting this touch.")
                break

            dxy, action = self.visual_xy_correction(
                img_contact, marker_info, T_use)

            if action == 'abort':
                return None, predicted
            if action == 'accept':
                print("Operator accepted the current landing point.")
                break
            if action == 'retouch':
                continue

            mag = float(np.linalg.norm(dxy))
            print(f"Applying XY correction: dX={dxy[0]:+.2f} "
                  f"dY={dxy[1]:+.2f} mm (|d|={mag:.2f})")
            total_corr += dxy
            if mag > 60.0:
                print("修正量过大 (>60mm)，拒绝执行。请检查点击位置/标定。")
                break
            self._move_relative([dxy[0], dxy[1], 0.0],
                                speed=self.MOVE_SPEED)
            if mag < self.ALIGN_TOL_MM:
                print("Within tolerance; doing one final touch.")
                cp, sp, info = self.descend_to_contact(predicted_z)
                if cp is not None:
                    contact_pose = cp
                self._retract(self.RETRACT_MM)
                break
        else:
            print("Reached the maximum number of alignment iterations.")

        self.applied_corrections_mm.append(float(np.linalg.norm(total_corr)))
        self.contact.close_window()

        if contact_pose is None:
            return None, predicted
        print("Contact pose (mm/deg):", np.round(contact_pose, 3))
        return contact_pose, predicted

    def _retract(self, mm):
        cur = self._get_pose()
        if cur is None:
            return
        self._move_to([cur[0], cur[1], cur[2] + mm],
                      speed=self.MOVE_SPEED)

    # ==================================================================
    # Sampling
    # ==================================================================
    def capture_sample(self, index, total):
        self.arm.clean_warn()
        self.arm.clean_error()
        self.arm.motion_enable(True)
        self.arm.set_mode(0)
        self.arm.set_state(0)
        time.sleep(0.3)

        print("\n" + "=" * 62)
        print(f"Sample {index+1}/{total}  -  Step 1: detect the marker")
        print("=" * 62)
        input("确认 marker 可见、机械臂不遮挡相机，然后回车...")

        marker_info = self.get_marker_pose()
        if marker_info is None:
            print("Marker detection failed.")
            return None

        print("\n" + "=" * 62)
        print(f"Sample {index+1}/{total}  -  Step 2: touch the marker centre")
        print("=" * 62)

        contact_pose, _ = self.touch_marker_at(
            marker_info, phase=f"sample_{index+1}")

        if contact_pose is None:
            print("自动流程未取得接触，转手动 jog。")
            contact_pose, above, _ = self.jog_to_marker(marker_info)
            if contact_pose is None:
                print("Sample failed.")
                return None
            self._retract(self.RETRACT_MM)
            self.contact.close_window()

        p_cam = marker_info['p_cam']
        p_base = np.asarray(contact_pose[:3], float) / 1000.0

        print("\nSample recorded:")
        print(f"  p_cam  (m): {np.round(p_cam, 4)}")
        print(f"  p_base (m): {np.round(p_base, 4)}")

        # Lift to a clearly safe height so it doesn't block the camera view for the next sample
        safe = np.asarray(contact_pose, float).copy()
        safe[2] = max(self.TRAVEL_Z_MM, safe[2] + 120.0)
        self._return_to_pose(safe)

        return p_cam, p_base

    # ==================================================================
    def _warn_fixed_point(self):
        """
        The most dangerous trap when using the old calibration for coarse
        positioning: if every sample presses directly on the point predicted
        by the old calibration and never gets corrected, then
        p_base == T_old @ p_cam holds exactly every time, so the new
        solution will exactly reproduce the old one -- the residuals will
        look great, but the accuracy hasn't improved at all.
        """
        if not self.applied_corrections_mm:
            return
        corr = np.array(self.applied_corrections_mm)
        print(f"\nApplied XY corrections per sample (mm): "
              f"{np.round(corr, 2)}")
        if np.all(corr < 0.5):
            print("  WARNING: 所有样本都几乎没有纠偏 —— "
                  "新标定基本上是在复制上一次的结果，")
            print("  残差好看但精度未必提高。请确认每次都真的在图像里"
                  "核对了工具落点。")

    # ==================================================================
    def calibrate(self, num_samples=4):
        print("\n" + "=" * 62)
        print(f"Starting calibration with {num_samples} samples")
        print("=" * 62)
        print("每次采样之间把 marker 挪到明显不同的位置 (>=100mm)，")
        print("并且至少有 1~2 次把它垫高 30~80mm。")
        if num_samples < 5:
            print("提示：4 个点只比 6 自由度多 6 个方程，残差会严重乐观。"
                  "条件允许时建议 6~8 个点。")

        points_cam, points_base = [], []
        self.applied_corrections_mm = []

        for i in range(num_samples):
            sample = self.capture_sample(i, num_samples)
            if sample is None:
                print("Sample collection failed. Calibration aborted.")
                self.T_base_cam = None
                return False
            points_cam.append(sample[0])
            points_base.append(sample[1])

            if i < num_samples - 1:
                print("\n" + "-" * 62)
                print(f"Prepare sample {i+2}/{num_samples}")
                print("1. 确认机械臂已抬到安全高度")
                print("2. 把 marker 移到一个新位置（与其它点相距 >=100mm）")
                # Raise samples 3 and 5 (index 2, 4) to ensure the points are not coplanar
                if (i + 1) in (2, 4):
                    print("   *** 这一组请把 marker 垫高 30~80mm（书/盒子）"
                          " —— 破坏共面，给旋转和 Z 提供约束！***")
                print("3. 确认相机能看到 marker")
                input("Press Enter when ready...")

        points_cam = np.array(points_cam)
        points_base = np.array(points_base)

        print("\n" + "=" * 62)
        print("Solving hand-eye calibration (point-set registration)")
        print("=" * 62)

        check_point_geometry(points_cam)

        R_mat, t_vec = kabsch_umeyama(points_cam, points_base)
        self.T_base_cam = np.eye(4)
        self.T_base_cam[:3, :3] = R_mat
        self.T_base_cam[:3, 3] = t_vec

        print("\nFinal T_base_cam:")
        print(np.array2string(self.T_base_cam, precision=6,
                              suppress_small=True))

        # ---------- Residuals ----------
        predicted = (R_mat @ points_cam.T).T + t_vec
        residuals = predicted - points_base
        norms = np.linalg.norm(residuals, axis=1) * 1000.0
        print("\nPer-sample residuals (mm):")
        for i, (res, n) in enumerate(zip(residuals * 1000.0, norms)):
            print(f"  Sample {i+1}: dX={res[0]:+7.2f} dY={res[1]:+7.2f} "
                  f"dZ={res[2]:+7.2f}  |e|={n:6.2f}")
        rmse = float(np.sqrt(np.mean(norms ** 2)))
        print(f"\nFit RMSE     : {rmse:.2f} mm   (乐观值)")
        print(f"Max residual : {np.max(norms):.2f} mm")

        # ---------- Scale diagnostic ----------
        try:
            _, _, scale = kabsch_umeyama(points_cam, points_base,
                                         with_scale=True)
            print(f"\nDiagnostic scale (should be ~1.000): {scale:.4f}")
            if abs(scale - 1.0) > 0.02:
                print(f"  WARNING: 尺度偏离 {100*(scale-1):+.1f}%。")
                print("  最常见原因：marker_size 填错（必须是 ArUco 黑色"
                      "外框的实测边长，不是整块打印件的边长），")
                print("  或者相机焦距 fx/fy 不准。修正后重标定。")
        except Exception as exc:
            print("scale diagnostic failed:", exc)

        # ---------- Cross-validation ----------
        loo_cross_validation(points_cam, points_base)
        self._warn_fixed_point()

        # ---------- Sanity checks ----------
        det = np.linalg.det(R_mat)
        print(f"\ndet(R) = {det:.6f} (must be +1)")
        print(f"||R^T R - I|| = "
              f"{np.linalg.norm(R_mat.T @ R_mat - np.eye(3)):.2e}")
        print(f"Camera-to-base distance = {np.linalg.norm(t_vec):.3f} m")
        cam_z = R_mat[:, 2]
        print(f"Camera optical axis in base = {np.round(cam_z, 3)}")
        if cam_z[2] > 0.3:
            print("  WARNING: 相机看起来朝上，通常意味着符号/约定错误。")

        ts = time.strftime("%Y%m%d_%H%M%S")
        fname = f"hand_eye_calib_{ts}_{num_samples}samp.npz"
        np.savez(fname, T_base_cam=self.T_base_cam,
                 points_cam=points_cam, points_base=points_base,
                 residuals_mm=residuals * 1000.0, rmse_mm=rmse)
        print(f"\nSaved to '{fname}'")
        return True

    # ==================================================================
    def test_calibration(self):
        print("\n" + "=" * 62)
        print("Testing calibration")
        print("=" * 62)
        if self.T_base_cam is None:
            print("Error: no calibration available.")
            return False

        input("把 marker 放到一个全新的位置，确认可见，然后回车...")

        marker_info = self.get_marker_pose()
        if marker_info is None:
            print("Marker detection failed during test.")
            return False

        # For testing, only do "predict -> press", no correction --
        # otherwise this wouldn't be measuring calibration accuracy anymore.
        saved_iters = self.ALIGN_MAX_ITERS
        self.ALIGN_MAX_ITERS = 1
        try:
            contact_pose, predicted = self.touch_marker_at(
                marker_info, phase="test")
        finally:
            self.ALIGN_MAX_ITERS = saved_iters

        if contact_pose is None:
            print("Test failed: no contact.")
            return False

        contact_xyz = np.asarray(contact_pose[:3], float)
        error = contact_xyz - predicted

        print("\n=== Test Result ===")
        print(f"Predicted contact (mm): {np.round(predicted, 2)}")
        print(f"Actual contact    (mm): {np.round(contact_xyz, 2)}")
        print(f"Error             (mm): dX={error[0]:+.2f} "
              f"dY={error[1]:+.2f} dZ={error[2]:+.2f}")
        print("\n注意：这里的 dX/dY 恒为 0（机器人就是走到了预测点），")
        print("真正的横向精度要看纠偏窗口里显示的 correction 值，")
        print("或者看上面的 LOO 交叉验证结果。dZ 反映的是表面高度预测误差。")

        # Let the operator read the real lateral error from the correction window
        img = self._color_image()
        if img is not None:
            T_use = self.T_base_cam
            dxy, _ = self.visual_xy_correction(img, marker_info, T_use)
            print(f"Measured lateral error (mm): "
                  f"dX={dxy[0]:+.2f} dY={dxy[1]:+.2f}  "
                  f"|dXY|={np.linalg.norm(dxy):.2f}")

        ts = time.strftime("%Y%m%d_%H%M%S")
        fname = f"calibration_test_{ts}.npz"
        np.savez(fname, T_base_cam=self.T_base_cam,
                 p_cam=marker_info['p_cam'],
                 predicted_xyz_mm=predicted,
                 contact_pose=contact_pose, error_mm=error)
        print(f"\nTest result saved to '{fname}'")
        return True

    # ==================================================================
    def test_calibration_two_markers(self, marker_ids=(20, 21)):
        """
        Test stage: have the xArm use the same calibration to touch two
        markers laid flat on the table, one after another. The calibration
        stage itself is completely unchanged -- this just extends the
        single-point test into "touch multiple points in sequence", to
        demonstrate whether the calibration can autonomously and accurately
        find and touch multiple new targets.

        Key design point: detect and record both markers first, then touch
        them in sequence. Because touching the first one may have the
        arm/DIGIT block the camera's view of the second, detecting first
        and touching afterward avoids being affected by that.

        After touching both, the arm stops at a safe height above the second marker.

        Note: since both markers are on the table, they are coplanar -- the
        value of this step is in "automated demonstration", not in improving
        calibration accuracy (the rotational constraint is unchanged).
        """
        print("\n" + "=" * 62)
        print(f"Two-marker touch test  (IDs: {marker_ids[0]} then "
              f"{marker_ids[1]})")
        print("=" * 62)
        if self.T_base_cam is None:
            print("Error: no calibration available.")
            return False

        input(f"把两个 marker (ID {marker_ids[0]} 和 {marker_ids[1]}) "
              "都平放在桌面上，确认相机都能看到，然后回车...")

        # ---------- Stage 1: detect and record both markers first ----------
        detected = []
        for mid in marker_ids:
            print(f"\n--- Detecting marker {mid} ---")
            info = self.get_marker_pose(marker_id=mid)
            if info is None:
                print(f"Marker {mid} detection failed. Aborting test.")
                return False
            detected.append((mid, info))
            print(f"Marker {mid} recorded at cam (m): "
                  f"{np.round(info['p_cam'], 4)}")

        print("\n两个 marker 都已记录。开始依次触碰。")

        # ---------- Stage 2: touch each one in sequence ----------
        results = []
        for k, (mid, info) in enumerate(detected):
            print("\n" + "-" * 62)
            print(f"Touching marker {mid}  ({k+1}/{len(detected)})")
            print("-" * 62)

            # Same as the single-point test: only "predict->press", no
            # correction, so this actually measures calibration accuracy.
            saved_iters = self.ALIGN_MAX_ITERS
            self.ALIGN_MAX_ITERS = 1
            try:
                contact_pose, predicted = self.touch_marker_at(
                    info, phase=f"test_marker_{mid}")
            finally:
                self.ALIGN_MAX_ITERS = saved_iters

            if contact_pose is None:
                print(f"No contact on marker {mid}.")
                results.append((mid, None, predicted, None))
            else:
                contact_xyz = np.asarray(contact_pose[:3], float)
                error = contact_xyz - predicted
                print(f"\nMarker {mid} result:")
                print(f"  Predicted (mm): {np.round(predicted, 2)}")
                print(f"  Actual    (mm): {np.round(contact_xyz, 2)}")
                print(f"  dZ (surface height error, mm): {error[2]:+.2f}")

                # Read out the real lateral error
                img = self._color_image()
                lateral = None
                if img is not None:
                    dxy, _ = self.visual_xy_correction(
                        img, info, self.T_base_cam)
                    lateral = float(np.linalg.norm(dxy))
                    print(f"  Measured lateral error (mm): "
                          f"dX={dxy[0]:+.2f} dY={dxy[1]:+.2f} "
                          f"|dXY|={lateral:.2f}")
                results.append((mid, contact_pose, predicted, lateral))

            # After touching this marker, lift to a safe height before moving to the next one
            cur = self._get_pose()
            if cur is not None:
                self._move_to([cur[0], cur[1],
                               max(self.TRAVEL_Z_MM, cur[2] + 120.0)])

        # ---------- Stop at a safe height above the second marker ----------
        last_touched = None
        for mid, cp, pred, lat in reversed(results):
            if cp is not None:
                last_touched = cp
                break
        if last_touched is not None:
            print("\n停在最后触碰的 marker 上方安全高度。")
            self._move_to([last_touched[0], last_touched[1],
                           self.TRAVEL_Z_MM])
        self.contact.close_window()

        # ---------- Summary + save ----------
        print("\n" + "=" * 62)
        print("Two-marker test summary")
        print("=" * 62)
        for mid, cp, pred, lat in results:
            if cp is None:
                print(f"  Marker {mid}: NO CONTACT")
            else:
                lat_s = f"{lat:.2f} mm" if lat is not None else "n/a"
                print(f"  Marker {mid}: touched, lateral error = {lat_s}")

        ts = time.strftime("%Y%m%d_%H%M%S")
        fname = f"two_marker_test_{ts}.npz"
        np.savez(
            fname,
            T_base_cam=self.T_base_cam,
            marker_ids=np.array(marker_ids),
            p_cam_list=np.array([info['p_cam'] for _, info in detected]),
            predicted_list=np.array(
                [r[2] if r[2] is not None else [np.nan] * 3
                 for r in results]),
            contact_list=np.array(
                [r[1] if r[1] is not None else [np.nan] * 6
                 for r in results]),
            lateral_err_mm=np.array(
                [r[3] if r[3] is not None else np.nan for r in results]),
        )
        print(f"\nSaved to '{fname}'")
        return True

    # ==================================================================
    def tune_contact(self):
        """
        An on-site parameter-tuning tool (strongly recommended before the
        very first run). Workflow: have the arm hover above the marker ->
        run this function -> jog down manually in small steps while
        watching the area% in the window:
          - area should be < 0.3% when fully off the surface
          - the area at the moment of first contact -> set as CONTACT_AREA_ON_PCT
          - the area when it "feels just right" -> set as PRESS_TARGET_AREA_PCT
        Keys: r/f = +Z/-Z 0.2mm, b = re-capture baseline, q = quit.
        """
        print("\n--- CONTACT TUNING ---")
        print("  r/f = up/down 0.2mm   b = re-baseline   q = quit")
        if not self.contact.capture_baseline(status="tuning baseline"):
            return
        while True:
            area, cen, _ = self.contact.measure(status="TUNING  r/f=+-0.2mm")
            off = self.contact.contact_offset_mm(cen)
            pose = self._get_pose()
            print(f"  Z={pose[2]:8.3f} mm   area={area:6.2f}%   "
                  f"blob offset={np.round(off, 1) if off is not None else '-'}")
            k = cv2.waitKey(150) & 0xFF
            if k == ord('f'):
                self._move_relative([0, 0, -0.2],
                                    speed=self.DESCEND_SPEED_SLOW)
            elif k == ord('r'):
                self._move_relative([0, 0, 0.2],
                                    speed=self.DESCEND_SPEED_SLOW)
            elif k == ord('b'):
                self.contact.capture_baseline(status="re-baseline")
            elif k == ord('q'):
                break
        self.contact.close_window()

    # ==================================================================
    def shutdown(self):
        self.contact.close_window()
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass
        for closer in (lambda: self.pipeline.stop(),
                       lambda: self.digit.disconnect(),
                       lambda: self.arm.disconnect()):
            try:
                closer()
            except Exception:
                pass
        print("\nSystem resources released.")


# ===========================================================================
if __name__ == "__main__":
    calibrator = None
    try:
        calibrator = HandEyeCalibrator(
            arm_ip='192.168.1.222',
            marker_size=0.1,        # <-- must be the measured side length of the ArUco black border
            marker_id=20,
            dry_run=False,
            use_previous_calibration=True,
        )

        # For first-time use, set this to True first to tune the contact
        # thresholds before running the actual calibration.
        TUNE_ONLY = False
        if TUNE_ONLY:
            calibrator.tune_contact()
            raise SystemExit(0)

        # Press on a marker corner to calibrate the DIGIT's real-center
        # offset relative to the TCP (two stages: automatic convergence +
        # manual fine-tuning). This is independent of the main calibration
        # flow -- it just uses the existing calibration/last saved file, no
        # need to run calibrate() first.
        TUNE_DIGIT_CENTER = False
        TUNE_DIGIT_CORNER_INDEX = 0  # 0=top-left 1=top-right 2=bottom-right 3=bottom-left
        if TUNE_DIGIT_CENTER:
            info = calibrator.get_marker_pose()
            if info is None:
                print("Marker detection failed. 中止。")
                raise SystemExit(0)
            corner_p_cam = calibrator.marker_corner_p_cam(
                info, corner_index=TUNE_DIGIT_CORNER_INDEX)
            result = calibrator.calibrate_digit_center_at_corner(
                corner_p_cam)
            print("\n结果:", result)
            raise SystemExit(0)

        # Whether the test stage touches two markers in sequence
        # (calibration is still single-marker).
        # True  -> touch ID 20 then ID 21, stop at a safe height above the second.
        # False -> the original single-marker test.
        TWO_MARKER_TEST = True
        SECOND_MARKER_IDS = (20, 21)

        if calibrator.calibrate(num_samples=6):
            print("\nCalibration finished. Starting the test...")
            if TWO_MARKER_TEST:
                ok = calibrator.test_calibration_two_markers(
                    marker_ids=SECOND_MARKER_IDS)
            else:
                ok = calibrator.test_calibration()
            if ok:
                print("\nComplete workflow finished successfully.")
            else:
                print("\nCalibration OK, but the test failed.")
        else:
            print("\nCalibration failed. Test skipped.")

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    except Exception as exc:
        print(f"\nError: {exc}")
        import traceback
        traceback.print_exc()
    finally:
        if calibrator is not None:
            calibrator.shutdown()
