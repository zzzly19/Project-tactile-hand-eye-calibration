#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Evaluation Bench — Touch vs Non-Touch Calibration
Fair comparison between the tactile calibration and the classical
pure-vision calibration
==========================================================

Why it is needed
-----------------
The tactile version reports LOO cross-validation accuracy (0.39mm), while
the classical version reports AX=XB residuals (tens of mm). These two
numbers **cannot be compared directly**, because they measure different
things. For a fair comparison, both calibrations must be evaluated
separately using **the same set of test points and the same evaluation
method**.

What it does
------------
Place the marker at N new positions, and for each position:
  1. Detect once with the camera -> p_cam;
  2. Repeatedly touch with closed-loop visual correction until convergence
     -> obtain an independent, trustworthy ground-truth base-frame position
     p_base_true (a single touch cannot be used: descend_to_contact only
     searches along Z and never corrects XY, so the coordinate recorded by
     a single touch is always identical to the predicted coordinate it was
     commanded to — it cannot reveal true accuracy. touch_marker_converged
     instead converges to <0.2mm through repeated "touch -> visual
     correction" cycles, giving an independent ground truth that does not
     depend on either calibration's own prediction);
  3. Predict that point using the **tactile calibration T** -> tactile
     lateral error;
  4. Predict that point using the **classical calibration T** -> classical
     lateral error;
  Both T's use the same p_cam and the same independent ground truth, so
  the errors can be compared directly.

Output
------
For each test point, the lateral/depth error of both methods, followed by
the mean lateral error of both methods — this is the fair comparison
number that can go into the report.

Usage
-----
    python3 unified_evaluation.py
Fill in the two calibration file paths below as TOUCH_CALIB / CLASSICAL_CALIB.
No calibration is performed here (existing results are loaded directly);
only test touches are performed.
"""

import time
import glob
import os
import numpy as np

from TactiCal import HandEyeCalibrator


class UnifiedEvaluator(HandEyeCalibrator):

    def __init__(self, touch_calib, classical_calib, *args, **kwargs):
        # The evaluation bench does not recalibrate; disable auto-loading
        # of any previous calibration to avoid interference
        kwargs.setdefault('use_previous_calibration', False)
        super().__init__(*args, **kwargs)

        self.T_touch = self._load_T(touch_calib, 'T_base_cam')
        # Classical version: load all four methods (per Niklas's suggestion
        # -- don't pick based on agreement between methods, but instead
        # evaluate all four methods against the tactile ground truth and
        # pick whichever is actually most accurate)
        self.T_classical_all = self._load_classical_all(classical_calib)
        # Key point: to convert a camera-frame point p_cam into the base
        # frame, the matrix must go in the "camera->base" direction. In the
        # classical calibration file this is T_cam2base_best (not
        # T_base_cam_best, which is the inverse -- the opposite direction;
        # using it for prediction would be off by hundreds of mm).
        self.T_classical = self._load_T(classical_calib, 'T_cam2base_best')

        if self.T_touch is None or self.T_classical is None:
            raise SystemExit("两个标定必须都能加载。检查文件路径和字段名。")

        # Use the tactile calibration to locate the marker for touching
        # (it is more accurate, ensuring the arm can safely find the marker)
        self.T_base_cam = self.T_touch
        print("\n两个标定已加载，评估台就绪。")
        if self.T_classical_all:
            print(f"  经典版含 {len(self.T_classical_all)} 种方法: "
                  f"{list(self.T_classical_all.keys())}")

    # ------------------------------------------------------------------
    def _load_classical_all(self, path):
        """
        Load all four methods of the classical calibration. The
        T_cam2base_all stored in the file is already in the "camera->base"
        direction, which is exactly the direction needed to convert p_cam
        to base, so it is used directly without inversion.
        Returns {method_name: T_cam2base}.
        """
        if path is None or not os.path.exists(path):
            return {}
        try:
            data = np.load(path, allow_pickle=True)
            if 'T_cam2base_all' not in data.files:
                print("经典标定文件里没有 T_cam2base_all，只能用最优方法。")
                return {}
            all_c2b = data['T_cam2base_all'].item()  # dict, camera->base
            return dict(all_c2b)   # used directly, no inversion needed
        except Exception as exc:
            print(f"加载四种方法失败: {exc}")
            return {}

    # ------------------------------------------------------------------
    def _load_T(self, path, field):
        """Load a calibration matrix. path can be a specific file, or None (take the latest)."""
        if path is None:
            return None
        if not os.path.exists(path):
            print(f"文件不存在: {path}")
            return None
        data = np.load(path, allow_pickle=True)
        if field not in data.files:
            print(f"WARNING: field '{field}' not found in {path}. "
                  f"Available: {data.files} / "
                  f"字段'{field}'不在{path}。可用: {data.files}")
            return None
        T = data[field]
        print(f"Loaded '{field}' from {path}")
        return T

    # ------------------------------------------------------------------
    def _predict_mm(self, T, p_cam):
        """Use calibration T to predict a camera-frame point as base-frame coordinates (mm)."""
        return (T[:3, :3] @ p_cam + T[:3, 3]) * 1000.0

    # ------------------------------------------------------------------
    def evaluate(self, n_positions=8):
        print("\n" + "=" * 62)
        print(f"Unified evaluation over {n_positions} test positions")
        print("=" * 62)
        print("每个位置：检测 -> 闭环收敛触碰得独立真值 -> 两个标定各"
              "预测一次并比较。")

        rows = []
        for i in range(n_positions):
            print("\n" + "-" * 62)
            print(f"Test position {i+1}/{n_positions}")
            print("-" * 62)
            input("把 marker 放到一个新位置，确认相机可见，然后回车...")

            info = self.get_marker_pose()
            if info is None:
                print("检测失败，跳过该位置。")
                continue
            p_cam = info['p_cam']

            # Prediction from each of the two calibrations (using the same p_cam)
            pred_touch = self._predict_mm(self.T_touch, p_cam)
            pred_classical = self._predict_mm(self.T_classical, p_cam)
            print(f"  touch 预测    (mm): {np.round(pred_touch, 2)}")
            print(f"  classical 预测(mm): {np.round(pred_classical, 2)}")

            # Repeatedly touch with closed-loop visual correction, converging
            # to an independent ground truth -- a single touch is not used:
            # descend_to_contact only searches along Z and never corrects
            # XY, so the coordinate recorded by a single touch is always
            # identical to the predicted coordinate it was commanded to go
            # to (i.e. pred_touch itself), which has no independent
            # validation value.
            contact_pose, dxy_history = self.touch_marker_converged(
                info, tol_mm=0.2, max_iters=5)

            if contact_pose is None:
                print("触碰失败，跳过该位置。")
                continue
            p_true = np.asarray(contact_pose[:3], float)
            print(f"  独立真值 (mm): {np.round(p_true, 2)}  "
                  f"(收敛用了 {len(dxy_history)} 次)")

            # Error of both methods (against the same ground truth)
            err_touch = pred_touch - p_true
            err_classical = pred_classical - p_true
            lat_touch = float(np.linalg.norm(err_touch[:2]))
            lat_classical = float(np.linalg.norm(err_classical[:2]))

            print(f"  >>> touch     横向误差: {lat_touch:.2f} mm  "
                  f"(dZ {err_touch[2]:+.2f})")
            print(f"  >>> classical 横向误差: {lat_classical:.2f} mm  "
                  f"(dZ {err_classical[2]:+.2f})")

            row = {
                'pos': i + 1,
                'p_cam': p_cam.tolist(),
                'p_true_mm': p_true.tolist(),
                'ref_touch_iters': len(dxy_history),
                'touch_pred_mm': pred_touch.tolist(),
                'classical_pred_mm': pred_classical.tolist(),
                'touch_lateral_mm': lat_touch,
                'touch_dz_mm': float(err_touch[2]),
                'classical_lateral_mm': lat_classical,
                'classical_dz_mm': float(err_classical[2]),
            }

            # Record lateral error for each of the four classical methods
            # separately (against the same independent ground truth), so
            # "which method to pick" is based on external ground-truth
            # measurement rather than agreement between methods
            for name, T_m in self.T_classical_all.items():
                pred_m = self._predict_mm(T_m, p_cam)
                lat_m = float(np.linalg.norm((pred_m - p_true)[:2]))
                row[f'classical_{name}_lateral_mm'] = lat_m
                print(f"      [{name}] 横向误差: {lat_m:.2f} mm")

            rows.append(row)

            # Lift to a safe height before moving on to the next one
            cur = self._get_pose()
            if cur is not None:
                self._move_to([cur[0], cur[1],
                               max(self.TRAVEL_Z_MM, cur[2] + 120.0)])

        self.contact.close_window()

        if not rows:
            print("没有有效测试点。")
            return

        self._summarize(rows)
        self._save(rows)

    # ------------------------------------------------------------------
    def _summarize(self, rows):
        lt = np.array([r['touch_lateral_mm'] for r in rows])
        lc = np.array([r['classical_lateral_mm'] for r in rows])
        zt = np.array([abs(r['touch_dz_mm']) for r in rows])
        zc = np.array([abs(r['classical_dz_mm']) for r in rows])

        print("\n" + "=" * 62)
        print("FAIR COMPARISON SUMMARY (same test points, same truth)")
        print("=" * 62)
        print(f"{'':16}{'Touch':>12}{'Classical':>12}")
        print(f"{'lateral mean':16}{lt.mean():>10.2f}mm{lc.mean():>10.2f}mm")
        print(f"{'lateral max':16}{lt.max():>10.2f}mm{lc.max():>10.2f}mm")
        print(f"{'|dZ| mean':16}{zt.mean():>10.2f}mm{zc.mean():>10.2f}mm")
        print(f"\n测试点数: {len(rows)}")
        better = "Touch" if lt.mean() < lc.mean() else "Classical"
        diff = abs(lt.mean() - lc.mean())
        print(f"横向平均误差更小的是: {better}（相差 {diff:.2f} mm）")

        # ---- Use the tactile ground truth to pick the most accurate classical method (addressing Niklas's suggestion) ----
        method_names = [k.replace('classical_', '').replace('_lateral_mm', '')
                        for k in rows[0].keys()
                        if k.startswith('classical_')
                        and k.endswith('_lateral_mm')
                        and k != 'classical_lateral_mm']
        if method_names:
            print("\n各经典方法在触觉真值下的实测平均横向误差:")
            method_means = {}
            for name in method_names:
                vals = [r.get(f'classical_{name}_lateral_mm')
                        for r in rows
                        if r.get(f'classical_{name}_lateral_mm') is not None]
                if vals:
                    method_means[name] = float(np.mean(vals))
                    print(f"  {name:12s}: {method_means[name]:.2f} mm")
            if method_means:
                best = min(method_means, key=method_means.get)
                print(f"\n>>> 用触觉真值实测，最准的经典方法是: {best} "
                      f"({method_means[best]:.2f} mm)")
                print("    注意：这是用独立的触觉真值选的，不是靠方法间一致性。")

        print("\n这是在相同测试点、相同独立真值下的公平对比，")
        print("两种方法的横向误差可以直接比大小。")

    def _save(self, rows):
        import csv
        ts = time.strftime("%Y%m%d_%H%M%S")
        path = f"unified_evaluation_{ts}.csv"
        fields = ['pos', 'touch_lateral_mm', 'classical_lateral_mm',
                  'touch_dz_mm', 'classical_dz_mm', 'ref_touch_iters',
                  'p_true_mm', 'touch_pred_mm', 'classical_pred_mm', 'p_cam']
        with open(path, 'w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=fields, extrasaction='ignore')
            w.writeheader()
            w.writerows(rows)
        print(f"\nSaved to '{path}'")


# ===========================================================================
if __name__ == "__main__":
    # Paths to the two calibration files. None = automatically take the latest matching file in the current directory.
    TOUCH_CALIB = None       # e.g. 'hand_eye_calib_20260728_152804_6samp.npz'
    CLASSICAL_CALIB = None   # e.g. 'classical_handeye_20260804_135412.npz'

    # Automatically find the latest file
    if TOUCH_CALIB is None:
        c = sorted(glob.glob("hand_eye_calib_*.npz"), key=os.path.getmtime)
        TOUCH_CALIB = c[-1] if c else None
    if CLASSICAL_CALIB is None:
        c = sorted(glob.glob("classical_handeye_*.npz"), key=os.path.getmtime)
        CLASSICAL_CALIB = c[-1] if c else None

    print(f"Touch calibration    : {TOUCH_CALIB}")
    print(f"Classical calibration: {CLASSICAL_CALIB}")

    ev = None
    try:
        ev = UnifiedEvaluator(
            TOUCH_CALIB, CLASSICAL_CALIB,
            arm_ip='192.168.1.222',
            marker_size=0.1,
            marker_id=20,
            dry_run=False,
        )
        ev.evaluate(n_positions=8)
    except KeyboardInterrupt:
        print("\nInterrupted.")
    except Exception as exc:
        print(f"\nError: {exc}")
        import traceback
        traceback.print_exc()
    finally:
        if ev is not None:
            ev.shutdown()
