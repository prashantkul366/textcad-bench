"""
TextCAD-Bench unified evaluation metrics.
CD x10^2, IR, Voxel IoU — all under a single frozen protocol.
"""

import numpy as np
import trimesh
import signal
from scipy.spatial import cKDTree


def load_and_normalize(path):
    """Load OBJ/STL mesh and normalise to [-0.5, 0.5]^3."""
    mesh = trimesh.load(path, force='mesh')
    if mesh is None or mesh.is_empty or len(mesh.vertices) == 0:
        return None
    b = mesh.bounds
    scale = (b[1] - b[0]).max()
    if scale < 1e-8:
        return None
    mesh.apply_translation(-(b[0] + b[1]) / 2)
    mesh.apply_scale(1.0 / scale)
    return mesh


def chamfer_distance(m1, m2, n_points=8192):
    """
    Symmetric Chamfer Distance between two meshes.
    Returns (mean_cd, median_cd) both scaled by x10^2.
    """
    p1, _ = trimesh.sample.sample_surface(m1, n_points)
    p2, _ = trimesh.sample.sample_surface(m2, n_points)
    d1 = cKDTree(p2).query(p1)[0]
    d2 = cKDTree(p1).query(p2)[0]
    mean_cd   = float(np.mean(d1) + np.mean(d2)) * 100
    median_cd = float(np.median(d1) + np.median(d2)) * 100
    return mean_cd, median_cd


def voxel_iou(m1, m2, resolution=64):
    """
    Volumetric IoU at 64^3 resolution.
    Only meaningful for watertight meshes.
    Returns float in [0, 100] or None on failure.
    """
    try:
        v1 = m1.voxelized(pitch=1.0 / resolution).fill().matrix
        v2 = m2.voxelized(pitch=1.0 / resolution).fill().matrix
        # Pad to same shape
        s = tuple(max(a, b) for a, b in zip(v1.shape, v2.shape))
        a = np.zeros(s, bool)
        b = np.zeros(s, bool)
        a[:v1.shape[0], :v1.shape[1], :v1.shape[2]] = v1
        b[:v2.shape[0], :v2.shape[1], :v2.shape[2]] = v2
        inter = (a & b).sum()
        union = (a | b).sum()
        return float(inter) / float(union) * 100 if union > 0 else 0.0
    except Exception:
        return None


def with_timeout(fn, seconds=10):
    """Run fn with a SIGALRM timeout. Returns None on timeout."""
    class TimeoutError(Exception):
        pass

    def handler(signum, frame):
        raise TimeoutError()

    signal.signal(signal.SIGALRM, handler)
    signal.alarm(seconds)
    try:
        result = fn()
        signal.alarm(0)
        return result
    except TimeoutError:
        return None
    except Exception:
        signal.alarm(0)
        return None


def evaluate_pair(pred_path, gt_path, timeout_mesh=10, timeout_iou=15):
    """
    Evaluate one prediction against one ground truth.

    Returns dict with:
        ir_compile   : 1 if pred_path is None/empty, else 0
        ir_brepcheck : 1 if mesh is not watertight, else 0
        watertight   : bool
        cd_mean      : float or None
        cd_median    : float or None
        iou          : float or None
    """
    result = dict(
        ir_compile=1, ir_brepcheck=1,
        cd_mean=None, cd_median=None,
        iou=None, watertight=False
    )

    if not pred_path or not hasattr(pred_path, '__len__'):
        return result
    try:
        import os
        if not os.path.exists(pred_path) or os.path.getsize(pred_path) == 0:
            return result
    except Exception:
        return result

    result['ir_compile'] = 0
    pred = with_timeout(lambda: load_and_normalize(pred_path), timeout_mesh)
    if pred is None:
        return result

    result['ir_brepcheck'] = 0 if pred.is_watertight else 1
    result['watertight']   = bool(pred.is_watertight)

    gt = with_timeout(lambda: load_and_normalize(gt_path), timeout_mesh)
    if gt is None:
        return result

    try:
        result['cd_mean'], result['cd_median'] = chamfer_distance(pred, gt)
    except Exception:
        pass

    if pred.is_watertight and gt.is_watertight:
        result['iou'] = with_timeout(lambda: voxel_iou(pred, gt), timeout_iou)

    return result
