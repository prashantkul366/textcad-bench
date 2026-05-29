"""
Mesh loading, normalisation, and sampling utilities for TextCAD-Bench.
"""
import numpy as np
def load_mesh(path: str):
    """Load an OBJ or STL mesh. Returns (vertices, faces) or None on failure."""
    try:
        import trimesh
        mesh = trimesh.load(path, force="mesh")
        if mesh.is_empty:
            return None
        return mesh
    except Exception:
        return None
def normalize_mesh(mesh):
    """Normalise mesh to fit in [-0.5, 0.5]^3 bounding box."""
    bounds = mesh.bounds          # shape (2, 3)
    center = (bounds[0] + bounds[1]) / 2.0
    scale  = (bounds[1] - bounds[0]).max()
    if scale < 1e-8:
        return mesh
    mesh = mesh.copy()
    mesh.vertices = (mesh.vertices - center) / scale
    return mesh
def sample_surface_points(mesh, n_points: int = 8192):
    """Sample n_points uniformly from the mesh surface."""
    points, _ = mesh.sample(n_points, return_index=True)
    return points.astype(np.float32)
def is_watertight(mesh) -> bool:
    """Return True if mesh is watertight (closed, no boundary edges)."""
    return mesh.is_watertight
