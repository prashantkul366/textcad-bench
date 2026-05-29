"""
CADmium JSON → trimesh converter for TextCAD-Bench.
Handles lines, arcs, circles, multi-face profiles,
and NewBody/Cut/Join boolean operations.
"""

import numpy as np
import trimesh
import cadquery as cq


def euler_to_rotation_matrix(angles_deg):
    """Convert Euler angles (degrees) to 3x3 rotation matrix (ZYX order)."""
    rx, ry, rz = [np.radians(a) for a in angles_deg]
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(rx), -np.sin(rx)],
                   [0, np.sin(rx),  np.cos(rx)]])
    Ry = np.array([[ np.cos(ry), 0, np.sin(ry)],
                   [0,           1, 0          ],
                   [-np.sin(ry), 0, np.cos(ry)]])
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0],
                   [np.sin(rz),  np.cos(rz), 0],
                   [0,           0,           1]])
    return Rz @ Ry @ Rx


def json_to_mesh(data, uid_safe):
    """
    Convert CADmium JSON output to a trimesh.Trimesh.

    Args:
        data     : parsed JSON dict with 'parts' key
        uid_safe : safe stem for temp file naming

    Returns:
        trimesh.Trimesh or None
    """
    compound = None

    for part_name in sorted(data.get('parts', {}).keys()):
        part = data['parts'][part_name]
        if part is None:
            continue

        cs  = part.get('coordinate_system', {})
        ext = part.get('extrusion', {})
        sk  = part.get('sketch', {})

        scale      = ext.get('sketch_scale', 1.0)
        depth_fwd  = ext.get('extrude_depth_towards_normal', 0.0)
        depth_back = ext.get('extrude_depth_opposite_normal', 0.0)
        total_d    = depth_fwd + depth_back

        if total_d < 1e-6:
            continue

        wires = []
        for face_name in sorted(sk.keys()):
            for loop_name in sorted(sk[face_name].keys()):
                loop  = sk[face_name][loop_name]
                edges = []

                for prim_name in sorted(loop.keys()):
                    prim = loop[prim_name]
                    if not isinstance(prim, dict):
                        continue
                    try:
                        if 'Mid Point' in prim:
                            # Arc through 3 points
                            sp = [v * scale for v in prim['Start Point']]
                            mp = [v * scale for v in prim['Mid Point']]
                            ep = [v * scale for v in prim['End Point']]
                            edges.append(cq.Edge.makeThreePointArc(
                                cq.Vector(sp[0], sp[1], 0),
                                cq.Vector(mp[0], mp[1], 0),
                                cq.Vector(ep[0], ep[1], 0)))
                        elif 'Center' in prim and 'Radius' in prim:
                            # Full circle
                            cx, cy = [v * scale for v in prim['Center']]
                            r = prim['Radius'] * scale
                            edges.append(cq.Edge.makeCircle(
                                r, cq.Vector(cx, cy, 0), cq.Vector(0, 0, 1)))
                        elif 'Start Point' in prim and 'End Point' in prim:
                            # Line segment
                            sp = [v * scale for v in prim['Start Point']]
                            ep = [v * scale for v in prim['End Point']]
                            edges.append(cq.Edge.makeLine(
                                cq.Vector(sp[0], sp[1], 0),
                                cq.Vector(ep[0], ep[1], 0)))
                    except Exception:
                        continue

                if edges:
                    try:
                        wires.append(cq.Wire.assembleEdges(edges))
                    except Exception:
                        continue

        if not wires:
            continue

        # Build face from wires (first = outer, rest = holes)
        try:
            face_shape = cq.Face.makeFromWires(
                wires[0], wires[1:] if len(wires) > 1 else [])
        except Exception:
            try:
                face_shape = cq.Face.makeFromWires(wires[0])
            except Exception:
                continue

        # Apply coordinate system
        R      = euler_to_rotation_matrix(cs.get('Euler Angles', [0, 0, 0]))
        normal = cq.Vector(*R[:, 2])
        origin = cq.Vector(*cs.get('Translation Vector', [0, 0, 0]))
        start  = origin - normal * depth_back if depth_back > 0 else origin

        try:
            solid = cq.Solid.extrudeLinear(face_shape, normal * total_d)
            solid = solid.moved(cq.Location(start))
        except Exception:
            continue

        op = ext.get('operation', 'NewBodyFeatureOperation')
        if compound is None:
            compound = solid
        else:
            try:
                if 'Cut' in op:
                    compound = cq.Shape(compound.wrapped.Cut(solid.wrapped))
                else:
                    compound = cq.Shape(compound.wrapped.Fuse(solid.wrapped))
            except Exception:
                continue

    if compound is None:
        return None

    stl_path = f"/tmp/{uid_safe}.stl"
    try:
        cq.Shape(compound.wrapped).exportStl(stl_path)
        mesh = trimesh.load(stl_path, force='mesh')
        import os
        if os.path.exists(stl_path):
            os.remove(stl_path)
        return mesh if (mesh and not mesh.is_empty) else None
    except Exception:
        return None
