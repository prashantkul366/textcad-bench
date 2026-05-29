"""
Shared CadQuery execution wrapper for TextCAD-Bench.
Used by: run_qwen7b.py, run_llama8b.py, run_t2cq.py,
         run_cadrille.py, run_procad.py
"""

import os
import re
import subprocess
import trimesh


def execute_cadquery_code(code, uid_safe, timeout=20, result_var='result'):
    """
    Execute CadQuery Python code and export result to OBJ.

    Args:
        code        : Python code string
        uid_safe    : safe filename stem (no slashes)
        timeout     : subprocess timeout in seconds
        result_var  : variable name to look for ('result' or 'r' for ProCAD)

    Returns:
        (obj_path, True)  on success
        (None, False)     on failure
    """
    stl_path    = f"/tmp/cq_{uid_safe}.stl"
    obj_path    = f"/tmp/cq_{uid_safe}.obj"
    script_path = f"/tmp/cq_{uid_safe}.py"

    # Clean code
    code = re.sub(r'show_object\s*\(.*?\)', '', code)
    code = re.sub(r'```python\n?|```\n?', '', code).strip()

    export_block = f"""
import cadquery as _cq
_stl = "{stl_path}"
_obj = None
# Try named result variable first
for _k, _v in list(globals().items()):
    if _k == '{result_var}' and isinstance(_v, (_cq.Workplane, _cq.Assembly)):
        _obj = _v
        break
# Fall back to last assigned Workplane/Assembly
if _obj is None:
    _cands = [v for k, v in list(globals().items())
              if not k.startswith('_')
              and isinstance(v, (_cq.Workplane, _cq.Assembly))]
    if _cands:
        _obj = _cands[-1]
if _obj is not None:
    try:
        _obj.val().exportStl(_stl)
    except Exception:
        try:
            _obj.exportStl(_stl)
        except Exception:
            pass
"""

    with open(script_path, 'w') as f:
        f.write(code + "\n" + export_block)

    try:
        subprocess.run(
            ['python', script_path],
            timeout=timeout,
            capture_output=True,
            text=True
        )
        if os.path.exists(stl_path) and os.path.getsize(stl_path) > 100:
            mesh = trimesh.load(stl_path, force='mesh')
            if mesh and not mesh.is_empty and len(mesh.vertices) > 0:
                mesh.export(obj_path)
                return obj_path, True
        return None, False
    except subprocess.TimeoutExpired:
        return None, False
    except Exception:
        return None, False
    finally:
        for p in [script_path, stl_path]:
            try:
                os.remove(p)
            except Exception:
                pass
