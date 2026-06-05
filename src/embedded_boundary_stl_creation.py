import trimesh
import numpy as np
from pathlib import Path
from src.utils.paths import ROOT_DIR


def make_coil_stl(b_dia, b_offset, L, N, full=True):
    """
    Generate a watertight STL file representing 6 polywell coil rings.
    
    Parameters
    ----------
    b_dia     : coil diameter (m)
    b_offset  : coil center distance from origin (m)
    dx        : cell size (m), used to set tube radius
    output_path : output STL file path
    """

    L = L
    dx = L / N if not full else 2 * L / N
    Path(ROOT_DIR / "coil_stls").mkdir(parents=True, exist_ok=True)
    output_path = Path(ROOT_DIR / "coil_stls" / f"coil_bdia{b_dia}_boffset{b_offset}_L{L}_N{N}_dx{dx}_full{full}.stl")
    if output_path.exists():
        print("stl file already exists for this configuration, skipping re-creation, loading from file.")
        mesh = trimesh.load_mesh(output_path)
        print(f"[stl] is_watertight: {mesh.is_watertight}")
        print(f"[stl] bounds: {mesh.bounds}")
        print(f"[stl] n_faces: {len(mesh.faces)}")
        mesh.close()
        return output_path
    major_radius = b_dia / 2
    minor_radius = dx

    coil_configs = [
        ([0, 0, +b_offset], [0, 0, 0]),          # +z, no rotation
        ([0, 0, -b_offset], [0, 0, 0]),          # -z, no rotation
        ([+b_offset, 0, 0], [0, 90, 0]),         # +x, rotate 90 around y
        ([-b_offset, 0, 0], [0, 90, 0]),         # -x, rotate 90 around y
        ([0, +b_offset, 0], [90, 0, 0]),         # +y, rotate 90 around x
        ([0, -b_offset, 0], [90, 0, 0]),         # -y, rotate 90 around x
    ]

    meshes = []
    for position, rotation_deg in coil_configs:
        torus = trimesh.creation.torus(
            major_radius=major_radius,
            minor_radius=minor_radius,
        )
        # apply rotation only when it is required
        rx, ry, rz = [np.deg2rad(d) for d in rotation_deg]
        if rx: torus.apply_transform(trimesh.transformations.rotation_matrix(rx, [1, 0, 0]))
        if ry: torus.apply_transform(trimesh.transformations.rotation_matrix(ry, [0, 1, 0]))
        if rz: torus.apply_transform(trimesh.transformations.rotation_matrix(rz, [0, 0, 1]))
        # apply translation
        torus.apply_translation(position)
        meshes.append(torus)

    combined = trimesh.util.concatenate(meshes)
    
    print(f"[stl] N = {N}, L = {L}")
    print(f"[stl] is_watertight: {combined.is_watertight}")
    print(f"[stl] bounds: {combined.bounds}")
    print(f"[stl] n_faces: {len(combined.faces)}")
    
    combined.export(output_path)
    print(f"[stl] saved to {output_path}")
    
    return output_path