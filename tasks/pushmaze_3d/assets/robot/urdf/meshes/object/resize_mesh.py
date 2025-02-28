"""Script to normalize mesh (stl only)"""
import argparse
import copy

import numpy as np
import trimesh


def _resize_mesh(mesh_in: trimesh.Trimesh, size: float) -> trimesh.Trimesh:
    """Resizes the mesh so that the diametrical size is equal to size specified.

    Resizing assumes symmetrical mesh.
    """

    vertices = mesh_in.vertices
    assert isinstance(vertices, np.ndarray), "Vertices is not numpy.ndarray"
    assert len(vertices) > 0, "Empty vertices"
    center = np.mean(vertices, axis=0)
    vertices_centered = vertices - center
    rmax = np.max(np.linalg.norm(vertices_centered, axis=1))
    if rmax > 0:
        vertices_centered_normalized = size * 0.5 * vertices_centered / rmax
    else:
        raise ValueError(
            "Something went wrong. 'rmax' must be positive real number greater \
            than zero."
        )
    mesh_out = copy.deepcopy(mesh_in)
    mesh_out.vertices = vertices_centered_normalized

    return mesh_out


def main(mesh_file_in: str, size: float, mesh_file_out: str) -> None:
    """Wraps _normalize_mesh() peforming I/O."""

    mesh_in = trimesh.load(mesh_file_in)
    mesh_out = _resize_mesh(mesh_in, size)
    with open(mesh_file_out, "wb") as outfile:
        trimesh.exchange.export.export_mesh(
            mesh_out, file_obj=outfile, file_type="stl"
        )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("mesh_in", type=str)
    parser.add_argument("--size", type=np.float64)
    parser.add_argument("--out", type=str, default="out.stl")
    args = parser.parse_args()
    main(mesh_file_in=args.mesh_in, size=args.size, mesh_file_out=args.out)
