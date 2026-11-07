"""Generates the twisted turbine-blade HEX8 mesh blade_050_000.inp used by experiments/5_blade.

Run with `python toga/mesh/blade/generate_twisted_blade_mesh.py` (needs gmsh); output is written next to this file.
"""
import contextlib
import math
import os

import gmsh
import meshio

MESH_DIR = os.path.dirname(os.path.abspath(__file__))


def make_blade_mesh(file_name, elem_size):

    # User parameters
    mesh_size = elem_size   # target edge length for the unstructured quad mesh

    r1 = 6. / 2     # radius of leading edge circle
    r2 = 2. / 2     # radius of trailing edge circle
    base_length = 40.0 # cross-sectional length of the blade

    twist_angle = math.pi / 8  # total twist
    span = 50. # extrusion length of the blade


    c1 = (r1, 0.0)  # center of leading edge circle
    c2 = (base_length - r2, 0.0) # center of trailing edge circle

    r3 = 160. / 2   # radius of pressure side tangent circle
    r4 = 140. / 2   # radius of suction side tangent circle

    twist_origin = (0.0, 0.0, -base_length / 2)  # origin for the twist operation

    n_span_layers = round(span / mesh_size)

    def tangent_circle_centers(c1, r1, c2, r2, r3, internal1=False, internal2=False, tol=1e-12):
        x1, y1 = c1
        x2, y2 = c2

        # Distance from solution circle center to each original circle center
        R1 = abs(r1 - r3) if internal1 else (r1 + r3)
        R2 = abs(r2 - r3) if internal2 else (r2 + r3)

        dx = x2 - x1
        dy = y2 - y1
        D = math.hypot(dx, dy) # Distance between original circle centers

        if D < tol:
            raise ValueError("Circle centers coincide.")

        if D > R1 + R2 + tol:
            print("No solutions: circles are too far apart.")
            return []
        if D < abs(R1 - R2) - tol:
            print("No solutions: one circle is contained within the other.")
            return []

        a = (R1**2 - R2**2 + D**2) / (2 * D)
        h_sq = R1**2 - a**2
        if h_sq < -tol:
            print("No solutions: circles do not intersect.")
            return []
        h_sq = max(h_sq, 0.0)
        h = math.sqrt(h_sq)

        xm = x1 + a * dx / D
        ym = y1 + a * dy / D

        rx = -dy * (h / D)
        ry =  dx * (h / D)

        p1 = (xm + rx, ym + ry)
        p2 = (xm - rx, ym - ry)

        if h <= tol:
            return [p1]
        return [p1, p2]


    def circle_tangency_point(c1, r1, c2, r2, tol=1e-12):
        """
        Return the tangency point of two tangent circles.

        Parameters
        ----------
        c1, c2 : tuple[float, float]
            Centers of the circles.
        r1, r2 : float
            Radii of the circles.
        tol : float
            Numerical tolerance.

        Returns
        -------
        tuple[float, float]
            Tangency point.

        Raises
        ------
        ValueError
            If the circles are not tangent.
        """
        x1, y1 = c1
        x2, y2 = c2

        dx = x2 - x1
        dy = y2 - y1
        d = math.hypot(dx, dy)

        if d < tol:
            raise ValueError("Coincident centers: tangency point is undefined.")

        # Check whether circles are tangent
        external = abs(d - (r1 + r2)) <= tol
        internal = abs(d - abs(r1 - r2)) <= tol

        if not (external or internal):
            raise ValueError("Circles are not tangent.")

        ux = dx / d
        uy = dy / d

        if external:
            # Point lies from c1 toward c2 at distance r1
            px = x1 + r1 * ux
            py = y1 + r1 * uy
        else:
            # Internal tangency
            # Tangency point still lies on the center-center line
            # If r1 >= r2, point is from c1 toward c2 by r1
            # else it is from c1 away from c2 by r1
            sign = 1.0 if r1 >= r2 else -1.0
            px = x1 + sign * r1 * ux
            py = y1 + sign * r1 * uy

        return (px, py)


    # Get pressure side tangent circle center
    sols = tangent_circle_centers(c1, r1, c2, r2, r3, internal1=False, internal2=False)
    for center in sols:
        if center[1] > 0:
            c3 = center
            break

    # Get suction side tangent circle center
    sols = tangent_circle_centers(c1, r1, c2, r2, r4, internal1=True, internal2=True)
    for center in sols:
        if center[1] > 0:
            c4 = center
            break

    # Get endpoints of the circular arcs to construct the blade cross-section
    p1 = (c1[0] - r1, c1[1])  # leftmost point of LE circle
    p2 = circle_tangency_point(c1, r1, c3, r3)  # upper tangent on LE circle
    p3 = circle_tangency_point(c2, r2, c3, r3)  # upper tangent on TE circle
    p4 = (c2[0] + r2, c2[1])  # rightmost point of TE circle
    p5 = circle_tangency_point(c2, r2, c4, r4)  # lower tangent on TE circle
    p6 = circle_tangency_point(c1, r1, c4, r4)  # lower tangent on LE circle

    gmsh.initialize()
    gmsh.model.add("turbine_blade")
    geo = gmsh.model.geo

    # Add points for the blade cross-section
    p1 = gmsh.model.geo.addPoint(p1[1], 0.0, -p1[0], mesh_size)
    p2 = gmsh.model.geo.addPoint(p2[1], 0.0, -p2[0], mesh_size)
    p3 = gmsh.model.geo.addPoint(p3[1], 0.0, -p3[0], mesh_size)
    p4 = gmsh.model.geo.addPoint(p4[1], 0.0, -p4[0], mesh_size)
    p5 = gmsh.model.geo.addPoint(p5[1], 0.0, -p5[0], mesh_size)
    p6 = gmsh.model.geo.addPoint(p6[1], 0.0, -p6[0], mesh_size)

    # Add points for the tangent circle centers (for defining circular arcs)
    c1 = gmsh.model.geo.addPoint(c1[1], 0.0, -c1[0], mesh_size)
    c2 = gmsh.model.geo.addPoint(c2[1], 0.0, -c2[0], mesh_size)
    c3 = gmsh.model.geo.addPoint(c3[1], 0.0, -c3[0], mesh_size)
    c4 = gmsh.model.geo.addPoint(c4[1], 0.0, -c4[0], mesh_size)

    # Add circular arcs for leading and trailing edges
    l1 = gmsh.model.geo.addCircleArc(p2, c1, p1)  # suction leading edge arc
    l2 = gmsh.model.geo.addCircleArc(p3, c3, p2)  # suction arc
    l3 = gmsh.model.geo.addCircleArc(p4, c2, p3)  # suction trailing edge arc
    l4 = gmsh.model.geo.addCircleArc(p5, c2, p4)  # pressure trailing edge arc
    l5 = gmsh.model.geo.addCircleArc(p6, c4, p5)  # pressure side arc
    l6 = gmsh.model.geo.addCircleArc(p1, c1, p6)  # pressure leading edge arc

    # Curve loop and surface
    cl = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4, l5, l6])
    surf = gmsh.model.geo.addPlaneSurface([cl])
    gmsh.model.geo.mesh.setRecombine(2, surf)

    # Mesh the surface
    gmsh.model.geo.synchronize()

    # Unstructured quad settings:
    # Algorithm 8 starts from an unstructured quad-oriented surface mesh.
    # RecombineAll converts triangle pairs into quads; SubdivisionAlgorithm=1
    # splits any remaining triangles, so the final 2-D elements are all quads.
    gmsh.option.setNumber("Mesh.Algorithm", 8)
    gmsh.option.setNumber("Mesh.RecombineAll", 1)
    gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 1)
    gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", 1)

    gmsh.model.mesh.generate(2)

    elem_types, elem_tags, _ = gmsh.model.mesh.getElements(dim=2)
    for elem_type, tags in zip(elem_types, elem_tags):
        elem_name = gmsh.model.mesh.getElementProperties(elem_type)[0]
        print(f"{elem_name}: {len(tags)}")

    out = gmsh.model.geo.twist(
        [(2, surf)],
        *twist_origin,            # twist center
        0, span, 0,              # translation vector
        0, 1, 0,                 # twist axis
        -twist_angle,             # twist angle
        [n_span_layers], [], True   # number of layers in the extrusion direction
    )

    geo.synchronize()

    gmsh.model.mesh.generate(3)
    gmsh.model.mesh.setOrder(1)

    inp_file = os.path.join(MESH_DIR, f"{file_name}.inp")
    with contextlib.chdir(MESH_DIR):  # gmsh writes the path it is given into the *Heading
        gmsh.write(f"{file_name}.inp")

    nsets = {
        "suction_leading_edge": (2, out[3][1]),
        "suction_side": (2, out[4][1]),
        "suction_trailing_edge": (2, out[5][1]),
        "pressure_trailing_edge": (2, out[6][1]),
        "pressure_side": (2, out[7][1]),
        "pressure_leading_edge": (2, out[2][1]),
        "bottom": (2, surf),
        "top": (2, out[0][1]),
    }

    with open(inp_file, "a") as f:
        for name, (dim, tag) in nsets.items():
            node_tags, _, _ = gmsh.model.mesh.getNodes(dim, tag, includeBoundary=True)
            f.write(f"*NSET,NSET={name}\n")
            for i, n in enumerate(node_tags):
                f.write(f" {n}," if (i + 1) % 16 != 0 else f" {n},\n")
            f.write("\n")

    gmsh.finalize()

    meshio_mesh = meshio.read(inp_file)
    print("Mesh summary:")
    print(f"  Nodes: {len(meshio_mesh.points)}")
    print(f"  Elements: {len(meshio_mesh.cells[-1])}")
    print(f"  Cell types: {[meshio_mesh.cells[i].type for i in range(len(meshio_mesh.cells))]}")
    print(f"  Physical groups: {meshio_mesh.cell_sets.keys()}")
    print(f"  Node sets: {meshio_mesh.point_sets.keys()}")

    if meshio_mesh.cells[-1].type != "hexahedron":
        print(f"WARNING: last cell type is {meshio_mesh.cells[-1].type}, expected 'hexahedron'.")

    meshio.write(os.path.join(MESH_DIR, f"{file_name}.vtu"), meshio_mesh)


if __name__ == "__main__":
    # The name encodes the approximate cell count; other sizes tried during development:
    # blade_003_000 (elem_size=1.5), blade_010_000 (0.98), blade_100_000 (0.45), blade_800_000 (0.223).
    make_blade_mesh("blade_050_000", elem_size=0.57)
