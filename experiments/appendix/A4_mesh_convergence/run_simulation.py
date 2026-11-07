"""Appendix B.4 (Fig. B4): mesh convergence of the forward creep analysis of the Sec. 4.1-4.2 cantilever (theta = 1).
Writes results/appendix/A4_mesh_convergence/."""
import argparse
import gc
import os

import jax
import pandas as pd

from common import (END_TIME, LX, LY, NUM_TIME_STEPS, RESULTS_DIR,
                    build_cantilever, solid_design)
from creep_driver import run_history, validate_against_forward_solve

# Nodal probes: the corner of the loaded edge and the far top corner.  Neither
# is a singular point -- the traction jump sits at (LX, 0.2*LY) and the clamped
# corners at (0, 0) and (0, LY) -- and both exist on every mesh in the ladder.
PROBES = (("tip", LX, 0.0, 1), ("topright", LX, LY, 1))

# Factor-two ladder in Ny (Nx = 2*Ny), anchored on the paper's 200 x 100:
#   h = 4, 2, 1, 0.5, 0.25 mm  ->  1.25k, 5k, 20k, 80k, 320k cells.
MESH_LADDER = [25, 50, 100, 200, 400]

MESH_CSV = os.path.join(RESULTS_DIR, "mesh_convergence.csv")


def _release(*objs):
    """Drop a problem and its compiled executables.

    Each build_cantilever call creates a fresh Problem, and every jitted closure
    built on it retains its own compiled executable plus the assembly caches.
    Across a mesh ladder that accumulates into gigabytes and will OOM the
    machine before the largest mesh is reached.
    """
    for o in objs:
        del o
    jax.clear_caches()
    gc.collect()


def _resume(path, keys):
    """Rows already computed in `path`, as a set of `keys` tuples, plus the
    existing rows.

    An OOM on the finest mesh kills the sweep mid-ladder.  The sweep therefore
    writes its CSV as it goes and skips configurations already present, so a
    relaunch resumes instead of starting over -- and so a ladder that stops
    early still leaves usable results behind.
    """
    if not os.path.exists(path):
        return set(), []
    df = pd.read_csv(path)
    if not set(keys) <= set(df.columns):
        return set(), []
    done = {tuple(r) for r in df[list(keys)].drop_duplicates().itertuples(index=False)}
    return done, df.to_dict("records")


def _flush(path, rows):
    pd.DataFrame(rows).to_csv(path, index=False)


def _qoi_row(out, **extra):
    """Flatten a run_history record into one CSV row."""
    row = dict(extra)
    row.update({
        "C0": out["compliance"][0], "C_end": out["compliance"][-1], "J": out["J"],
        "umax_end": out["umax"][-1],
        "vm_avg_0": out["vm_avg"][0], "vm_avg_end": out["vm_avg"][-1],
        "vm_max_end": out["vm_max"][-1], "vm_pnorm8_end": out["vm_pnorm8"][-1],
        "eps_eq_avg_end": out["eps_eq_avg"][-1], "eps_eq_max_end": out["eps_eq_max"][-1],
        "finite_all_steps": out["finite_all_steps"], "wall_time_s": out["wall_time_s"],
    })
    for key in out:
        if key.startswith("probe_"):
            row[f"{key}_0"] = out[key][0]
            row[f"{key}_end"] = out[key][-1]
    return row


def run_mesh(ladder=None, methods=("implicit", "explicit"),
             num_time_steps=NUM_TIME_STEPS):
    """The refinement ladder, coarse to fine, for each time-integration scheme.

    Both schemes are run so that the spatial convergence statement is not
    confounded with the choice of integrator; the temporal error is held fixed
    at the paper's N_t = 10 throughout, and its own convergence is the subject
    of study 2.
    """
    ladder = MESH_LADDER if ladder is None else ladder
    done, rows = _resume(MESH_CSV, ("method", "ny"))

    for method in methods:
        for ny in ladder:
            nx = 2 * ny
            if (method, ny) in done:
                print(f"  {method:8s} {nx:>4}x{ny:<4} already done, skipping", flush=True)
                continue
            problem, mesh = build_cantilever(nx, ny, method)
            theta = solid_design(mesh)
            out = run_history(problem, theta, END_TIME, num_time_steps, probes=PROBES)
            rows.append(_qoi_row(
                out, design="solid", method=method, nx=nx, ny=ny, h_mm=LY / ny,
                num_cells=len(mesh.cells),
                num_dofs=problem.fe.num_total_nodes * problem.fe.vec,
                num_time_steps=num_time_steps))
            print(f"  {method:8s} {nx:>4}x{ny:<4} h={LY/ny:6.3f} mm  "
                  f"J = {out['J']:.10g}  finite={out['finite_all_steps']}  "
                  f"({out['wall_time_s']:.1f}s)", flush=True)
            _flush(MESH_CSV, rows)
            _release(problem, mesh, theta, out)

    _flush(MESH_CSV, rows)
    print(f"saved {MESH_CSV}")
    return rows


def run_validate(ny=25, num_time_steps=4):
    """Gate: the manual driver must reproduce Problem.forward_solve."""
    for method in ("explicit", "implicit"):
        problem, mesh = build_cantilever(2 * ny, ny, method)
        theta = solid_design(mesh)
        print(f"  [{method}]")
        validate_against_forward_solve(problem, theta, END_TIME, num_time_steps)
        _release(problem, mesh, theta)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", required=True, choices=["validate", "mesh"])
    parser.add_argument("--quick", action="store_true",
                        help="small parameter grids, for checking the plumbing")
    parser.add_argument("--ladder", nargs="*", type=int,
                        help=f"override the Ny ladder (default {MESH_LADDER})")
    args = parser.parse_args()

    if args.study == "validate":
        run_validate()
    else:
        ladder = args.ladder or ([12, 25] if args.quick else None)
        run_mesh(ladder=ladder)
