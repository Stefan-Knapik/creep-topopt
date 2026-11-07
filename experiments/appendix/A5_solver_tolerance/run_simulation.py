"""Appendix B.5 (Fig. B5): sensitivity of J and dJ/dtheta to the global Newton and local return-map tolerances.
Writes results/appendix/A5_solver_tolerance/."""
import argparse
import gc
import os
import time

import numpy as onp
import pandas as pd

from common import (DATA_DIR, END_TIME, GREY_THETA, NUM_TIME_STEPS, PAPER_NX,
                    PAPER_NY, RESULTS_DIR, build_cantilever, solid_design,
                    uniform_design)

import jax
import optimistix as optx

SOLVER_OPTIONS = {"pardiso_solver": {}}

# --- the production settings -----------------------------------------------
DEFAULT_GLOBAL_TOL = 1e-6
DEFAULT_GLOBAL_REL_TOL = 1e-8
DEFAULT_GLOBAL_MAX_ITERS = 50
DEFAULT_LOCAL_ATOL = 1e-10
DEFAULT_LOCAL_RTOL = 1e-4
DEFAULT_LOCAL_MAX_STEPS = 100

# --- the two ladders -------------------------------------------------------
# `gscale` multiplies BOTH global tolerances, so gscale = 1 is the production
# setting and gscale = 1e-2 is a hundred times tighter.  1e-2 is the tightest
# usable point: the global residual bottoms out on a round-off floor of a few
# 1e-10 (PARDISO on a 442 MPa initial residual), and asking for less than that
# would simply burn all 50 Newton iterations at every step.
GLOBAL_SCALES = [1e6, 1e4, 1e2, 1e0, 1e-2]
LOCAL_ATOLS = [1e-6, 1e-8, 1e-10, 1e-12, 1e-14]

REF_GSCALE = 1e-2
REF_ATOL = 1e-14

# Two design fields.  `solid` is the solid domain used by A4 and A5, so
# the statement is about the forward analysis alone.  `grey` is the stress test:
# a uniform intermediate density, where the RAMP penalisation of the creep
# prefactor makes the material creep ~3000x faster and the problem genuinely
# nonlinear, which is the regime the optimiser passes through.
DESIGNS = {"solid": lambda mesh: solid_design(mesh),
           "grey": lambda mesh: uniform_design(mesh, GREY_THETA)}


def results_csv(design):
    return os.path.join(RESULTS_DIR, f"tolerance_study_{design}.csv")


def grad_dir(design):
    d = os.path.join(DATA_DIR, "grads", design)
    os.makedirs(d, exist_ok=True)
    return d


def make_configs(rtol_check=False):
    """The ten runs, reference first so its gradient is on disk for the rest.

    Keyed by (gscale, atol, rtol); the tightest point of the global ladder and
    the tightest point of the local ladder are the same run and appear once.
    """
    cfgs = [{"name": "reference", "sweep": "reference", "gscale": REF_GSCALE,
             "atol": REF_ATOL, "rtol": DEFAULT_LOCAL_RTOL}]
    seen = {(REF_GSCALE, REF_ATOL, DEFAULT_LOCAL_RTOL)}

    def add(sweep, gscale, atol, rtol=DEFAULT_LOCAL_RTOL):
        key = (gscale, atol, rtol)
        if key in seen:
            return
        seen.add(key)
        cfgs.append({"name": f"{sweep}_{len(cfgs)}", "sweep": sweep,
                     "gscale": gscale, "atol": atol, "rtol": rtol})

    # The production point: both solvers at their defaults simultaneously.
    add("production", 1.0, DEFAULT_LOCAL_ATOL)
    # Global ladder, local solver held at the reference setting.
    for s in GLOBAL_SCALES:
        add("global", s, REF_ATOL)
    # Local ladder, global solver held at the reference setting.
    for a in LOCAL_ATOLS:
        add("local", REF_GSCALE, a)
    if rtol_check:
        # rtol is not a ladder: it is checked once, loose and once tight, at the
        # default atol, to show the local test is not rtol-controlled here.
        for r in (1e-2, 1e-8):
            add("local_rtol", REF_GSCALE, DEFAULT_LOCAL_ATOL, r)
    return cfgs


def evaluate(cfg, nx, ny, num_time_steps, end_time, want_grad, design="solid"):
    """One configuration: J and, optionally, dJ/dtheta.

    A fresh Problem per configuration.  `creep_solver` is read while the step
    function is traced, so a Problem reused across the sweep would silently keep
    whichever local tolerance it was first traced with.
    """
    problem, mesh = build_cantilever(nx, ny)
    problem.creep_solver = optx.Newton(rtol=cfg["rtol"], atol=cfg["atol"])
    problem.creep_solver_max_steps = DEFAULT_LOCAL_MAX_STEPS
    theta = DESIGNS[design](mesh)

    gopts = dict(SOLVER_OPTIONS,
                 tol=DEFAULT_GLOBAL_TOL * cfg["gscale"],
                 rel_tol=DEFAULT_GLOBAL_REL_TOL * cfg["gscale"],
                 max_iters=DEFAULT_GLOBAL_MAX_ITERS)

    def objective(th):
        # The paper's forward pass, verbatim: same entry point, same options on
        # both the forward solve and the adjoint (vjp) solve.
        full_sol, _ = problem.forward_solve(
            theta=th, end_time=end_time, num_time_steps=num_time_steps,
            fwd_solver_options=gopts, vjp_solver_options=gopts)
        u_full = full_sol[0]
        return (problem.compute_compliance(u_full[-1])
                - problem.compute_compliance(u_full[0]))

    t0 = time.perf_counter()
    if want_grad:
        J, dJ = jax.value_and_grad(objective)(theta)
        dJ = onp.asarray(dJ).reshape(-1)
    else:
        J, dJ = objective(theta), None
    J = float(J)
    wall = time.perf_counter() - t0

    finite = onp.isfinite(J) and (dJ is None or bool(onp.all(onp.isfinite(dJ))))
    del problem, mesh, theta
    jax.clear_caches()
    gc.collect()
    return J, dJ, wall, finite


def run(nx=PAPER_NX, ny=PAPER_NY, num_time_steps=NUM_TIME_STEPS,
        end_time=END_TIME, want_grad=True, rtol_check=False, fresh=False,
        design="solid", tag=None):
    # `tag` names the output files; --quick passes "<design>_quick" so a
    # plumbing check can never overwrite a committed production result.
    tag = tag or design
    csv_path, gdir = results_csv(tag), grad_dir(tag)
    if fresh and os.path.exists(csv_path):
        os.remove(csv_path)

    done = {}
    if os.path.exists(csv_path):
        prev = pd.read_csv(csv_path)
        done = {r["name"]: r for r in prev.to_dict("records")}

    rows = []
    for cfg in make_configs(rtol_check):
        gpath = os.path.join(gdir, f"{cfg['name']}.npy")
        if cfg["name"] in done and (not want_grad or os.path.exists(gpath)):
            rows.append(done[cfg["name"]])
            print(f"  {cfg['name']:<14s} cached", flush=True)
            continue

        J, dJ, wall, finite = evaluate(cfg, nx, ny, num_time_steps, end_time,
                                       want_grad, design)
        if dJ is not None:
            onp.save(gpath, dJ)
        rows.append({"name": cfg["name"], "sweep": cfg["sweep"],
                     "design": design, "gscale": cfg["gscale"],
                     "global_tol": DEFAULT_GLOBAL_TOL * cfg["gscale"],
                     "global_rel_tol": DEFAULT_GLOBAL_REL_TOL * cfg["gscale"],
                     "local_atol": cfg["atol"], "local_rtol": cfg["rtol"],
                     "J": J, "finite": finite, "wall_time_s": wall})
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        print(f"  {cfg['name']:<14s} gscale={cfg['gscale']:8.1e} "
              f"atol={cfg['atol']:8.1e} rtol={cfg['rtol']:8.1e}  "
              f"J = {J:.12g}  ({wall:.1f}s)", flush=True)

    return derive(rows, want_grad, tag)


def derive(rows, want_grad=True, tag="solid"):
    """Attach the deviation of every configuration from the reference."""
    csv_path, gdir = results_csv(tag), grad_dir(tag)
    df = pd.DataFrame(rows)
    ref = df[df["name"] == "reference"].iloc[0]
    df["rel_dJ"] = (df["J"] - float(ref["J"])).abs() / abs(float(ref["J"]))

    if want_grad:
        gref = onp.load(os.path.join(gdir, "reference.npy"))
        nref = onp.linalg.norm(gref)
        errs = []
        for name in df["name"]:
            p = os.path.join(gdir, f"{name}.npy")
            if not os.path.exists(p):
                errs.append(onp.nan)
                continue
            g = onp.load(p)
            errs.append(float(onp.linalg.norm(g - gref) / nref))
        df["rel_dgrad"] = errs

    df.to_csv(csv_path, index=False)
    print(f"saved {csv_path}")

    prod = df[df["sweep"] == "production"]
    if not prod.empty:
        p = prod.iloc[0]
        print(f"\n  at the production defaults: |dJ|/J = {p['rel_dJ']:.3e}"
              + (f", ||d(dJ/dtheta)|| / ||dJ/dtheta|| = {p['rel_dgrad']:.3e}"
                 if want_grad else ""))
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true",
                        help="40 x 20 mesh, 2 steps: checks the plumbing")
    parser.add_argument("--no-grad", action="store_true",
                        help="objective only, skip the adjoint solve")
    parser.add_argument("--rtol-check", action="store_true",
                        help="add the two local-rtol spot checks")
    parser.add_argument("--fresh", action="store_true",
                        help="discard the existing CSV and rerun everything")
    parser.add_argument("--design", default=["solid"], nargs="+",
                        choices=list(DESIGNS),
                        help="design field(s) to run; default solid")
    args = parser.parse_args()

    for design in args.design:
        print(f"=== design: {design} ===", flush=True)
        if args.quick:
            run(nx=40, ny=20, num_time_steps=2, want_grad=not args.no_grad,
                rtol_check=args.rtol_check, fresh=True, design=design,
                tag=f"{design}_quick")
        else:
            run(want_grad=not args.no_grad, rtol_check=args.rtol_check,
                fresh=args.fresh, design=design)
