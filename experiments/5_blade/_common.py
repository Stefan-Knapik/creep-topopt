"""Start-up shared by the scripts of experiments/5_blade: environment, x64 JAX, and blade_problem as bp."""
import csv
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("JAX_USE_SIMPLIFIED_JAXPR_CONSTANTS", "1")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
jax.config.update("jax_enable_x64", True)

import logging
import toga
toga.logger.setLevel(logging.WARNING)

sys.path.insert(0, str(Path(__file__).resolve().parent))
import blade_problem as bp  # noqa: E402


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path}")


def release():
    import gc
    jax.clear_caches()
    gc.collect()


def case_kwargs(args):
    """build_blade arguments from the command-line arguments of optimize_graded_blade.add_case_arguments."""
    return dict(
        t_root=bp.T_ROOT if args.t_root is None else args.t_root,
        t_hot=bp.T_HOT if args.t_hot is None else args.t_hot,
        r_hub=bp.R_HUB if args.r_hub is None else args.r_hub,
        elem_size=bp.ELEM_SIZE if args.elem_size is None else args.elem_size,
        mesh_file=bp.FINE_MESH if args.fine else None,
        num_time_steps=bp.NUM_TIME_STEPS if args.steps is None else args.steps,
        expansion=not args.no_expansion, alpha_mode=args.alpha_mode, cte_mismatch_scale=args.cte_mismatch_scale,
        radius_mm=bp.RADIUS_MM if args.radius_mm is None else args.radius_mm,
    )
