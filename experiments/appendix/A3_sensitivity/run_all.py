"""Appendix B.3 (Fig. B3): both sensitivity checks for both schemes, in one process so JAX compilation is shared."""
import argparse

from run_simulation import run_parity, run_fd_step

parser = argparse.ArgumentParser()
parser.add_argument("--skip", nargs="*", default=[], choices=["parity", "fd_step"])
parser.add_argument("--num_samples", type=int, default=None,
                    help="cells in the per-cell parity sweep; default is all of them")
args = parser.parse_args()

for method in ("explicit", "implicit"):
    if "parity" not in args.skip:
        print(f"\n=== parity, {method} ===")
        run_parity(method, num_samples=args.num_samples)
    if "fd_step" not in args.skip:
        print(f"\n=== fd step study, {method} ===")
        run_fd_step(method)
