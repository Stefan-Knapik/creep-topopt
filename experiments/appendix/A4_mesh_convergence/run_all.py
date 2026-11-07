"""Appendix B.4 (Fig. B4): fidelity gate, then the mesh ladder, in one process so JAX compilation is shared."""
import argparse

from run_simulation import run_mesh, run_validate

parser = argparse.ArgumentParser()
parser.add_argument("--skip-validate", action="store_true")
args = parser.parse_args()

if not args.skip_validate:
    print("=== fidelity gate: manual driver vs forward_solve ===", flush=True)
    run_validate()

print("\n=== mesh ladder ===", flush=True)
run_mesh()
