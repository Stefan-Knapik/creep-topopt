"""Appendix B.1 (Fig. B1): all TOGA case/method runs for the Abaqus comparison, in one process."""
from run_simulation import run

# N = 32 matches the 32 fixed increments of the Abaqus runs. N = 1024 is a temporally
# converged reference: the remaining deviation from the closed form is time-discretisation error.
for case in ("tension", "mixed"):
    for method in ("explicit", "implicit"):
        for num_time_steps in (32, 1024):
            print(f"=== {case}, {method}, {num_time_steps} steps ===")
            run(case, method, num_time_steps)
