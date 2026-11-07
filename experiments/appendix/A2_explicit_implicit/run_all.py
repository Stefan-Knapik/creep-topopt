"""Appendix B.2 (Fig. B2): full time-step refinement sweep in one process, so JAX compilation is shared."""
import os

import pandas as pd

from run_simulation import run, RESULTS_DIR, NUM_STEPS, N_REF

rows = []
for case in ("tension", "mixed"):
    for method in ("explicit", "implicit"):
        print(f"=== {case}, {method} ===")
        for n in NUM_STEPS:
            rows.append(run(case, method, n))

# Temporally converged implicit reference for the biaxial case, which has no
# closed-form solution.
print(f"=== reference runs, N = {N_REF} ===")
for case in ("tension", "mixed"):
    rows.append(run(case, "implicit", N_REF))

df = pd.DataFrame(rows)

# Error against the converged reference, per case (the closed form is only
# available for the tension case and is filled in by run()).
for case in df["case"].unique():
    m_ref = (df["case"] == case) & (df["method"] == "implicit") & (df["num_time_steps"] == N_REF)
    ref = float(df.loc[m_ref, "sigma_xx_end"].iloc[0])
    m = df["case"] == case
    df.loc[m, "sigma_xx_ref"] = ref
    df.loc[m, "err_vs_ref"] = (df.loc[m, "sigma_xx_end"] - ref).abs() / abs(ref)

out = os.path.join(RESULTS_DIR, "scheme_convergence.csv")
df.to_csv(out, index=False)
print(f"saved {out}")
print(df.to_string(index=False))
