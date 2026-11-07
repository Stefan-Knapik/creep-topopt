# Material data

One CSV per alloy (`IN625.csv`, `SS316.csv`), read by `toga.mat.build_material_properties`.
Each column is fitted by the model named in the experiment's `property_configs`. Columns given
only in the first row are temperature independent (fit them with `"temperature_dependent": False`).

## Units

All of toga uses one SI-mm system: mm, N, MPa (= N/mm²), s, mJ, W, °C.
Temperatures are in °C everywhere; the Arrhenius term converts to K internally.

| Column        | Meaning                                    | Unit                 |
|---------------|--------------------------------------------|----------------------|
| `T`           | Temperature of the row                     | °C                   |
| `k`           | Thermal conductivity                       | W/(mm·K)             |
| `alpha`       | Coefficient of thermal expansion           | 1/K                  |
| `E`           | Young's modulus                            | MPa                  |
| `YS`          | Yield strength                             | MPa                  |
| `rho`         | Density                                    | g/mm³ (= 10³ kg/m³)  |
| `cost`        | Relative cost per unit volume              | arbitrary            |
| `logA`        | log10 of the creep pre-exponential factor A| A in MPa⁻ⁿ·s⁻¹       |
| `n`           | Creep stress exponent                      | –                    |
| `Q`           | Creep activation energy                    | J/mol                |
| `criticality` | Material criticality index                 | –                    |

## Creep law

The creep strain rate is

    eps_dot = A * exp(-Q / (R * (T + 273.15))) * sigma_vm**n * (3/2) * s / sigma_vm

with `A = 10**logA`, `R = 8.314 J/(mol·K)`, von Mises stress `sigma_vm` in MPa and deviatoric
stress `s`. The isothermal problems (`CreepExplicit`, `CreepImplicit`) omit the Arrhenius factor,
so there `A` is the full prefactor. Either an `A` or a `logA` column can be tabulated; the other is
derived on load. Fitting `logA` interpolates the exponent linearly between alloys.

`criticality` is not used by the paper's experiments. `ThermalCreepExplicit` accepts it as a
property model so it can be totalled with `compute_total_by_volume(params, property="criticality")`,
like `cost` and `rho`.
