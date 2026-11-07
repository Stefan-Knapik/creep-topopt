

def _validate_penalization_model(penalization_model):
    if penalization_model is None:
        return "SIMP", 3.0

    if not isinstance(penalization_model, dict) or len(penalization_model) != 1:
        raise ValueError(
            "penalization_model must be a single-entry dict like {'RAMP': 8.0} or {'SIMP': 3.0}."
        )

    model_name, penalty = next(iter(penalization_model.items()))
    model_name = model_name.upper()

    return model_name, penalty


def penalize(density, penalization_model=None):
    model_name, penalty = _validate_penalization_model(penalization_model)

    if model_name == "SIMP":
        return density ** penalty
    
    if model_name == "RAMP":
        return density / (1.0 + penalty * (1.0 - density))

    raise ValueError(f"Unsupported penalization model '{model_name}'. Expected one of: 'SIMP', 'RAMP'.")
