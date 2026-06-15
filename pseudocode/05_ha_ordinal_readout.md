# 05 HA Ordinal Readout

Extracellular HA and total HA are modeled as low-resolution ordered log2 dilution-step readouts.

## Tolerance-Aware Point Loss

```text
function POINT_TOLERANCE_LOSS(true_step, predicted_step):
    distance = absolute_value(true_step - predicted_step)
    excess_distance = maximum(distance - one_step_tolerance, 0)
    squared_bias = (true_step - predicted_step)^2

    return excess_distance^2 + small_weight * squared_bias
end function
```

## Cumulative Threshold Probability Model

```text
procedure FIT_ORDINAL_THRESHOLD_MODEL(X_train, y_train, candidate_steps):
    for each threshold in candidate_steps:
        binary_label = indicator(y_train >= threshold)
        fit L2-regularized logistic classifier with balanced class weights

    return list_of_threshold_classifiers
end procedure
```

## Monotone Probability Reconstruction

```text
procedure RECONSTRUCT_STEP_PROBABILITIES(threshold_probabilities):
    enforce cumulative threshold probabilities to be monotone non-increasing
    convert cumulative probabilities into per-step probabilities
    clip invalid probabilities to the feasible range
    normalize per-step probabilities to sum to one
    return step_probability_vector
end procedure
```

## HA Model Selection

```text
procedure HA_ORDINAL_MODEL_SELECTION(endpoint_data):
    candidate_routes = [
        continuous PLSR baseline,
        process-time-only ordinal control,
        Raman ordinal probability baseline,
        Raman ordinal probability with process-time descriptors,
        Raman ordinal probability with HA-aware readout,
        Raman ordinal probability with both process-time descriptors and HA-aware readout
    ]

    for each candidate_route:
        for each point-level cross-validation fold:
            build training and validation rows using offline sampling points

            if candidate_route uses Raman features:
                compute self-reference Raman features when specified
                preprocess Raman features
                fit PLS latent-score model inside training fold
                transform training and validation folds into latent scores

            if candidate_route uses process-time descriptors:
                compute descriptors from culture time and inoculation time
                optionally scale descriptor block by route-specific weight

            if candidate_route is ordinal:
                fit cumulative threshold classifiers on training fold
                predict threshold probabilities for validation spectra
                reconstruct per-step probabilities
            else:
                fit continuous baseline model and predict step values

            if candidate_route uses HA-aware readout:
                pool spectrum-level probabilities within each offline sampling point
                decode final point-level step using tolerance-aware expected loss
            else:
                aggregate spectrum-level outputs to the offline sampling point

        compute cross-validated tolerance-aware loss and secondary metrics

    select final HA route using calibration-set tolerance-aware loss
    return locked HA route
end procedure
```

## Final-Step Decoder

```text
procedure DECODE_FINAL_STEP(point_probability_vector, candidate_steps):
    for each predicted_step in candidate_steps:
        expected_loss[predicted_step] = 0

        for each possible_true_step in candidate_steps:
            probability = point_probability_vector[possible_true_step]
            expected_loss[predicted_step] += probability * POINT_TOLERANCE_LOSS(possible_true_step, predicted_step)

    final_step = candidate step with minimum expected_loss
    return final_step
end procedure
```
