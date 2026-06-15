# 04 Endpoint Models

## Common PLSR Baseline

```text
procedure COMMON_PLSR_BASELINE(endpoint_data):
    for each candidate number_of_components:
        for each point-level cross-validation fold:
            X_train, y_train = training fold spectra and labels
            X_valid, y_valid = validation fold spectra and labels

            fit Raman preprocessing on X_train
            transform X_train and X_valid

            fit median imputer on X_train
            fit scaler on X_train
            fit PLSR model on X_train, y_train

            predict y_valid

        aggregate fold predictions
        compute endpoint metrics

    select number_of_components using calibration-set cross-validation
    return selected PLSR route
end procedure
```

## Viability Route

Viability is treated as a bounded percentage response.

```text
procedure VIABILITY_MODEL_SELECTION(endpoint_data):
    candidate_routes = [
        common PLSR baseline,
        Raman selected-feature SVR on identity response,
        Raman selected-feature SVR on bounded-logit response,
        identity response with process-time descriptors,
        bounded-logit process-time-only control,
        bounded-logit Raman plus process-time route
    ]

    for each candidate_route:
        for each point-level cross-validation fold:
            fit all preprocessing, imputation, scaling, and feature selection inside training fold

            if candidate_route uses bounded-logit response:
                clip viability into open interval
                transform viability by logit

            if candidate_route uses Raman features:
                preprocess spectra
                select Raman variables using training fold only

            if candidate_route uses process-time descriptors:
                compute descriptors from current culture time and inoculation time

            fit SVR on training fold
            predict validation fold
            inverse-transform predictions when required

        aggregate validation metrics

    select final route using calibration-set cross-validation
    return locked viability route
end procedure
```

## Cell-Diameter Route

Cell diameter is treated as a narrow-range morphology response rather than as an unconstrained concentration.

```text
procedure DIAMETER_MODEL_SELECTION(endpoint_data):
    candidate_routes = [
        common PLSR baseline,
        absolute-diameter Raman selected-feature SVR,
        logit-range Raman selected-feature SVR,
        absolute-diameter Raman plus process-time SVR,
        logit-range process-time-only control,
        logit-range Raman plus process-time SVR
    ]

    for each candidate_route:
        for each point-level cross-validation fold:
            fit Raman preprocessing and feature selection inside training fold

            if candidate_route uses logit-range response:
                map diameter into predefined open morphology interval
                transform relative position by logit

            if candidate_route uses process-time descriptors:
                compute post-inoculation indicator
                compute pre- and post-inoculation elapsed times
                compute post-inoculation nonlinear time terms
                compute infection-stage window descriptors

            fit selected regression model on training fold
            predict validation fold
            inverse-transform predictions when required

        aggregate validation metrics

    select final route using calibration-set cross-validation
    return locked diameter route
end procedure
```
