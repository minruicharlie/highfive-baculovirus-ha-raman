# 01 Workflow Overview

This pseudocode describes the complete modeling workflow without exposing raw spectra, paired offline measurements, or production records.

## Conceptual Inputs

```text
spectra
  Online Raman spectra indexed by batch, acquisition time, and spectrum identifier.

endpoint_labels
  Offline endpoint measurements indexed by batch, sampling time, endpoint, and sampling-point identifier.

process_metadata
  Non-spectral process metadata required to compute culture time and inoculation-aligned descriptors.

endpoint_definitions
  Endpoint names, response type, evaluation metrics, and model family.
```

## Top-Level Procedure

```text
procedure RUN_ARTICLE_WORKFLOW:
    load spectra, endpoint_labels, and process_metadata

    define the independent_test_batch before any model selection
    calibration_batches = all batches except independent_test_batch

    for each endpoint:
        endpoint_data = match Raman spectra to available offline labels
        endpoint_data = remove rows without valid endpoint labels

        if endpoint is a routine continuous process variable:
            run COMMON_PLSR_BASELINE(endpoint_data)

        if endpoint is viability:
            run VIABILITY_MODEL_SELECTION(endpoint_data)

        if endpoint is cell diameter:
            run DIAMETER_MODEL_SELECTION(endpoint_data)

        if endpoint is extracellular HA or total HA:
            run HA_ORDINAL_MODEL_SELECTION(endpoint_data)

    lock selected endpoint-specific routes using calibration batches only

    for each endpoint:
        refit locked route on all calibration batches
        evaluate once on independent_test_batch

    compile manuscript-level summaries from locked models and predefined metrics
end procedure
```

## Development Principle

```text
All route comparisons, feature selection, response transformations, hyperparameter choices,
threshold calibration, and model selection are performed only on calibration batches.

The independent test batch is used only after the final route has been locked.
```
