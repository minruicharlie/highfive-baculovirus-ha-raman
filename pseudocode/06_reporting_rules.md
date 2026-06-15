# 06 Reporting Rules

## Model Selection Reporting

```text
for each endpoint:
    report candidate-route performance using calibration-set cross-validation only
    do not use independent-test results to select routes
    do not report independent-test results for non-selected candidate routes
```

## Final Independent-Test Reporting

```text
for each endpoint:
    refit locked final route on all calibration batches
    apply locked route to independent test batch
    compute final endpoint metrics
```

For routine continuous process variables, the locked final route can be the common PLSR baseline. For viability, cell diameter, and HA readouts, the locked route is the endpoint-specific route selected by calibration-set cross-validation.

## HA-Specific Reporting

```text
for extracellular HA and total HA:
    use tolerance-aware loss as the primary model-selection metric
    report RMSE, MAE, bias, and rounded within-one-step accuracy as secondary metrics
    interpret predictions as positive HA step-readout estimates
```

HA model outputs should be interpreted as soft-sensing estimates of HA-related process-state readouts, not as direct absolute quantification of HA protein concentration or baculovirus particle concentration.

## Leakage-Control Checklist

```text
before reporting any cross-validation metric:
    confirm that imputation was fitted inside each training fold
    confirm that scaling was fitted inside each training fold
    confirm that Raman feature selection was fitted inside each training fold
    confirm that PLS latent scores were fitted inside each training fold
    confirm that threshold classifiers were fitted inside each training fold
    confirm that decoder choices were selected using calibration data only

before reporting independent-test metrics:
    confirm that the model route was locked before independent-test evaluation
    confirm that the independent test batch was not used for route comparison
```
