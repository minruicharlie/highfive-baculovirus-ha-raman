# 02 Validation Design

## Batch Split

```text
procedure DEFINE_SPLIT(all_batches):
    independent_test_batch = prespecified held-out batch
    calibration_batches = all_batches excluding independent_test_batch
    return calibration_batches, independent_test_batch
end procedure
```

The independent test batch is not used for model-family comparison, feature selection, hyperparameter tuning, response-scale selection, threshold calibration, or cross-validation.

## Point-Level Cross-Validation

Each offline sampling point may be associated with multiple matched Raman spectra. To avoid splitting spectra from the same offline label across folds, cross-validation uses the offline sampling point as the minimum split unit.

```text
procedure MAKE_POINT_LEVEL_FOLDS(calibration_data, number_of_folds):
    unique_points = unique offline_sampling_point_id in calibration_data
    shuffle unique_points using fixed random seed
    split unique_points into number_of_folds folds

    for each fold:
        validation_points = points assigned to this fold
        training_points = all other points

        training_rows = rows whose offline_sampling_point_id is in training_points
        validation_rows = rows whose offline_sampling_point_id is in validation_points

        yield training_rows, validation_rows
end procedure
```

## Fold-Internal Fitting Rule

All data-dependent transformations are learned within each training fold and then applied to the corresponding validation fold.

```text
for each cross_validation_fold:
    fit imputation parameters on training fold only
    fit Raman preprocessing parameters that require estimation on training fold only
    fit scalers on training fold only
    fit feature selectors on training fold only
    fit PLS latent-score models on training fold only
    fit regression, classification, threshold, or decoder components on training fold only

    apply fitted components to validation fold
    record validation predictions and metrics
```

After route selection, the locked pipeline is refitted using all calibration batches and applied once to the independent test batch.

## Metric Aggregation

```text
procedure EVALUATE_CV(predictions_by_fold):
    concatenate validation predictions across folds

    for continuous endpoints:
        compute RMSE, MAE, R2, MAPE when applicable, and bias

    for HA step readouts:
        compute tolerance-aware loss
        compute RMSE, MAE, bias
        compute rounded within-one-step accuracy

    return endpoint_metrics
end procedure
```
