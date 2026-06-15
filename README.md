# High Five Baculovirus HA Raman Pseudocode

This repository provides clean pseudocode for endpoint-structure-aware online Raman soft sensing of HA antigen expression in a High Five cell-recombinant baculovirus expression system.

The pseudocode documents the modeling workflow at the methodological level: Raman preprocessing, calibration-set cross-validation, common PLSR baselines, endpoint-specific response handling, process-time descriptors, ordinal HA readout modeling, model locking, and independent-test evaluation.

## Scope

Included:

- pseudocode for the full modeling workflow
- validation and model-locking rules
- endpoint-specific model logic for routine process variables, viability, cell diameter, extracellular HA, and total HA
- leakage-control rules for preprocessing, feature selection, latent-score extraction, and model fitting

Not included:

- raw Raman spectral matrices
- matched offline assay tables
- production records
- article figures, tables, or numerical result files
- executable scripts requiring private manufacturing data

The repository is intended to support methodological transparency without disclosing commercially sensitive spectral records or paired offline measurements.

## Repository Layout

```text
pseudocode/
  01_workflow_overview.md
  02_validation_design.md
  03_raman_preprocessing.md
  04_endpoint_models.md
  05_ha_ordinal_readout.md
  06_reporting_rules.md
```

## Method Summary

The workflow separates model development from independent testing. Candidate models, feature choices, response transformations, process-time descriptors, and hyperparameters are selected only within calibration batches using point-level cross-validation. After route selection, the locked pipeline is refitted on all calibration batches and applied once to the independent test batch.

For routine continuous process variables, a common Raman-PLSR baseline is used. For viability and cell diameter, endpoint-specific response-scale handling and inoculation-aligned process-time descriptors are combined with Raman features. For HA hemagglutination readouts, low-resolution log2 dilution steps are modeled using cumulative ordinal threshold probabilities, point-level probability pooling, and tolerance-aware final-step decoding.

## Data Availability Note

Raw Raman spectra and paired offline measurements are not included in this repository because they contain commercially sensitive manufacturing information. The pseudocode uses generic object names such as `spectra`, `endpoint_labels`, and `process_metadata` to describe the workflow without providing or reconstructing the underlying data.

## Suggested Citation

If this repository is used, please cite the associated manuscript once publication details are available.
