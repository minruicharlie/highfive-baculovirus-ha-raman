# High Five Baculovirus HA Raman

This repository provides public methodological materials and clean submission
code for Raman soft-sensing models developed for a High Five
cell-recombinant baculovirus HA antigen expression process.

## Repository Layout

```text
pseudocode/
  Method-level pseudocode for the modeling workflow.

code/
  Clean Python scripts, configuration, and requirements for regenerating
  article tables and figures when the non-public input files are restored.
```

## Scope

Included:

- Raman preprocessing workflow documentation
- calibration-set cross-validation and model-locking rules
- endpoint-specific modeling logic for routine process variables, viability,
  cell diameter, extracellular HA, and total HA
- leakage-control rules for preprocessing, feature selection, latent-score
  extraction, and model fitting
- clean executable analysis scripts without local paths or private input data

Not included:

- spectral matrices used for model fitting
- paired offline assay records
- production records
- manuscript figures, tables, or numerical result files

## Method Summary

The workflow separates model development from independent testing. Candidate
models, feature choices, response transformations, process-time descriptors,
and hyperparameters are selected only within calibration batches using
point-level cross-validation. After route selection, the locked pipeline is
refitted on all calibration batches and applied once to the independent test
batch.

For routine continuous process variables, a common Raman-PLSR baseline is used.
For viability and cell diameter, endpoint-specific response-scale handling and
inoculation-aligned process-time descriptors are combined with Raman features.
For HA hemagglutination readouts, low-resolution log2 dilution steps are modeled
using cumulative ordinal threshold probabilities, point-level probability
pooling, and tolerance-aware final-step decoding.

## Data Availability Note

Raw spectra and paired offline measurements are not included in this repository
because they contain commercially sensitive manufacturing information. The
`pseudocode/` directory describes the workflow at the methodological level. The
`code/` directory contains the cleaned analysis scripts and requires the
non-public input files to be restored locally before a full rerun.

## Suggested Citation

If this repository is used, please cite the associated manuscript once
publication details are available.
