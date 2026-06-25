# High Five Baculovirus HA Raman

This repository provides the public, non-sensitive materials supporting a Raman
soft-sensor study of recombinant hemagglutinin production in a High Five
cell-recombinant baculovirus expression process.

**Public-release objective:** all non-sensitive materials supporting the study
are shared here, including clean submission code, method-level pseudocode,
supplementary materials, machine-readable table/figure source data, and
model-output summaries; only the raw Raman spectral matrices and paired offline
manufacturing/reference records are withheld because they contain commercially
sensitive process information.

## Repository Layout

```text
code/
  Clean Python scripts, configuration, and requirements documenting the final
  modeling workflow. Full reruns require the withheld raw input files to be
  restored locally.

data/inputs/
  Non-sensitive process-time metadata retained for descriptor definitions.

pseudocode/
  Method-level pseudocode for preprocessing, validation, model locking,
  endpoint-specific routes, and reporting rules.

source_data/
  Machine-readable non-sensitive table and figure source-data files.

model_output_summaries/
  Aggregated model metrics, ablation summaries, and candidate summaries.

supplementary_material/
  Public supplementary material and Supplementary Figure S1 image.

submission_documents/
  Data availability statement used for the submission package.
```

## Included

- Clean executable analysis scripts without local absolute paths or private
  input matrices.
- Method-level pseudocode for Raman preprocessing, validation, route assignment,
  endpoint-specific modeling, HA ordinal readout handling, and reporting.
- Non-sensitive operation-time metadata used to define inoculation-aligned
  descriptors.
- Machine-readable manuscript and supplementary table source data.
- Non-sensitive aggregate/summary figure source data.
- Aggregated model-output summaries and ablation tables.
- Supplementary material prepared for the submission package.

## Not Included

- Raw Raman spectral matrices.
- Paired offline manufacturing/reference records used for model fitting.
- Point-level prediction/source files that disclose the withheld reference
  records.
- Vendor/raw spectral files or other proprietary manufacturing records.

## Data Availability Note

Raw Raman spectral matrices and paired offline manufacturing/reference records
cannot be publicly shared because they contain commercially sensitive process
information. All other non-sensitive materials supporting the study, including
machine-readable table/figure source data, model-output summaries, clean
submission code, method-level pseudocode, and supplementary materials, are
provided in this repository and in the accompanying supplementary/source-data
files submitted with the manuscript.

## Use Notes

The scripts in `code/` document the final workflow and can be rerun only after
the withheld input files are restored locally under `data/inputs/`. The public
CSV/XLSX files are intended for auditing the reported tables, figures, model
summaries, and supplementary results without disclosing the raw spectral or
paired offline reference records.

## Suggested Citation

If this repository is used, please cite the associated manuscript once
publication details are available.
