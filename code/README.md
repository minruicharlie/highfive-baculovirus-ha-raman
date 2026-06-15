# Submission Code Package

This package contains the clean analysis code for the Raman soft-sensing study
of the High Five cell-recombinant baculovirus HA antigen expression process. It
is provided to document the final analysis workflow and to regenerate article
tables and figures when the non-public input matrices are restored locally.

The public repository does not include the spectral matrices or paired offline
production records used for model fitting, because those inputs contain
commercially sensitive manufacturing information. All paths are resolved
relative to the submission-package root, which contains `code/`, `data/`, and
`results/`; no local absolute path is required.

## Structure

- `configs/config.yaml`: shared data, preprocessing, target, and validation settings.
- `requirements.txt`: Python package versions used for the clean run.
- `src/raman_preprocessing.py`: common Raman preprocessing utilities.
- `src/run_plsr_baselines.py`: common PLSR baselines and Table 1 base outputs.
- `src/run_via_bounded_logit_process_selectsvr.py`: final Via ablation and predictions.
- `src/run_dim_logit_range_process_selectsvr.py`: final Dim ablation and predictions.
- `src/run_ha_ordinal_tolerance_ablation.py`: HA ordinal/tolerance ablation and predictions.
- `src/run_via_coarse_weighting_supplement.py`: Table S5 weighting sensitivity check.
- `src/compile_article_tables.py`: manuscript and supplementary table CSV compiler.
- `src/make_figure1_source_data.py`: source data tables for Figure 1.
- `src/make_figure1.py`: Figure 1 renderer.
- `src/make_final_figures.py`: Figures 1-5 renderer.
- `src/run_all_article_outputs.py`: ordered entry point for the final article pipeline.

## Run

From the submission-package root:

```powershell
python -m pip install -r code\requirements.txt
```

Full reruns require the non-public inputs to be restored under `data/inputs/`
before calling:

```powershell
python code\src\run_all_article_outputs.py
```

The current scripts also require those non-public inputs when rebuilding article
tables and figures from existing model outputs:

```powershell
python code\src\run_all_article_outputs.py --skip-models
```

To print the run order:

```powershell
python code\src\run_all_article_outputs.py --list
```

## Inputs and Outputs

Inputs expected by the final scripts:

- `data/inputs/LabelData_time.csv`
- `data/inputs/UnlabelRamanData.csv`
- `data/inputs/Batch_operation_times.csv`

`LabelData_time.csv` and `UnlabelRamanData.csv` are not included in the public
repository. `Batch_operation_times.csv` is non-spectral process-time metadata.

Expected outputs:

- Model outputs: `results/model_outputs` (public summaries keep candidate
  ablations CV-only; independent-test metrics are reported only for selected
  final models and corresponding PLSR baselines)
- Article tables and supplementary source workbook: `results/tables`
- Article figures: `results/figures`
- Figure source data: `results/figures/source_data`
