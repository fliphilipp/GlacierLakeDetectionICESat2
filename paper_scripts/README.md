# `paper_scripts/` — building the archived datasets and models

The scripts that turned the FLUID/SuRRF output of this repository into the two Zenodo data
records accompanying Arndt and Fricker, *Five Years of ICESat-2 Supraglacial Lake Depths Improve
Greenland Meltwater Volume Estimates* (submitted to Annals of Glaciology, 2026), and that trained
and cross-validated the Sentinel-2 depth models described in it.

| record | DOI (concept, always newest version) |
|---|---|
| ICESat-2 lake depths | [10.5281/zenodo.24352679](https://doi.org/10.5281/zenodo.24352679) |
| Training dataset and trained models | [10.5281/zenodo.24352681](https://doi.org/10.5281/zenodo.24352681) |

## A note on paths

These scripts were written against a working tree laid out like the author's, in which a
`zenodo_build/` directory sits beside this repository and holds the staged records. **Their
default arguments reflect that layout and will not resolve in a fresh clone.** Every script takes
explicit paths, so pass them:

```bash
python paper_scripts/train_model.py \
    --features training_features.parquet \
    --config   model_config.json \
    --feature-set rfe_noangle \
    --out      rf_depth_rfe_noangle.joblib
```

`training_features.parquet` and `model_config.json` are in the training-data record above. A
seeded refit under scikit-learn 1.3.2 reproduces the archived models byte for byte.

`zenodo_push.py` is the exception that needs more than arguments: it locates the staged records
through a build directory that it infers from its own location, which was right only in the
author's tree. Set `ZENODO_BUILD_DIR` to point at that directory, or it will tell you to.

## What each script does

**Building the ICESat-2 depth record**

| script | |
|---|---|
| `extract_depth_profiles.py` | along-track depth measurements from the FLUID/SuRRF HDF5 files into one Parquet table |
| `build_summary_table.py` | the per-segment summary table |
| `build_ancillary.py` | per-granule coverage statistics and the drainage basin polygons |
| `bundle_assets.py` | per-segment HDF5 files and plots into one ZIP per melt season |
| `segment_extents.py` | along-track extent and maximum depth of every segment, at three confidence thresholds |
| `document_r1_extras.py` | data-dictionary entries for files that had none |

**Training data and models**

| script | |
|---|---|
| `build_training_record.py` | stage the fused ICESat-2 / Sentinel-2 training dataset |
| `train_model.py` | train a seeded, persistable random forest (62-, 19- or 16-predictor) |
| `cross_validate_model.py` | 10-fold cross-validation grouped by ICESat-2 segment |
| `feature_importance.py` | permutation importance for the 16-predictor model |
| `add_models_to_record.py` | add the trained models to the record, checking feature order against each model's own `feature_names_in_` |
| `load_archived_model.py` | load the original 13.5 GB pickle and re-persist it compactly |

**Figures, validation and housekeeping**

| script | |
|---|---|
| `build_supplement_data.py` | inputs for the supplementary ML figures (S1–S3) |
| `fetch_fig2_chip.py` | the Sentinel-2 chip for the Figure 2 example lake |
| `timefix.py` | recover true UTC Sentinel-2 acquisition times from a naive-local-time bug |
| `sonar_validation_rerun.py` | re-run the sonar validation with corrected acquisition times |
| `validate_16feature_model.py` | repeat both independent validations with the 16-predictor model |
| `check_environment.py` | smoke-test the `icelakes` environment against every code path used here |
| `zenodo_push.py` | create Zenodo drafts, reserve DOIs, upload with checksum verification, publish on demand |

## Caveat

These were written to build one archive once, for one paper. They are recorded here so the
provenance of the archived data is inspectable, not as a maintained pipeline. The processing code
proper is the rest of this repository.
