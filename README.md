# Multi-Hospital EHR Foundation Models Without Data Sharing

Code and manuscript for the paper:

> Elemento O. *Multi-Hospital Electronic Health Record Foundation Models Without Data Sharing: A Comparison of Federated Learning and Inference-Time Ensembling.* 2026.

A GPT-style decoder-only foundation model for electronic health records, evaluated under three multi-hospital training strategies on MIMIC-IV: centralized training (upper bound), federated averaging, and inference-time ensembling. The paper identifies and resolves a previously undescribed failure mode of standard federated averaging on tied-embedding GPT architectures, and shows that an estimated 87% of participating hospitals receive a better model from inference-time ensembling than they would by training alone.

## What's in this repo

```
src/                      # All training and evaluation Python code
configs/                  # Model config (poc.yaml) and example SLURM scripts
docs/manuscript_*.md      # Paper source (markdown)
manuscript/build/         # Node script that compiles the markdown into a Word docx
figures/                  # Final figures used in the paper (PNG/JPG)
data/processed/*.json     # Aggregate evaluation metrics for every reported result
                          # (no patient data — only AUROC/AUPRC/Brier/ECE numbers)
environment.yml           # Conda environment (Python 3.11, PyTorch 2.2+)
LICENSE                   # MIT
CITATION.cff              # Machine-readable citation metadata
```

**Not included** (and never will be): MIMIC-IV raw or processed patient data, tokenized sequences, train/val/test splits, or trained model checkpoints. See [Reproducing the experiments](#reproducing-the-experiments) below.

## Data access

This work uses [MIMIC-IV v2.2](https://physionet.org/content/mimiciv/2.2/), which is freely available but **requires credentialed PhysioNet access** (CITI training + signed data use agreement). To reproduce, you must obtain your own access. We never redistribute MIMIC-IV data, derived tokens, vocabularies, splits, or model weights, since these are derivative works of the credentialed dataset.

After PhysioNet approval, place the unzipped MIMIC-IV directory at `data/MIMIC-IV/` (the path the cohort and tokenizer scripts expect).

## Reproducing the experiments

```bash
# 1. Environment
conda env create -f environment.yml
conda activate ehr-fm

# 2. Build cohort and tokenize (requires data/MIMIC-IV/ in place)
PYTHONPATH=src python src/cohort.py
PYTHONPATH=src python src/tokenize_events.py

# 3. Centralized training (upper bound, untied embeddings, 3 seeds)
sbatch configs/centralized_untied.slurm

# 4. Federated learning (untied FedAvg across 5 simulated sites, 3 seeds)
sbatch configs/sweep_untied_control.slurm

# 5. Per-site training and inference-time ensembling (3 seeds)
sbatch configs/sweep_untied_ensemble.slurm

# 6. FedPer / FedPer + LoRA personalization controls
sbatch configs/sweep_fedper.slurm
sbatch configs/fedper_personalized_eval.slurm

# 7. Calibration (Brier score, ECE, reliability bins)
sbatch configs/calibration.slurm

# 8. Heterogeneity sweep (Dirichlet alpha in {0.5, 1.0, 2.0})
sbatch configs/sweep_splits.slurm

# 9. Build paper figures
PYTHONPATH=src python src/make_paper_figures.py
PYTHONPATH=src python src/make_calibration_figure.py
```

The SLURM scripts in `configs/` are written for a generic Slurm cluster with a `gpu` partition and a `conda` environment named `ehr-fm`; adjust `--partition`, `--account`, and module/environment activation lines for your site.

For local runs without Slurm, the same Python entrypoints can be invoked directly (each `.slurm` file shows the exact `python …` invocations).

## Building the manuscript

The Word manuscript is generated from `docs/manuscript_federated_ehr.md` with figures embedded inline:

```bash
npm install --global docx
NODE_PATH=$(npm root -g) node manuscript/build/generate_manuscript.js
# → manuscript/build/manuscript_draft.docx
```

## Repository structure (training pipeline)

| File | Purpose |
|---|---|
| `src/cohort.py` | Selects MIMIC-IV patients with ≥2 admissions and produces train/val/test splits. |
| `src/tokenize_events.py` | Builds token vocabulary and tokenizes per-patient sequences (DX/PX/MED/LAB tokens). |
| `src/model.py` | Decoder-only transformer with two-level attention, RoPE indexed by inter-visit days, repeat-token decay loss. Supports tied and untied embeddings. |
| `src/train.py` | Centralized training entrypoint. |
| `src/train_per_site.py` | Independent per-site training (no inter-site communication). |
| `src/train_federated.py` | FedAvg / FedProx / FedPer / FedPer + LoRA federated training. |
| `src/evaluate.py` | Whole-vocabulary AUPRC evaluation. |
| `src/evaluate_conditions.py` | Per-condition AUROC for the five acute conditions. |
| `src/evaluate_ensemble.py` | Inference-time ensembling across per-site models. |
| `src/evaluate_fedper_personalized.py` | Personalized eval for FedPer (uses each site's own output projection). |
| `src/compute_calibration.py` | Brier score, ECE, and reliability bins per condition. |
| `src/baseline_linear.py` | Linear bag-of-diagnoses baseline. |
| `src/make_paper_figures.py` | Figures 2 and 3. |
| `src/make_calibration_figure.py` | Figure 4. |
| `src/generate_figure1.py` | Figure 1 schematic (uses the Gemini image API). |

## Aggregate result files (`data/processed/`)

Every JSON file under `data/processed/` is a small aggregate-metrics record (per-condition AUROC, overall and new-onset AUPRC, Brier, ECE, reliability-bin counts, etc.) keyed by the seed and configuration that produced it. These contain no patient-level data and are committed so the figures can be regenerated without re-training. The naming scheme is `eval_<config>_<alpha>_<seed>.json`, `eval_cond_<config>_<alpha>_<seed>.json` (per-condition), and `calib_<config>_<seed>.json`.

## Citation

```bibtex
@article{elemento2026federatedehr,
  title   = {Multi-Hospital Electronic Health Record Foundation Models Without Data Sharing:
             A Comparison of Federated Learning and Inference-Time Ensembling},
  author  = {Elemento, Olivier},
  year    = {2026}
}
```

See `CITATION.cff` for machine-readable metadata.

## License

Code: MIT (see `LICENSE`). The manuscript text and figures in `docs/` and `figures/` are the author's; reuse for academic or non-commercial purposes is permitted with citation.
