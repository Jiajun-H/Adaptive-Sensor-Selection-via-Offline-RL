# Adaptive Sensor Selection for Resource-Constrained Compartmental Fire Early Warning via Offline Reinforcement Learning

This repository contains the Python source code and environment specifications for the project. The real DOCX document, experimental Excel data, model checkpoints, and generated figures are not included.

## Contents

- Python scripts for data conversion, offline dataset construction, predictor and policy training, evaluation, baselines, inference, and visualization.
- `environment.yml` and `requirements.txt` for environment setup.

## Input data

The experimental workbooks are required to run the training and evaluation pipelines and must be supplied separately. By default, the multi-experiment loader reads `converted_excel/*.xlsx`; the legacy single-workbook path reads `data.xlsx`. These files are excluded from Git by `.gitignore`.

If you have the original Origin project files, the optional `convert_opju_to_excel.py` script can export them to `converted_excel/`. That conversion requires Origin or OriginPro with Windows COM automation and `pywin32`.

## Set up

Create and activate the environment from the repository root:

```powershell
conda env create -f environment.yml
conda activate rl
```

Alternatively, install packages from `requirements.txt` in your active Python environment, then install a compatible PyTorch build as described there.

## Run

Run scripts from the repository root. For example:

```powershell
python run_full_experiments.py --smoke --smoke_max_files 3
python run_full_experiments.py
```

Use `python run_repeated_experiments.py --help` to see fold and seed options. Training and evaluation create `checkpoints/` and `figures/` beside the source files; inference scripts need checkpoints produced by training.

## Visualization scripts

- `visualize.py` draws figures from experiment results.
- `visualize_paper.py` reads saved experiment JSON files for manuscript figures.
- `visualize_ch5.py` is the Chapter 5 command-line entry point; data loading and label helpers are in `visualize_ch5_data.py`, and plotting functions and styles are in `visualize_ch5_plots.py`.
- `compose_sensor_selection_figures.py` combines previously generated sensor heatmaps; `export_sensor_selection_masks.py` creates those summaries from checkpoints.

`visualize_ch5_data.py` is an explanatory-figure loader: it interpolates within each workbook and can infer time direction from signal trends. The training pipeline uses `dataset.py` instead, including training-split-only imputation and normalization. Use experiment JSON or CSV outputs for model metrics.

## Path handling

Core paths in `config.py` are based on the source file location. Run commands from the repository root because some optional command-line paths are relative to the current working directory.