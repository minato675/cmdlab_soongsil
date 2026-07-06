# GATGNN Training, Evaluation, and Prediction Manual

[한국어](README.md) | [English](README_EN.md)

## Table of contents

- [Main directories](#1-main-directories)
- [Runtime environment](#2-runtime-environment)
- [Supported properties and filenames](#3-supported-properties-and-filenames)
- [Dataset source conventions](#4-dataset-source-conventions)
- [Prepare training data](#5-prepare-training-data)
- [Train a model](#6-train-a-model)
- [Evaluate a model](#7-evaluate-a-model)
- [Predict properties for new CIF files](#8-predict-properties-for-new-cif-files)
- [Calculate CIF volumes](#9-calculate-cif-volumes)
- [Important command options](#10-important-command-options)
- [Recommended end-to-end workflow](#11-recommended-end-to-end-workflow)
- [Troubleshooting](#12-troubleshooting)
- [Checklist](#13-checklist)
- [Usage manual](#usage-manual)

This directory trains GATGNN models from CIF crystal structures and property CSV files, evaluates trained models, and predicts properties for new CIF structures.

```text
Training CIF files + DATA/properties-reference/<property>.csv
  → automatically generate DATA/<dataset>/id_prop.csv
  → train.py
  → TRAINED/<property>.pt
  → evaluate.py → RESULTS/<property>_results.csv
  → predict.py  → PREDICTIONS/pred_<property>_<source>_<target>.csv
```

## 1. Main directories

| Path | Purpose | Tracked by Git |
| --- | --- | --- |
| `DATA/train&evaluate/<data_src>/` | Training and evaluation CIF sources | Directory only |
| `DATA/prediction/<data_src>/` | CIF composition sources for prediction | Directory only |
| `DATA/properties-reference/` | ID-value CSV for each property | Yes |
| `TRAINED/` | Trained checkpoints | Model storage |
| `RESULTS/` | Evaluation CSV output | Directory only |
| `PREDICTIONS/` | Prediction CSV output | Directory only |

Only the three directories above belong at the DATA root. Property CSV files are committed, while each data-source directory tracks only `.gitkeep`; actual CIF data remains local.

## 2. Runtime environment

All scripts use relative paths. Run every command from the repository's `GATGNN/` directory.

```powershell
cd GATGNN
```

Required Python packages include:

- PyTorch
- PyTorch Geometric
- NumPy, pandas, and scikit-learn
- pymatgen
- tabulate

PyTorch and PyTorch Geometric packages must be compatible with each other and with the CPU/CUDA runtime. Prefer an existing environment that has already been validated for this project.

Check the main imports:

```powershell
python -c "import torch, torch_geometric, pymatgen, pandas, sklearn; print('GATGNN ready')"
```

Check GPU availability:

```powershell
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

The scripts select `cuda:0` when CUDA is available and otherwise use the CPU.

## 3. Supported properties and filenames

The `--property` value determines which CSV is read from `DATA/properties-reference` and which checkpoint filename is used.

| `--property` | Reference CSV | Model file |
| --- | --- | --- |
| `bulk-modulus` | `bulkmodulus.csv` | `TRAINED/bulk-modulus.pt` |
| `shear-modulus` | `shearmodulus.csv` | `TRAINED/shear-modulus.pt` |
| `poisson-ratio` | `poissonratio.csv` | `TRAINED/poisson-ratio.pt` |
| `density` | `density.csv` | `TRAINED/density.pt` |
| `thermal-conductivity` | `thermalconductivity.csv` | `TRAINED/thermal-conductivity.pt` |
| `new-property` | `newproperty.csv` | `TRAINED/new-property.pt` |
| `new_bulk-modulus` | `newbulkmodulus.csv` | `TRAINED/new_bulk-modulus.pt` |
| `new_Youngs-modulus` | `newyoungsmodulus.csv` | `TRAINED/new_Youngs-modulus.pt` |

The scripts also support `absolute-energy`, `band-gap`, `fermi-energy`, and `formation-energy`.

Each reference CSV must contain exactly two columns without a header:

```csv
6794,7.81
6904,8.03
6905,7.95
```

The first column is the CIF ID and the second is a numeric property value. Empty values and the string `None` are removed.

## 4. Dataset source conventions

### CMD

CMD is the primary format for the current custom data.

- CIF directory: `DATA/train&evaluate/CIF-DATA_CMD/`
- CIF filename: `<numeric ID>.cif`
- ID in the reference CSV: `<ID>` without the prefix
- Command option: `--data_src CMD`

For example, CSV ID `6794` must correspond to `6794.cif`. Only CIF IDs that match the property CSV in the selected folder are used automatically.

### NEW

- CIF directory: `DATA/train&evaluate/CIF-DATA_NEW/`
- Command option: `--data_src NEW`
- Custom properties normally use `new-property`

### CGCNN and MEGNET

- CIF directory: `DATA/train&evaluate/CIF-DATA/`
- Command options: `--data_src CGCNN` or `--data_src MEGNET`
- The original reference and filtering files are required

Each training data source requires `atom_init.json`. If missing, the code copies it from an existing default source under `DATA/train&evaluate`.

## 5. Prepare training data

### Interactive execution

Running `train.py`, `evaluate.py`, or `predict.py` without options opens a step-by-step configuration menu.

```powershell
python train.py
python evaluate.py
python predict.py
```

1. Select a property discovered from `DATA/properties-reference/*.csv`.
2. Training and evaluation list sources under `DATA/train&evaluate`; prediction lists sources under `DATA/prediction`.
3. Enter another value for each remaining option or press Enter to accept its displayed default.

To add a property, place `<new-property>.csv` in `DATA/properties-reference/` using the headerless `ID,value` format, then run a script again. The CSV appears in the menu automatically and is treated as a regression property by default. Traditional command-line arguments remain supported.

```powershell
python train.py --property new-property-name --data_src CIF-DATA_CMD
```

### Data preparation example

For a CMD density model:

1. Prepare `DATA/properties-reference/density.csv`.
2. Place matching `<numeric ID>.cif` files in the selected `DATA/train&evaluate/<data_src>/`.
3. Confirm that every CSV ID has a corresponding CIF file.
4. Close `id_prop.csv` if it is open in another program.

When `train.py`, `evaluate.py`, or `predict.py` runs, `file_setter.py` generates or overwrites:

```text
DATA/train&evaluate/CIF-DATA_CMD/id_prop.csv
```

Keeping this file open in Excel may cause a `PermissionError`.

## 6. Train a model

Basic CMD training example:

```powershell
python train.py --property density --data_src CIF-DATA_CMD
```

Examples for properties currently used in this repository:

```powershell
python train.py --property thermal-conductivity --data_src CIF-DATA_CMD
python train.py --property poisson-ratio --data_src CIF-DATA_CMD
python train.py --property new_bulk-modulus --data_src CIF-DATA_CMD
python train.py --property new_Youngs-modulus --data_src CIF-DATA_CMD
```

Custom NEW dataset example:

```powershell
python train.py --property new-property --data_src NEW --train_size 0.8
```

Current training defaults are fixed in the code:

- Maximum epochs: 200
- Batch size: 256
- Learning rate: `5e-3`
- Optimizer: AdamW
- Early-stopping patience: 150
- Random seed: 456
- Default training ratio: 0.8, unless a predefined dataset size takes precedence

The best checkpoint is first stored at `TRAINED/crystal-checkpoint.pt`. At the end of training it is copied to:

```text
TRAINED/<property>.pt
```

Training the same property again may overwrite the existing model. Back up important checkpoints first.

## 7. Evaluate a model

Evaluation must use the same architecture options that were used for training.

```powershell
python evaluate.py --property density --data_src CIF-DATA_CMD
```

The result is written to:

```text
RESULTS/density_results.csv
```

The CSV contains the material ID, measured value, predicted value, number of atoms, and dataset index.

If training used custom attention or layer settings, repeat them during evaluation:

```powershell
python evaluate.py --property density --data_src CIF-DATA_CMD --num_layers 5 --global_attention cluster --cluster_option fixed
```

Using a different architecture causes checkpoint size-mismatch errors.

## 8. Predict properties for new CIF files

`predict.py --to_predict` accepts a directory, one CIF file, or one material ID.

### Predict a directory

```powershell
python predict.py --property density --data_src prediction-directory
```

All `.cif` and `.cif.gz` files directly inside the directory are sorted and processed.

### Predict one CIF file

```powershell
python predict.py --property density --data_src prediction-directory --to_predict DATA\prediction\prediction-directory\6794.cif
```

### Predict one ID from the default directory

```powershell
python predict.py --property density --data_src prediction-directory --to_predict 6794
```

The default ID form looks for `DATA/prediction/prediction-directory/6794.cif` or `.cif.gz`.

Example output path:

```text
PREDICTIONS/pred_density_CMD_prediction-directory.csv
```

Output format:

```csv
material_id,prediction
6794,7.812345
```

Prediction must use the same `--property`, `--data_src`, and architecture options as training.

## 9. Calculate CIF volumes

`volume_predict.py` does not use GATGNN. It reads each CIF with pymatgen and calculates its unit-cell volume.

```powershell
python volume_predict.py --to_predict DATA\prediction\prediction-directory
```

It also accepts one file or one ID:

```powershell
python volume_predict.py --to_predict DATA\prediction\prediction-directory\6794.cif
python volume_predict.py --to_predict 6794
```

Default output:

```text
PREDICTIONS/volume_prediction-directory.csv
```

Use `--out_dir` to select another output directory.

## 10. Important command options

| Option | Default | Description |
| --- | --- | --- |
| `--property` | `bulk-modulus` | Property to train, evaluate, or predict |
| `--data_src` | `CGCNN` | `CGCNN`, `MEGNET`, `NEW`, or `CMD` |
| `--to_predict` | selected `DATA/prediction/<data_src>` | Prediction ID, CIF file, or directory |
| `--num_layers` | 3 | Number of AGAT layers |
| `--num_neurons` | 64 | Neurons per layer |
| `--num_heads` | 4 | Number of attention heads |
| `--global_attention` | `composition` | `composition` or `cluster` |
| `--cluster_option` | `fixed` | `fixed`, `random`, or `learnable` |
| `--train_size` | 0.8 | Training data ratio |

`--use_hidden_layers` and `--concat_comp` currently use `type=bool` in `argparse`. A string such as `False` may still be interpreted as true. Inspect or change the code when a non-default boolean value is required.

## 11. Recommended end-to-end workflow

One complete CMD density cycle:

```powershell
cd GATGNN

# 1. Train
python train.py --property density --data_src CIF-DATA_CMD

# 2. Evaluate
python evaluate.py --property density --data_src CIF-DATA_CMD

# 3. Predict a directory of new CIF files
python predict.py --property density --data_src prediction-directory

# 4. Optionally calculate volumes
python volume_predict.py --to_predict DATA\prediction\prediction-directory
```

## 12. Troubleshooting

### `Missing atom_init.json`

Provide `atom_init.json` in the selected training source or in a default source under `DATA/train&evaluate`.

### CIF file not found

- Every training CIF must follow `<numeric ID>.cif`.
- ID-based prediction looks in the selected `DATA/prediction/<data_src>/<ID>.cif`.
- Confirm that the command is running from the repository's `GATGNN/` directory.

### `Permission denied ... id_prop.csv`

Close the relevant `id_prop.csv` in Excel or another editor and retry.

### Missing model checkpoint

Confirm that `TRAINED/<property>.pt` exists. Case, underscores, and hyphens in `--property` must match the filename.

### Checkpoint size mismatch

Make sure training, evaluation, and prediction use identical layer, neuron, head, and attention options.

### CUDA out of memory

Reduce `batch_size` in `train.py`. It is currently fixed in the source rather than exposed as a command option.

### Invalid CSV values

The reference CSV must have two columns and no header. Check for extra commas and ensure the property column is numeric.

## 13. Checklist

- [ ] Run commands from the repository's `GATGNN/` directory
- [ ] Confirm the property name and reference CSV filename
- [ ] Match every reference ID with a CIF filename
- [ ] Confirm that `atom_init.json` exists
- [ ] Back up an existing `TRAINED/<property>.pt` if needed
- [ ] Use identical model options for training, evaluation, and prediction
- [ ] Review evaluation output in `RESULTS/`
- [ ] Review prediction output in `PREDICTIONS/`

## Usage manual

### GATGNN terminology

- **property**: The target learned or predicted by the model, such as density or thermal conductivity.
- **data source**: A directory grouping CIF files for training or prediction.
- **CIF ID**: The numeric identifier used as a training filename; `1328.cif` has ID `1328`.
- **reference CSV**: A file containing the known target value for each CIF ID.
- **checkpoint**: Learned model parameters stored as `TRAINED/<property>.pt`.
- **epoch**: One complete pass over the training dataset.

### Step 1: Install the environment

Run once from the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For each new PowerShell session, activate the environment again from the repository root, then enter GATGNN:

```powershell
cd GATGNN
```

### Step 2: Predict with an existing model

This is the simplest first-time workflow.

1. Confirm that the property checkpoint exists in `TRAINED/`; density requires `TRAINED/density.pt`.
2. Create a named batch directory under `DATA/prediction/`.
3. Put `.cif` or `.cif.gz` files in that directory.

```text
DATA/prediction/my-samples/
├─ sample-a.cif
└─ sample-b.cif
```

4. Start interactive prediction:

```powershell
python predict.py
```

5. Select the property in the first menu.
6. Select `my-samples` in the second menu.
7. Press Enter for architecture defaults if the checkpoint was trained with defaults.
8. Open `PREDICTIONS/pred_<property>_<data_src>_<target>.csv`.

The prediction CSV contains:

| Column | Meaning |
| --- | --- |
| `material_id` | CIF filename without its extension |
| `prediction` | Model-predicted property value |

### Step 3: Prepare training data

This is needed only for training a new model.

1. Create a source directory under `DATA/train&evaluate/`.
2. Name every training structure `<numeric ID>.cif`. Names such as `cmd-1328.cif` and `sample.cif` are rejected.
3. Create `DATA/properties-reference/<property>.csv`.
4. Write `numeric ID,property value` without a header.

```text
DATA/train&evaluate/my-training-data/
├─ 1328.cif
├─ 1329.cif
└─ atom_init.json
```

```csv
1328,107.68
1329,137.49
```

A file named `my-property.csv` appears as `my-property` in the property menu. The code intersects the selected directory's actual CIF IDs with CSV IDs and automatically generates `id_prop.csv` from matching rows only.

If `atom_init.json` is missing, the code copies it from an existing default training source. Provide a valid file manually if no fallback exists.

### Step 4: Train a model

```powershell
python train.py
```

Answer the interactive questions:

1. **property**: Select the target CSV.
2. **data_src**: Select the prepared training directory.
3. **layers / neurons / heads**: Model capacity; beginners should press Enter for defaults.
4. **attention options**: Use defaults for the first run.
5. **train size**: The training fraction; `0.8` means 80%.

Before training, verify that `Selected ... matching samples` reports the expected sample count. Training prints per-epoch training and validation losses. The completed model is stored as `TRAINED/<property>.pt`.

Training the same property again may overwrite the checkpoint. Back up important models first.

### Step 5: Evaluate the model

```powershell
python evaluate.py
```

Select exactly the same property, source, layers, neurons, heads, and attention settings used during training. Different architecture settings cause checkpoint size-mismatch errors.

Open `RESULTS/<property>_results.csv` to compare measured and predicted values. For regression, a smaller MAE means a smaller average prediction error, but it must be interpreted with the property's units and range.

### Step 6: Repeat with command-line arguments

After recording a working interactive configuration, pass it directly:

```powershell
python train.py --property density --data_src CIF-DATA_NEW
python evaluate.py --property density --data_src CIF-DATA_NEW
python predict.py --property density --data_src my-samples
```

### Beginner checks

- Run commands from `GATGNN/`, not the repository root.
- Use numeric training CIF filenames matching the first CSV column exactly.
- Confirm that the property checkpoint exists in `TRAINED/` before prediction.
- Keep model options identical between training and evaluation.
- Close `id_prop.csv` in Excel before running scripts.
- Reduce batch size or use CPU when CUDA memory is insufficient.

See the [root beginner workflow](../README_EN.md#usage-manuals) for the complete pipeline beginning with CHGNet optimization.
