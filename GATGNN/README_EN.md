# GATGNN Training, Evaluation, and Prediction Manual

[한국어](README.md) | [English](README_EN.md)

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

All scripts use relative paths. Run every command from `C:\work\GATGNN`.

```powershell
cd C:\work\GATGNN
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
python train.py --property density --data_src CMD
```

Examples for properties currently used in this repository:

```powershell
python train.py --property thermal-conductivity --data_src CMD
python train.py --property poisson-ratio --data_src CMD
python train.py --property new_bulk-modulus --data_src CMD
python train.py --property new_Youngs-modulus --data_src CMD
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
python evaluate.py --property density --data_src CMD
```

The result is written to:

```text
RESULTS/density_results.csv
```

The CSV contains the material ID, measured value, predicted value, number of atoms, and dataset index.

If training used custom attention or layer settings, repeat them during evaluation:

```powershell
python evaluate.py --property density --data_src CMD --num_layers 5 --global_attention cluster --cluster_option fixed
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
python predict.py --property density --data_src CMD --to_predict 6794
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
cd C:\work\GATGNN

# 1. Train
python train.py --property density --data_src CMD

# 2. Evaluate
python evaluate.py --property density --data_src CMD

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
- Confirm that the command is running from `C:\work\GATGNN`.

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

- [ ] Run commands from `C:\work\GATGNN`
- [ ] Confirm the property name and reference CSV filename
- [ ] Match every reference ID with a CIF filename
- [ ] Confirm that `atom_init.json` exists
- [ ] Back up an existing `TRAINED/<property>.pt` if needed
- [ ] Use identical model options for training, evaluation, and prediction
- [ ] Review evaluation output in `RESULTS/`
- [ ] Review prediction output in `PREDICTIONS/`
