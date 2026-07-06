# Material Structure Optimization and Property Prediction Workspace

[한국어](README.md) | [English](README_EN.md)

This repository is an integrated workspace for optimizing CIF crystal structures with CHGNet and training, evaluating, and running material-property predictions with GATGNN.

```text
Original CIF
  → CHGNet structure optimization
  → optimized CIF
  → GATGNN property prediction
  → prediction CSV
```

## Project layout

| Path | Purpose | Detailed manual |
| --- | --- | --- |
| `chgnet/` | CIF structure optimization with CHGNet | [한국어](chgnet/README.md) · [English](chgnet/README_EN.md) |
| `GATGNN/` | GATGNN training, evaluation, and prediction | [한국어](GATGNN/README.md) · [English](GATGNN/README_EN.md) |
| `requirements.txt` | Minimal combined CHGNet and GATGNN runtime | — |

## End-to-end workflow

### 1. Optimize structures with CHGNet

Place input CIF files in `chgnet/Element/` and run:

```powershell
cd C:\work\chgnet
python optimizer.py
```

Optimized structures are written to `chgnet/opt_cif/` with the original filenames.

The current script applies 10% strain and a `0.005` atomic-position perturbation before relaxing each structure with the default pretrained CHGNet model and the BFGS optimizer.

### 2. Prepare GATGNN prediction input

Copy the optimized CIF files into a data-source folder under `GATGNN/DATA/prediction/`.

```powershell
Copy-Item C:\work\chgnet\opt_cif\*.cif C:\work\GATGNN\DATA\prediction\prediction-directory\
```

To predict unoptimized structures, place the desired CIF files directly in the prediction directory instead.

### 3. Predict properties with GATGNN

Select a trained model and run prediction:

```powershell
cd C:\work\GATGNN
python predict.py --property density --data_src prediction-directory
```

The resulting CSV is saved in `GATGNN/PREDICTIONS/`.

Other property examples:

```powershell
python predict.py --property thermal-conductivity --data_src CMD --to_predict DATA\prediction-directory
python predict.py --property poisson-ratio --data_src CMD --to_predict DATA\prediction-directory
python predict.py --property new_bulk-modulus --data_src CMD --to_predict DATA\prediction-directory
python predict.py --property new_Youngs-modulus --data_src CMD --to_predict DATA\prediction-directory
```

Prediction requires `TRAINED/<property>.pt`. Use the same dataset source and architecture options that were used to train the checkpoint.

### 4. Optionally calculate CIF volumes

```powershell
cd C:\work\GATGNN
python volume_predict.py --to_predict DATA\prediction-directory
```

By default, the result is saved as `GATGNN/PREDICTIONS/volume_prediction-directory.csv`.

## Train and evaluate GATGNN models

Prepare the matching CIF files and property reference CSV before training a new model.

```powershell
cd C:\work\GATGNN
python train.py --property density --data_src CMD
python evaluate.py --property density --data_src CMD
```

- Property references: `GATGNN/DATA/properties-reference/<property>.csv`
- Training CIF: `GATGNN/DATA/train&evaluate/<data_src>/<ID>.cif`
- Trained model: `GATGNN/TRAINED/<property>.pt`
- Evaluation output: `GATGNN/RESULTS/<property>_results.csv`

See the [detailed GATGNN manual](GATGNN/README_EN.md) for dataset formats and supported property names.

## Runtime environments

Use the root `requirements.txt` on Python 3.10-3.12 to install the validated minimal dependencies for both projects.

```powershell
cd C:\work
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The default resolver may select a CPU PyTorch build. For NVIDIA GPU use, install the CUDA-compatible PyTorch build through the official PyTorch method before installing the remaining requirements.

The scripts use relative paths. Always change to the corresponding project directory before running them.

## Git policy

The following large inputs and generated outputs are excluded from Git:

- `chgnet/Element/`
- `chgnet/opt_cif/`
- Actual data under `GATGNN/DATA/train&evaluate/` and `GATGNN/DATA/prediction/`
- Generated files in `GATGNN/PREDICTIONS/`
- Generated files in `GATGNN/RESULTS/`

Only `.gitkeep` is tracked in each training/evaluation or prediction data-source folder and in `GATGNN/PREDICTIONS` and `GATGNN/RESULTS`, preserving the empty structure after checkout.

## Quick checklist

- [ ] Activate the appropriate Python environment
- [ ] Place original CIF files in `chgnet/Element/`
- [ ] Run structure optimization from `C:\work\chgnet`
- [ ] Copy optimized results into the GATGNN prediction directory
- [ ] Confirm that the required `TRAINED/<property>.pt` exists
- [ ] Run property prediction from `C:\work\GATGNN`
- [ ] Review the generated CSV in `PREDICTIONS/`

## Detailed documentation

- [CHGNet structure optimization manual](chgnet/README_EN.md)
- [GATGNN training, evaluation, and prediction manual](GATGNN/README_EN.md)
