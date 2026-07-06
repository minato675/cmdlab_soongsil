# CHGNet Structure Optimization Manual

[한국어](README.md) | [English](README_EN.md)

## Table of contents

- [Directory layout](#1-directory-layout)
- [Set up the environment](#2-set-up-the-environment)
- [Prepare input CIF files](#3-prepare-input-cif-files)
- [Run structure optimization](#4-run-structure-optimization)
- [Current optimization settings](#5-current-optimization-settings)
- [Validate the results](#6-validate-the-results)
- [Troubleshooting](#7-troubleshooting)
- [Checklist](#8-checklist)
- [Usage manual](#usage-manual)

This directory uses a pretrained CHGNet model to relax multiple CIF structures in a batch.

```text
Element/*.cif
  → apply 10% strain
  → perturb atomic positions by 0.005
  → optimize with CHGNet + BFGS
  → opt_cif/*.cif
```

## 1. Directory layout

| Path | Purpose | Tracked by Git |
| --- | --- | --- |
| `Element/` | Original CIF input files | No |
| `opt_cif/` | Optimized CIF output files | No |
| `optimizer.py` | Batch optimization script | Yes |

Input structures and generated results are kept out of Git because they can become large.

## 2. Set up the environment

Python 3.10 or newer is required. The following example creates a virtual environment and installs this local CHGNet package in editable mode.

```powershell
# Run from the repository root
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Verify the installation:

```powershell
python -c "from chgnet.model import CHGNet; CHGNet.load(); print('CHGNet ready')"
```

When using CUDA, make sure the installed PyTorch build supports the local CUDA environment. PyTorch falls back to the CPU when CUDA is unavailable, which can make optimization significantly slower.

## 3. Prepare input CIF files

Place all `.cif` files to optimize in the repository's `chgnet/Element/` directory.

```text
chgnet/
├─ Element/
│  ├─ sample_01.cif
│  └─ sample_02.cif
├─ opt_cif/
└─ optimizer.py
```

The script processes only `.cif` files directly inside `Element`. It does not search subdirectories.

## 4. Run structure optimization

The script uses relative paths, so run it from the `chgnet` directory.

```powershell
cd chgnet
python optimizer.py
```

The script creates `opt_cif` automatically when it does not exist. Each result keeps the input filename.

```text
Element/sample_01.cif → opt_cif/sample_01.cif
```

An existing result with the same filename may be overwritten. Back up results that must be preserved before starting another run.

## 5. Current optimization settings

The following settings are fixed in `optimizer.py`:

- Model: default pretrained model loaded by `CHGNet.load()`
- Optimizer: `BFGS`
- Cell strain: `0.1` along each axis
- Atomic-position perturbation: `0.005`
- Input directory: `Element/`
- Output directory: `opt_cif/`

The input is therefore modified before relaxation. To relax the original structure directly, remove or comment out these lines:

```python
unrelaxed_structure.apply_strain([0.1, 0.1, 0.1])
unrelaxed_structure.perturb(0.005)
```

Change the optimizer here:

```python
relaxer = StructOptimizer(optimizer_class="BFGS")
```

Available examples include `FIRE`, `BFGS`, `LBFGS`, `LBFGSLineSearch`, `MDMin`, `SciPyFminCG`, `SciPyFminBFGS`, and `BFGSLineSearch`.

## 6. Validate the results

After a run:

1. Compare the number of input and output CIF files.
2. Open output structures in pymatgen, VESTA, or another structure viewer.
3. Check the terminal output for failed files.
4. For important calculations, independently validate energy, forces, stress, and convergence.

Count input and output files in PowerShell:

```powershell
(Get-ChildItem .\Element -Filter *.cif).Count
(Get-ChildItem .\opt_cif -Filter *.cif).Count
```

## 7. Troubleshooting

### `FileNotFoundError: Element`

The script was probably launched outside `chgnet/`. From the repository root, run `cd chgnet` and try again.

### Package import error

Make sure the virtual environment is active. From the repository root, run `python -m pip install -r requirements.txt` again.

### Only some CIF files fail

Check CIF syntax, element symbols, occupancies, and lattice data. The current script may stop the entire batch when one file raises an exception, so isolate the failing file before retrying.

### GPU memory error

The script processes one structure at a time. If memory is still insufficient, stop other GPU workloads or run on the CPU.

## 8. Checklist

- [ ] Activate the virtual environment
- [ ] Place input CIF files in `Element/`
- [ ] Back up existing `opt_cif/` results if needed
- [ ] Run `python optimizer.py` from the repository's `chgnet/` directory
- [ ] Compare input and output file counts
- [ ] Review optimized structures and logs

## Usage manual

### What does CHGNet do?

CHGNet reads atoms and lattice data from CIF files, predicts energies and forces, and relaxes the atoms and cell toward lower-force configurations. This repository processes every CIF in `Element/` and writes results to `opt_cif/`.

### First run

1. Create the environment and install packages from the repository root.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

2. Copy input CIF files into `chgnet/Element/`. Keep a separate backup of important originals.
3. Confirm the layout:

```text
chgnet/
├─ Element/
│  ├─ 1001.cif
│  └─ 1002.cif
├─ opt_cif/
└─ optimizer.py
```

4. From the repository root, enter the CHGNet directory and run:

```powershell
cd chgnet
python optimizer.py
```

5. Confirm that the terminal reports the start and completion of each file.
6. Check that files such as `opt_cif/1001.cif` and `opt_cif/1002.cif` were created.

### Important behavior before running

- The current code does not relax the untouched input directly. It first applies 10% strain along each axis and perturbs atomic positions by `0.005`.
- Output names match input names, so existing results may be overwritten.
- One invalid CIF may stop the entire batch. Move a failing file aside before retrying.
- CHGNet produces model predictions. Validate research-critical structures with an independent method such as DFT.

### How to validate results

1. Compare input and output CIF counts.
2. Open output files in VESTA or pymatgen and inspect for overlapping atoms or abnormal cells.
3. Review logs for tracebacks, isolated-atom warnings, or convergence problems.
4. Confirm that each filename corresponds to the intended structure.

```powershell
(Get-ChildItem .\Element -Filter *.cif).Count
(Get-ChildItem .\opt_cif -Filter *.cif).Count
```

### Next step

To predict properties with GATGNN, return to the repository root and copy optimized files into a prediction source.

```powershell
cd ..
New-Item -ItemType Directory -Force .\GATGNN\DATA\prediction\my-samples
Copy-Item .\chgnet\opt_cif\*.cif .\GATGNN\DATA\prediction\my-samples\
```

See the [root beginner workflow](../README_EN.md#usage-manuals) for the complete pipeline.
