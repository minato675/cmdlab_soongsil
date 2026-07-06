# CHGNet Structure Optimization Manual

[한국어](README.md) | [English](README_EN.md)

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
cd C:\work
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

Place all `.cif` files to optimize in `C:\work\chgnet\Element`.

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
cd C:\work\chgnet
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

The script was probably launched outside `C:\work\chgnet`. Change to that directory and try again.

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
- [ ] Run `python optimizer.py` from `C:\work\chgnet`
- [ ] Compare input and output file counts
- [ ] Review optimized structures and logs
