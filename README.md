# HEC-ML: Machine Learning Framework for Predicting Mechanical Properties of High-Entropy Carbides

This repository contains the implementation of the machine learning framework proposed in the paper:

> **DFT-Verified Database and Machine Learning Framework for Designing High Entropy Carbide with Superior Mechanical Strength/Elastic Stiffness**  
> Authors: Hyokyeong Kim, Jiho Kim, Kwonyeol Lee, Hayoung Son, Jaewon Choi, Inseong Bae, Haeun Lee, Sooah Kyung, and Jiwoong Kim  
> Journal: (TBD)

The framework is designed to predict the mechanical properties (**Bulk modulus** and **Young’s modulus**) of high-entropy carbides (HECs) using crystal graph neural networks (CGCNN/GATGNN).  
In addition, **CHGNet** is used to relax (optimize) candidate CIF structures before inference.

---

## Graphical Abstract



<img width="563" height="324" alt="image" src="https://https://github.com/minato675/cmdlab_soongsil/blob/master/front_pic.png" />

---

## 1. Overview

This project implements **CGCNN** and **GATGNN**-based models to learn quantitative relationships between **crystal structures (CIF)** and **elastic properties**.  
The models are trained on a **DFT-verified, internally consistent dataset** and applied to predict the mechanical properties of unexplored HEC compositions.

**Main features**
- Crystal structure-based learning (CIF input)
- Prediction of:
  - **Bulk modulus (B)**
  - **Young’s modulus (E)**
- Extrapolative screening of high-performance HEC candidates
- Optional **CHGNet-based structure relaxation** before model inference

---

## 2. Repository Structure

This repository is mainly composed of:
- `GATGNN/` : model training, evaluation, and prediction
- `chgnet/` : structure relaxation with CHGNet (`optimizer.py`)
- `requirements.txt` : Python dependencies

A simplified tree:

```
.
├── GATGNN/
│   ├── gatgnn/                 # GATGNN core modules
│   ├── train.py                # Training
│   ├── evaluate.py             # Evaluation
│   ├── predict.py              # Prediction (single CIF / directory)
│   ├── DATA/                   # Dataset & references (see below)
│   ├── TRAINED/                # Saved model checkpoints (*.pt)
│   └── RESULTS/                # Evaluation outputs (*.csv)
├── chgnet/
│   ├── optimizer.py            # CHGNet structure relaxation script
│   ├── Element/                # Input CIFs (to relax)
│   └── opt_cif/                # Output relaxed CIFs
└── requirements.txt
```

---

## 3. Requirements 

- Virtual environment recommended
- Python >= 3.11
- Other dependencies: see `requirements.txt`

Install:
```bash
pip install -r requirements.txt
```

---

## 4. Dataset (DFT-Verified In-house Database)

The dataset consists of **5,103 DFT calculation results**, all generated in-house using a **unified computational protocol** (identical exchange–correlation functionals, k-point sampling strategies, energy cutoffs, and convergence criteria).  
This ensures strong internal consistency and improves ML robustness.

### 4.1 Composition distribution (by dominant anion)
- **Nitrides:** 2,796
- **Carbides:** 1,386
- **Oxides:** 265
- **Others (non-C/N/O systems):** 656

### 4.2 Crystal system distribution
- **Hexagonal:** 1,177
- **Triclinic:** 1,091
- **Cubic:** 994
- **Orthorhombic:** 760
- **Monoclinic:** 627
- **Tetragonal:** 450
- **Rhombohedral:** 4

---

## 5. GATGNN DATA Directory Layout

After extracting the provided `DATA.zip`, the `GATGNN/DATA/` folder is organized as follows:

```
GATGNN/DATA/
├── cgcnn-reference/          # Reference ids/splits for CGCNN setting
├── CIF-DATA/                 # Default CIF dataset (CGCNN/MEGNET-compatible)
├── CIF-DATA_NEW/             # Lab-curated dataset (in-house DFT dataset)
├── megnet-reference/         # Reference files for MEGNET setting
├── prediction-directory/     # Put CIFs here for batch prediction
└── properties-reference/     # Property csv files (labels)
```

**Notes**
- `CIF-DATA_NEW/` contains the in-house dataset used to train custom models (`--data_src NEW`).
- `prediction-directory/` is used for inference on arbitrary CIF files.
- `properties-reference/` stores label tables used to generate `id_prop.csv`.

---

## 6. Workflow (CHGNet → GATGNN)

Recommended pipeline:
1) **(Optional) Relax/optimize** CIF structures using CHGNet  
2) **Train** a GATGNN model on `CIF-DATA_NEW`  
3) **Evaluate** the trained model  
4) **Predict** properties for new CIFs in batch

---

## 6.1 Structure Relaxation with CHGNet

Run CHGNet-based structural relaxation (geometry optimization) for candidate CIFs.

### Step 1 — Input CIF location
Place CIF files to relax in:
- `chgnet/Element/`

### Step 2 — Run optimizer
```bash
cd chgnet
python optimizer.py
```

### Output
Relaxed CIFs will be generated in:
- `chgnet/opt_cif/`

---

## 6.2 Training (GATGNN)

Training uses the lab dataset in:
- `GATGNN/DATA/CIF-DATA_NEW/`

Example (Bulk modulus):
```bash
cd GATGNN
python train.py --property new_bulk-modulus --data_src NEW --train_size 0.8
```

Example (Young’s modulus):
```bash
cd GATGNN
python train.py --property new_Youngs-modulus --data_src NEW --train_size 0.8
```

Saved model checkpoints:
- `GATGNN/TRAINED/new_bulk-modulus.pt`
- `GATGNN/TRAINED/new_Youngs-modulus.pt`

---

## 6.3 Evaluation (GATGNN)

Evaluate the trained model:

```bash
cd GATGNN
python evaluate.py --property new_bulk-modulus --data_src NEW --train_size 0.8
```

or:
```bash
python evaluate.py --property new_Youngs-modulus --data_src NEW --train_size 0.8
```

Evaluation results (CSV) are saved to:
- `GATGNN/RESULTS/new_bulk-modulus_results.csv`
- `GATGNN/RESULTS/new_Youngs-modulus_results.csv`

---

## 6.4 Prediction (GATGNN)

### Batch prediction for all CIFs in a directory
Put target CIF files into:
- `GATGNN/DATA/prediction-directory/`

Then run:
```bash
cd GATGNN
python predict.py --property new_bulk-modulus --data_src NEW --to_predict DATA/prediction-directory
```

For Young’s modulus:
```bash
python predict.py --property new_Youngs-modulus --data_src NEW --to_predict DATA/prediction-directory
```

### Predict a single CIF
You can also pass:
- a single CIF file path, or
- a single CIF id (without extension)

Examples:
```bash
python predict.py --property new_bulk-modulus --data_src NEW --to_predict DATA/prediction-directory/1328.cif
python predict.py --property new_bulk-modulus --data_src NEW --to_predict 1328
```

---

## 7. Custom Properties (Lab-defined)

This repository supports two custom properties for the lab dataset:

- `new_bulk-modulus`
- `new_Youngs-modulus`

These read labels from:
- `GATGNN/DATA/properties-reference/newbulkmodulus.csv`
- `GATGNN/DATA/properties-reference/newyoungsmodulus.csv`

**Important**
- The first column (material id) must match the CIF filename stem in `CIF-DATA_NEW/` (e.g., `1328` ↔ `1328.cif`).
- If your CIF filenames include prefixes (e.g., `mp-1328.cif`), the ids in the CSV must match exactly (`mp-1328`).

---

## 8. Reproducibility

All experiments reported in the paper can be reproduced using the scripts in this repository.  
Random seeds, data splits, and hyperparameters are defined in the training/evaluation scripts.

Reproduction checklist:
1) Extract `DATA.zip` into `GATGNN/DATA/`
2) Train:
   - `python train.py --property new_bulk-modulus --data_src NEW`
   - `python train.py --property new_Youngs-modulus --data_src NEW`
3) Evaluate:
   - `python evaluate.py --property new_bulk-modulus --data_src NEW`
   - `python evaluate.py --property new_Youngs-modulus --data_src NEW`
4) Predict for new CIFs:
   - `python predict.py --property new_bulk-modulus --data_src NEW --to_predict DATA/prediction-directory`

---

## 9. Citation

If you use this repository, please cite our paper (to be updated after publication).
