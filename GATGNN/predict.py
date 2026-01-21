import sys
import os
import argparse
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from gatgnn.data                   import *
from gatgnn.model                  import *
from gatgnn.pytorch_early_stopping import *
from gatgnn.file_setter            import use_property
from gatgnn.utils                  import *

def _stem_cif_name(p: Path) -> str:
    """Return cif id stem for *.cif or *.cif.gz"""
    name = p.name
    if name.endswith(".cif.gz"):
        return name[:-7]  # remove .cif.gz
    if name.endswith(".cif"):
        return name[:-4]  # remove .cif
    return p.stem

def resolve_targets(to_predict: str):
    """
    --to_predict 처리:
      1) directory path -> 그 안의 모든 .cif / .cif.gz 예측
      2) file path      -> 해당 파일 1개 예측
      3) id/name string -> DATA/prediction-directory/{id}.cif 예측
    Returns: (material_ids(list[str]), root_dir(str))
    """
    p = Path(to_predict)

    if p.exists() and p.is_dir():
        files = sorted(list(p.glob("*.cif")) + list(p.glob("*.cif.gz")))
        if not files:
            raise FileNotFoundError(f"No .cif or .cif.gz files found in directory: {p}")
        ids = [_stem_cif_name(f) for f in files]
        return ids, str(p)

    if p.exists() and p.is_file():
        ids = [_stem_cif_name(p)]
        return ids, str(p.parent)

    # fallback: treat as cif id name, look in default prediction-directory
    return [to_predict], "DATA/prediction-directory"


# MOST CRUCIAL DATA PARAMETERS
parser = argparse.ArgumentParser(description='GATGNN')

parser.add_argument(
    '--property',
    default='bulk-modulus',
    choices=[
        'absolute-energy', 'band-gap', 'bulk-modulus',
        'fermi-energy', 'formation-energy',
        'poisson-ratio', 'shear-modulus',
        'new-property',
        # ✅ added
        'new_bulk-modulus',
        'new_Youngs-modulus',
    ],
    help='material property to train/predict'
)

parser.add_argument(
    '--data_src',
    default='CGCNN',
    choices=['CGCNN', 'MEGNET', 'NEW'],
    help='selection of the materials dataset to use (default: CGCNN)'
)

# ✅ to_predict: id OR path-to-cif OR path-to-directory
parser.add_argument(
    '--to_predict',
    default='mp-1',
    help="cif id (without extension) OR a .cif/.cif.gz file path OR a directory path (predict all CIFs in it)"
)

# MOST CRUCIAL MODEL PARAMETERS
parser.add_argument('--num_layers', default=3, type=int,
                    help='number of AGAT layers to use in model (default:3)')
parser.add_argument('--num_neurons', default=64, type=int,
                    help='number of neurons to use per AGAT Layer(default:64)')
parser.add_argument('--num_heads', default=4, type=int,
                    help='number of Attention-Heads to use  per AGAT Layer (default:4)')
parser.add_argument('--use_hidden_layers', default=True, type=bool,
                    help='option to use hidden layers following global feature summation (default:True)')
parser.add_argument('--global_attention', default='composition', choices=['composition', 'cluster'],
                    help='selection of the unpooling method (default:composition)')
parser.add_argument('--cluster_option', default='fixed', choices=['fixed', 'random', 'learnable'],
                    help='selection of the cluster unpooling strategy (default: fixed)')
parser.add_argument('--concat_comp', default=False, type=bool,
                    help='option to re-use vector of elemental composition (default: False)')
parser.add_argument('--train_size', default=0.8, type=float,
                    help='ratio size of the training-set (default:0.8)')

args = parser.parse_args(sys.argv[1:])

# -----------------------------
# Property handling
# -----------------------------
prop_arg = args.property  # 사용자가 입력한 값 (모델 파일명/데이터 파일 선택용)

# 모델 설정(회귀/분류 등)은 내부 키로 통일하는 게 안전
MODEL_PROPERTY_ALIASES = {
    "new_bulk-modulus": "new-property",
    "new_Youngs-modulus": "new-property",
}
model_property = MODEL_PROPERTY_ALIASES.get(prop_arg, prop_arg)

data_src = args.data_src

# ✅ NEW 두 물성은 file_setter가 각각 newbulkmodulus.csv / newyoungsmodulus.csv를 읽도록
_, _, RSM = use_property(prop_arg, data_src, True)
norm_action, classification = set_model_properties(model_property)

# targets
material_ids, predict_root_dir = resolve_targets(args.to_predict)

number_layers       = args.num_layers
number_neurons      = args.num_neurons
n_heads             = args.num_heads
xtra_l              = args.use_hidden_layers
global_att          = args.global_attention
attention_technique = args.cluster_option
concat_comp         = args.concat_comp

# SETTING UP CODE TO RUN ON GPU
gpu_id = 0
device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')

# MODEL HYPER-PARAMETERS
learning_rate = 5e-3
batch_size    = 256
test_param    = {'batch_size': batch_size, 'shuffle': False}

# -----------------------------
# Build dataset for prediction
# -----------------------------
dataset = pd.DataFrame()
dataset['material_ids'] = [str(x).strip() for x in material_ids]
dataset['label']        = [0.00001] * len(dataset)  # dummy label
NORMALIZER              = DATA_normalizer(dataset.label.values)

src_CIF = 'CIF-DATA_NEW' if data_src == 'NEW' else 'CIF-DATA'
CRYSTAL_DATA = CIF_Dataset(dataset, root_dir=f'DATA/{src_CIF}/', **RSM)

# ✅ prediction directory override (file/dir/id)
CRYSTAL_DATA.root_dir = predict_root_dir

test_idx    = list(range(len(dataset)))
testing_set = CIF_Lister(test_idx, CRYSTAL_DATA, NORMALIZER, norm_action, df=dataset, src=data_src)

# -----------------------------
# Network
# -----------------------------
the_network = GATGNN(
    n_heads, classification,
    neurons=number_neurons,
    nl=number_layers,
    xtra_layers=xtra_l,
    global_attention=global_att,
    unpooling_technique=attention_technique,
    concat_comp=concat_comp,
    edge_format=data_src
)
net = the_network.to(device)

# LOSS & OPTIMIZER (optimizer는 예측엔 사실상 불필요하지만 구조 유지)
if classification == 1:
    criterion = nn.CrossEntropyLoss().cuda()
    funct = torch_accuracy
else:
    criterion = nn.SmoothL1Loss().cuda()
    funct = torch_MAE
optimizer = optim.AdamW(net.parameters(), lr=learning_rate, weight_decay=1e-1)

# ✅ LOADING MODEL: 저장 파일명은 CLI property 기준
model_path = f'TRAINED/{prop_arg}.pt'
net.load_state_dict(torch.load(model_path, map_location=device))

# METRICS (예측 출력에는 거의 안 쓰지만, 내부 호환 위해 유지)
metrics = METRICS(model_property, 0, criterion, funct, device)

print(f'> PREDICTING MATERIAL-PROPERTY ...')
print(f'> model: {model_path}')
print(f'> root_dir: {predict_root_dir}')
print(f'> num_targets: {len(dataset)}')

# TESTING PHASE
test_loader = torch_DataLoader(dataset=testing_set, **test_param)
net.eval()

offset = 0
with torch.no_grad():
    for batch in test_loader:
        batch = batch.to(device)
        pred = net(batch)

        # pred shape 방어: (B,1) or (B,) 등
        pred_flat = pred.view(-1).detach().cpu().numpy()

        for i, val in enumerate(pred_flat):
            mid = dataset['material_ids'].iloc[offset + i]
            print(f'> {prop_arg} of material ({mid}.cif) = {float(val):.6f}')

        offset += len(pred_flat)
