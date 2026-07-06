import os
import pandas as pd
import numpy as np
from shutil import copyfile
from pathlib import Path

from gatgnn.interactive_config import resolve_data_source


def _ensure_atom_init(cif_dir):
    dst = os.path.join('DATA', cif_dir, 'atom_init.json')
    if os.path.exists(dst):
        return

    candidates = [
        os.path.join('DATA', 'train&evaluate', name, 'atom_init.json')
        for name in ('CIF-DATA', 'CIF-DATA_CMD', 'CIF-DATA_NEW')
    ]
    for src in candidates:
        if os.path.exists(src):
            copyfile(src, dst)
            return

    raise FileNotFoundError(
        f"Missing atom_init.json. Expected '{dst}', and no fallback exists under DATA/train&evaluate."
    )


def use_property(property_name, source, do_prediction=False):
    print('> Preparing dataset to use for Property Prediction. Please wait ...')

    # -----------------------------
    # Property -> filename mapping
    # -----------------------------
    if property_name in ['band', 'bandgap', 'band-gap']:
        filename = 'bandgap.csv'              ; p = 1 ; num_T = 36720

    elif property_name in ['bulk', 'bulkmodulus', 'bulk-modulus', 'bulk-moduli']:
        filename = 'bulkmodulus.csv'          ; p = 3 ; num_T = 4664

    elif property_name in ['energy-1', 'formationenergy', 'formation-energy']:
        filename = 'formationenergy.csv'      ; p = 2 ; num_T = 60000

    elif property_name in ['energy-2', 'fermienergy', 'fermi-energy']:
        filename = 'fermienergy.csv'          ; p = 2 ; num_T = 60000

    elif property_name in ['energy-3', 'absoluteenergy', 'absolute-energy']:
        filename = 'absoluteenergy.csv'       ; p = 2 ; num_T = 60000

    elif property_name in ['shear', 'shearmodulus', 'shear-modulus', 'shear-moduli']:
        filename = 'shearmodulus.csv'         ; p = 4 ; num_T = 4664

    elif property_name in ['poisson', 'poissonratio', 'poisson-ratio']:
        filename = 'poissonratio.csv'         ; p = None ; num_T = None

    elif property_name in ['is_metal', 'is_not_metal']:
        filename = 'ismetal.csv'              ; p = 2 ; num_T = 55391

    elif property_name in ['density', 'rho']:
        filename = 'density.csv'              ; p = None ; num_T = None

    elif property_name in ['thermal-conductivity', 'thermal_conductivity', 'thermalconductivity', 'kappa']:
        filename = 'thermalconductivity.csv'  ; p = None ; num_T = None

    elif property_name in ['new-property']:
        filename = 'newproperty.csv'          ; p = None ; num_T = None

    elif property_name in ['new_bulk-modulus', 'newbulkmodulus', 'new-bulk-modulus']:
        filename = 'newbulkmodulus.csv'       ; p = None ; num_T = None

    elif property_name in ['new_Youngs-modulus', 'newyoungsmodulus', 'new-youngs-modulus']:
        filename = 'newyoungsmodulus.csv'     ; p = None ; num_T = None

    else:
        # New properties are discovered from DATA/properties-reference/*.csv.
        # Match either the exact stem or a hyphen/underscore-insensitive stem.
        property_dir = Path('DATA/properties-reference')
        normalized = property_name.lower().replace('-', '').replace('_', '')
        matches = [
            path for path in property_dir.glob('*.csv')
            if path.stem.lower().replace('-', '').replace('_', '') == normalized
        ]
        if not matches:
            raise ValueError(
                f"Unknown property_name: {property_name}. "
                f"Add DATA/properties-reference/{property_name}.csv first."
            )
        filename = matches[0].name
        p = None
        num_T = None

    # -----------------------------
    # Read property CSV
    # -----------------------------
    df = (
        pd.read_csv(f'DATA/properties-reference/{filename}', header=None, names=['material_id', 'value'])
        .replace(to_replace='None', value=np.nan)
        .dropna()
    )

    # -----------------------------
    # Normalize/clean material_id
    # -----------------------------
    df['material_id'] = df['material_id'].astype(str).str.strip()
    # Excel/float로 들어온 "1328.0" 같은 케이스 방지
    df['material_id'] = df['material_id'].str.replace(r'\.0$', '', regex=True)

    # Every training data source uses plain numeric CIF filenames.

    # -----------------------------
    # Dataset source handling
    # -----------------------------
    if source in ['CGCNN', 'CIF-DATA']:
        cif_dir = os.path.join('train&evaluate', 'CIF-DATA')
        if filename in ['bulkmodulus.csv', 'shearmodulus.csv', 'poissonratio.csv']:
            small = pd.read_csv('DATA/cgcnn-reference/mp-ids-3402.csv', header=None, names=['mp_ids']).values.squeeze()
            df = df[df.material_id.isin(small)]
            num_T = 2041
        elif filename == 'bandgap.csv':
            medium = pd.read_csv('DATA/cgcnn-reference/mp-ids-27430.csv', header=None, names=['mp_ids']).values.squeeze()
            df = df[df.material_id.isin(medium)]
            num_T = 16458
        elif filename in ['formationenergy.csv', 'fermienergy.csv', 'ismetal.csv', 'absoluteenergy.csv']:
            large = pd.read_csv('DATA/cgcnn-reference/mp-ids-46744.csv', header=None, names=['mp_ids']).values.squeeze()
            df = df[df.material_id.isin(large)]
            num_T = 28046
        CIF_dict = {'radius': 8, 'step': 0.2, 'max_num_nbr': 12}

    elif source == 'MEGNET':
        cif_dir = os.path.join('train&evaluate', 'CIF-DATA')
        megnet_df = pd.read_csv('DATA/megnet-reference/megnet.csv')

        if p is None:
            raise ValueError(
                f"MEGNET source requires a valid column index p, but got p=None for property {property_name}"
            )

        use_ids = megnet_df[megnet_df.iloc[:, p] == 1].material_id.values.squeeze()
        df = df[df.material_id.isin(use_ids)]
        CIF_dict = {'radius': 4, 'step': 0.5, 'max_num_nbr': 16}

    elif source in ['CMD', 'CIF-DATA_CMD']:
        cif_dir = os.path.join('train&evaluate', 'CIF-DATA_CMD')
        CIF_dict = {'radius': 8, 'step': 0.2, 'max_num_nbr': 12}

        # atom_init.json 필요하면 복사 (CMD 폴더에 없을 때 대비)
        _ensure_atom_init(cif_dir)

    elif source in ['NEW', 'CIF-DATA_NEW']:
        cif_dir = os.path.join('train&evaluate', 'CIF-DATA_NEW')
        CIF_dict = {'radius': 8, 'step': 0.2, 'max_num_nbr': 12}

        _ensure_atom_init(cif_dir)

    else:
        cif_dir, _ = resolve_data_source(source)
        if not os.path.isdir(os.path.join('DATA', cif_dir)):
            raise ValueError(f"Unknown source directory: DATA/{cif_dir}")
        CIF_dict = {'radius': 8, 'step': 0.2, 'max_num_nbr': 12}
        _ensure_atom_init(cif_dir)

    # Build the training table from the CIF files that actually exist in the
    # selected data-source folder. CIF filenames must be plain numeric IDs.
    cif_root = Path('DATA') / cif_dir
    cif_files = list(cif_root.glob('*.cif')) + list(cif_root.glob('*.cif.gz'))
    invalid_names = []
    available_ids = set()
    for cif_path in cif_files:
        cif_id = cif_path.name[:-7] if cif_path.name.endswith('.cif.gz') else cif_path.stem
        if not cif_id.isdigit():
            invalid_names.append(cif_path.name)
        else:
            available_ids.add(cif_id)

    if invalid_names:
        preview = ', '.join(invalid_names[:10])
        raise ValueError(
            f"All CIF filenames in DATA/{cif_dir} must be numeric. "
            f"Invalid files: {preview}"
        )
    if not available_ids:
        raise FileNotFoundError(f"No numeric CIF files found in DATA/{cif_dir}")

    reference_count = len(df)
    df = df[df['material_id'].isin(available_ids)].copy()
    if df.empty:
        raise ValueError(
            f"No matching IDs between DATA/properties-reference/{filename} "
            f"and DATA/{cif_dir}"
        )
    print(
        f"> Selected {len(df)} matching samples from {len(available_ids)} CIF files "
        f"and {reference_count} property rows."
    )

    # -----------------------------
    # Additional cleaning
    # -----------------------------
    # bulk/shear 계열처럼 양수만 허용하는 케이스
    if p in [3, 4]:
        df = df[df.value > 0]

    # -----------------------------
    # Save id_prop.csv (overwrite-friendly)
    # -----------------------------
    out_path = f'DATA/{cif_dir}/id_prop.csv'
    tmp_path = out_path + '.tmp'
    df.to_csv(tmp_path, index=False, header=False)

    try:
        os.replace(tmp_path, out_path)
    except PermissionError:
        raise PermissionError(
            f"Permission denied while writing '{out_path}'.\n"
            f"- '{out_path}'가 Excel/편집기에서 열려있지 않은지 확인\n"
            f"- 파일/폴더가 읽기전용이 아닌지 확인\n"
            f"- 임시 파일은 '{tmp_path}'로 저장되어 있습니다."
        )

    if not do_prediction:
        print(f'> Dataset for {source}---{property_name} ready !\n\n')

    return source, num_T, CIF_dict
