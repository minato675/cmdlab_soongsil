import os
import pandas as pd
import numpy as np
from shutil import copyfile

def use_property(property_name, source, do_prediction=False):
    print('> Preparing dataset to use for Property Prediction. Please wait ...')

    # -----------------------------
    # Property -> filename mapping
    # -----------------------------
    if property_name in ['band','bandgap','band-gap']:
        filename = 'bandgap.csv'          ; p = 1 ; num_T = 36720

    elif property_name in ['bulk','bulkmodulus','bulk-modulus','bulk-moduli']:
        filename = 'bulkmodulus.csv'      ; p = 3 ; num_T = 4664

    elif property_name in ['energy-1','formationenergy','formation-energy']:
        filename = 'formationenergy.csv'  ; p = 2 ; num_T = 60000

    elif property_name in ['energy-2','fermienergy','fermi-energy']:
        filename = 'fermienergy.csv'      ; p = 2 ; num_T = 60000

    elif property_name in ['energy-3','absoluteenergy','absolute-energy']:
        filename = 'absoluteenergy.csv'   ; p = 2 ; num_T = 60000

    elif property_name in ['shear','shearmodulus','shear-modulus','shear-moduli']:
        filename = 'shearmodulus.csv'     ; p = 4 ; num_T = 4664

    elif property_name in ['poisson','poissonratio','poisson-ratio']:
        filename = 'poissonratio.csv'     ; p = 4 ; num_T = 4664

    elif property_name in ['is_metal','is_not_metal']:
        filename = 'ismetal.csv'          ; p = 2 ; num_T = 55391

    # ✅ 기존 new-property 유지
    elif property_name in ['new-property']:
        filename = 'newproperty.csv'      ; p = None ; num_T = None

    # ✅ 추가: new_bulk-modulus -> newbulkmodulus.csv
    elif property_name in ['new_bulk-modulus','newbulkmodulus','new-bulk-modulus']:
        filename = 'newbulkmodulus.csv'   ; p = None ; num_T = None

    # ✅ 추가: new_Youngs-modulus -> newyoungsmodulus.csv
    elif property_name in ['new_Youngs-modulus','newyoungsmodulus','new-youngs-modulus']:
        filename = 'newyoungsmodulus.csv' ; p = None ; num_T = None

    else:
        raise ValueError(f"Unknown property_name: {property_name}")

    # -----------------------------
    # Read property CSV
    # -----------------------------
    df = (
        pd.read_csv(f'DATA/properties-reference/{filename}', names=['material_id', 'value'])
        .replace(to_replace='None', value=np.nan)
        .dropna()
    )
    # --- FIX: make material_id match CIF filenames ---
    # newbulkmodulus.csv / newyoungsmodulus.csv have numeric ids (e.g., 1328)
    # but CIFs are named like "cmd-1328.cif"
    if property_name in ['new_bulk-modulus', 'newbulkmodulus', 'new-bulk-modulus',
                        'new_Youngs-modulus', 'newyoungsmodulus', 'new-youngs-modulus']:
        df['material_id'] = df['material_id'].astype(str).str.strip()
        df['material_id'] = df['material_id'].str.replace(r'\.0$', '', regex=True)
        df['material_id'] = 'cmd-' + df['material_id']
    else:
        # for other datasets, still enforce string (safe)
        df['material_id'] = df['material_id'].astype(str).str.strip()

    # -----------------------------
    # Dataset source handling
    # -----------------------------
    if source == 'CGCNN':
        cif_dir = 'CIF-DATA'
        if filename in ['bulkmodulus.csv','shearmodulus.csv','poissonratio.csv']:
            small = pd.read_csv('DATA/cgcnn-reference/mp-ids-3402.csv', names=['mp_ids']).values.squeeze()
            df = df[df.material_id.isin(small)]
            num_T = 2041
        elif filename == 'bandgap.csv':
            medium = pd.read_csv('DATA/cgcnn-reference/mp-ids-27430.csv', names=['mp_ids']).values.squeeze()
            df = df[df.material_id.isin(medium)]
            num_T = 16458
        elif filename in ['formationenergy.csv','fermienergy.csv','ismetal.csv','absoluteenergy.csv']:
            large = pd.read_csv('DATA/cgcnn-reference/mp-ids-46744.csv', names=['mp_ids']).values.squeeze()
            df = df[df.material_id.isin(large)]
            num_T = 28046
        CIF_dict = {'radius': 8, 'step': 0.2, 'max_num_nbr': 12}

    elif source == 'MEGNET':
        cif_dir = 'CIF-DATA'
        megnet_df = pd.read_csv('DATA/megnet-reference/megnet.csv')

        # p가 None이면 MEGNET 필터를 적용할 수 없으니 방어
        if p is None:
            raise ValueError(f"MEGNET source requires a valid column index p, but got p=None for property {property_name}")

        use_ids = megnet_df[megnet_df.iloc[:, p] == 1].material_id.values.squeeze()
        df = df[df.material_id.isin(use_ids)]
        CIF_dict = {'radius': 4, 'step': 0.5, 'max_num_nbr': 16}

    elif source == 'NEW':
        cif_dir = 'CIF-DATA_NEW'
        CIF_dict = {'radius': 8, 'step': 0.2, 'max_num_nbr': 12}
        d_src = 'DATA'
        src, dst = d_src + '/CIF-DATA/atom_init.json', d_src + '/CIF-DATA_NEW/atom_init.json'

        # atom_init.json 복사(이미 존재해도 덮어씀)
        copyfile(src, dst)

    else:
        raise ValueError(f"Unknown source: {source}")

    # -----------------------------
    # Additional cleaning
    # -----------------------------
    if p in [3, 4]:
        df = df[df.value > 0]

    # -----------------------------
    # Save id_prop.csv (overwrite-friendly)
    # -----------------------------
    out_path = f'DATA/{cif_dir}/id_prop.csv'

    # Windows에서 잠김/권한 문제 회피용: 임시 파일로 저장 후 교체
    tmp_path = out_path + '.tmp'
    df.to_csv(tmp_path, index=False, header=False)

    try:
        os.replace(tmp_path, out_path)  # 원자적 교체(가능한 경우)
    except PermissionError:
        # 교체 실패 시 tmp를 남기고 에러를 더 명확히 전달
        raise PermissionError(
            f"Permission denied while writing '{out_path}'.\n"
            f"- '{out_path}'가 Excel/편집기에서 열려있지 않은지 확인\n"
            f"- 파일/폴더가 읽기전용이 아닌지 확인\n"
            f"- 임시 파일은 '{tmp_path}'로 저장되어 있습니다."
        )

    if not do_prediction:
        print(f'> Dataset for {source}---{property_name} ready !\n\n')

    return source, num_T, CIF_dict