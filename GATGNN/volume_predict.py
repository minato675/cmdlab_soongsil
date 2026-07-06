import sys
import argparse
from pathlib import Path

import pandas as pd
from pymatgen.core import Structure


def _stem_cif_name(p: Path) -> str:
    """Return cif id stem for *.cif or *.cif.gz"""
    name = p.name
    if name.endswith(".cif.gz"):
        return name[:-7]   # remove .cif.gz
    if name.endswith(".cif"):
        return name[:-4]   # remove .cif
    return p.stem


def resolve_targets(to_predict: str):
    """
    --to_predict 처리:
      1) directory path -> 그 안의 모든 .cif / .cif.gz 처리
      2) file path      -> 해당 파일 1개 처리
      3) id/name string -> DATA/prediction-directory/{id}.cif 처리
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
    return [to_predict], "DATA/prediction/prediction-directory"


def get_cif_path(root_dir: str, material_id: str) -> Path:
    """
    material_id에 대해 root_dir 안에서 .cif 또는 .cif.gz 파일 경로를 찾음
    """
    p1 = Path(root_dir) / f"{material_id}.cif"
    p2 = Path(root_dir) / f"{material_id}.cif.gz"

    if p1.exists():
        return p1
    if p2.exists():
        return p2

    raise FileNotFoundError(f"CIF file not found for {material_id} in {root_dir}")


def compute_volume(cif_path: Path) -> float:
    """
    CIF 파일에서 unit cell volume 계산 (Å^3)
    """
    structure = Structure.from_file(str(cif_path))
    return float(structure.volume)


def main():
    parser = argparse.ArgumentParser(description="Compute CIF volumes and save to CSV")
    parser.add_argument(
        "--to_predict",
        default="mp-1",
        help="cif id (without extension) OR a .cif/.cif.gz file path OR a directory path"
    )
    parser.add_argument(
        "--out_dir",
        default="PREDICTIONS",
        help="directory to save output csv"
    )

    args = parser.parse_args(sys.argv[1:])

    material_ids, predict_root_dir = resolve_targets(args.to_predict)

    print("> COMPUTING CIF VOLUMES ...")
    print(f"> root_dir: {predict_root_dir}")
    print(f"> num_targets: {len(material_ids)}")

    results = []

    for mid in material_ids:
        try:
            cif_path = get_cif_path(predict_root_dir, mid)
            volume = compute_volume(cif_path)

            print(f"> volume of material ({cif_path.name}) = {volume:.6f} A^3")

            results.append([mid, volume])

        except Exception as e:
            print(f"[ERROR] {mid}: {e}")
            results.append([mid, None])

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tag = Path(args.to_predict).stem if Path(args.to_predict).exists() else str(args.to_predict)
    out_path = out_dir / f"volume_{tag}.csv"

    df_out = pd.DataFrame(results, columns=["material_id", "volume_A3"])
    df_out.to_csv(out_path, index=False, encoding="utf-8-sig")

    print(f"\n> Saved volumes to: {out_path}\n")


if __name__ == "__main__":
    main()
