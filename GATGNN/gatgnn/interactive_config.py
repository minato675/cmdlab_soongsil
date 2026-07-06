from argparse import ArgumentParser, Namespace
from pathlib import Path


DATA_ROOT = Path("DATA")
PROPERTY_ROOT = DATA_ROOT / "properties-reference"
TRAIN_ROOT = DATA_ROOT / "train&evaluate"
PREDICTION_ROOT = DATA_ROOT / "prediction"


def discover_properties():
    """Return property names from CSV filenames."""
    if not PROPERTY_ROOT.is_dir():
        return []
    aliases = {
        "absoluteenergy": "absolute-energy",
        "bandgap": "band-gap",
        "bulkmodulus": "bulk-modulus",
        "fermienergy": "fermi-energy",
        "formationenergy": "formation-energy",
        "newbulkmodulus": "new_bulk-modulus",
        "newproperty": "new-property",
        "newyoungsmodulus": "new_Youngs-modulus",
        "poissonratio": "poisson-ratio",
        "shearmodulus": "shear-modulus",
        "thermalconductivity": "thermal-conductivity",
    }
    properties = [aliases.get(path.stem.lower(), path.stem) for path in PROPERTY_ROOT.glob("*.csv")]
    return sorted(properties, key=str.lower)


def discover_data_directories(workflow="train"):
    """Return data-source directories for training/evaluation or prediction."""
    root = PREDICTION_ROOT if workflow == "predict" else TRAIN_ROOT
    if not root.is_dir():
        return []
    return sorted(
        (
            path.name
            for path in root.iterdir()
            if path.is_dir()
        ),
        key=str.lower,
    )


def resolve_data_source(data_src, workflow="train"):
    """Map legacy aliases or a directory name to (directory, graph format)."""
    aliases = {
        "CMD": "CIF-DATA_CMD",
        "NEW": "CIF-DATA_NEW",
        "CGCNN": "CIF-DATA",
        "MEGNET": "CIF-DATA",
    }
    source_name = aliases.get(data_src, data_src)
    base = "prediction" if workflow == "predict" else "train&evaluate"
    cif_dir = str(Path(base) / source_name)
    if data_src == "CIF-DATA":
        edge_src = "CGCNN"
    else:
        edge_src = data_src if data_src in {"CGCNN", "MEGNET"} else "NEW"
    return cif_dir, edge_src


def _select(prompt, options, default=None):
    print(f"\n{prompt}")
    for index, option in enumerate(options, start=1):
        print(f"  {index}. {option}")

    default_text = f" [{default}]" if default is not None else ""
    while True:
        value = input(f"> 선택 번호 또는 값 입력{default_text}: ").strip()
        if not value and default is not None:
            return default
        if value.isdigit() and 1 <= int(value) <= len(options):
            return options[int(value) - 1]
        if value in options:
            return value
        print("목록의 번호 또는 값을 입력하세요.")


def _convert(action, value):
    if action.type is None:
        return value
    if action.type is bool:
        normalized = value.strip().lower()
        if normalized in {"true", "t", "1", "yes", "y"}:
            return True
        if normalized in {"false", "f", "0", "no", "n"}:
            return False
        raise ValueError("true/false 중 하나를 입력하세요.")
    return action.type(value)


def parse_args_interactively(parser: ArgumentParser, argv, workflow="train"):
    """Use normal argparse with CLI arguments, or prompt for every option."""
    if argv:
        return parser.parse_args(argv)

    print("\n=== GATGNN 대화형 설정 ===")
    properties = discover_properties()
    if not properties:
        raise FileNotFoundError(f"CSV 파일이 없습니다: {PROPERTY_ROOT}")

    data_directories = discover_data_directories(workflow)
    if not data_directories:
        raise FileNotFoundError(f"사용 가능한 데이터 폴더가 없습니다: {DATA_ROOT}")

    values = {}
    for action in parser._actions:
        if action.dest == "help":
            continue
        if action.dest == "property":
            values[action.dest] = _select("1. property?", properties, action.default)
            continue
        if action.dest == "data_src":
            default = action.default
            default_dir = Path(resolve_data_source(default, workflow)[0]).name
            if default_dir in data_directories:
                default = default_dir
            elif default not in data_directories:
                default = data_directories[0]
            values[action.dest] = _select("2. data_src?", data_directories, default)
            continue

        label = action.dest.replace("_", " ")
        help_text = action.help or label
        default = action.default
        if action.dest == "to_predict" and workflow == "predict":
            default = str(PREDICTION_ROOT / values["data_src"])
        while True:
            raw = input(f"{label}? ({help_text}) [{default}]: ").strip()
            if not raw:
                values[action.dest] = default
                break
            try:
                converted = _convert(action, raw)
                if action.choices and converted not in action.choices:
                    raise ValueError(f"가능한 값: {', '.join(map(str, action.choices))}")
                values[action.dest] = converted
                break
            except ValueError as error:
                print(f"잘못된 값입니다: {error}")

    print("\n=== 선택된 설정 ===")
    for key, value in values.items():
        print(f"{key}: {value}")
    print()
    return parser.parse_args([], namespace=Namespace(**values))
