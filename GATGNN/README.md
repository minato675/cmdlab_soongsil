# GATGNN 학습·평가·예측 작업 매뉴얼

[한국어](README.md) | [English](README_EN.md)

## 목차

- [주요 폴더](#1-주요-폴더)
- [실행 환경](#2-실행-환경)
- [지원 물성과 파일명](#3-지원-물성과-파일명)
- [데이터 소스 규칙](#4-데이터-소스-규칙)
- [학습 데이터 준비](#5-학습-데이터-준비)
- [모델 학습](#6-모델-학습)
- [모델 평가](#7-모델-평가)
- [새로운 CIF 물성 예측](#8-새로운-cif-물성-예측)
- [CIF 부피 계산](#9-cif-부피-계산)
- [주요 명령 옵션](#10-주요-명령-옵션)
- [권장 전체 작업 순서](#11-권장-전체-작업-순서)
- [문제 해결](#12-문제-해결)
- [실행 체크리스트](#13-실행-체크리스트)
- [사용 매뉴얼](#사용-매뉴얼)

이 폴더에서는 CIF 결정 구조와 물성 CSV를 이용해 GATGNN 모델을 학습하고, 학습 모델을 평가하거나 새로운 CIF의 물성을 예측한다.

현재 작업 흐름은 다음과 같다.

```text
학습 CIF + DATA/properties-reference/<물성>.csv
  → DATA/<데이터셋>/id_prop.csv 자동 생성
  → train.py
  → TRAINED/<property>.pt
  → evaluate.py → RESULTS/<property>_results.csv
  → predict.py  → PREDICTIONS/pred_<property>_<source>_<target>.csv
```

## 1. 주요 폴더

| 경로 | 용도 | Git 추적 |
| --- | --- | --- |
| `DATA/train&evaluate/<data_src>/` | 학습·평가용 CIF 데이터 소스 | 폴더만 포함 |
| `DATA/prediction/<data_src>/` | 예측할 조성의 CIF 데이터 소스 | 폴더만 포함 |
| `DATA/properties-reference/` | 물성별 ID-값 CSV | 포함 |
| `TRAINED/` | 학습 모델 체크포인트 | 현재 모델 저장 위치 |
| `RESULTS/` | 평가 결과 CSV | 폴더만 포함 |
| `PREDICTIONS/` | 예측 결과 CSV | 폴더만 포함 |

`DATA`에는 위 세 폴더만 둔다. `properties-reference`의 CSV는 Git에 포함하며, 나머지 data source 폴더는 `.gitkeep`만 포함하고 실제 CIF는 로컬에서 관리한다.

## 2. 실행 환경

모든 명령은 상대 경로를 사용하므로 저장소 루트의 `GATGNN/`에서 실행한다.

```powershell
cd GATGNN
```

필수 Python 패키지는 다음과 같다.

- PyTorch
- PyTorch Geometric
- NumPy, pandas, scikit-learn
- pymatgen
- tabulate

PyTorch와 PyTorch Geometric 계열 패키지는 CPU/CUDA 및 각 패키지 버전이 서로 맞아야 한다. 기존에 계산이 검증된 가상환경을 우선 사용한다.

기본 import 확인:

```powershell
python -c "import torch, torch_geometric, pymatgen, pandas, sklearn; print('GATGNN ready')"
```

GPU 사용 여부 확인:

```powershell
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

스크립트는 CUDA가 있으면 `cuda:0`, 없으면 CPU를 선택한다.

## 3. 지원 물성과 파일명

`--property` 값에 따라 `DATA/properties-reference`에서 다음 CSV를 읽는다.

| `--property` | 참조 CSV | 모델 파일 |
| --- | --- | --- |
| `bulk-modulus` | `bulkmodulus.csv` | `TRAINED/bulk-modulus.pt` |
| `shear-modulus` | `shearmodulus.csv` | `TRAINED/shear-modulus.pt` |
| `poisson-ratio` | `poissonratio.csv` | `TRAINED/poisson-ratio.pt` |
| `density` | `density.csv` | `TRAINED/density.pt` |
| `thermal-conductivity` | `thermalconductivity.csv` | `TRAINED/thermal-conductivity.pt` |
| `new-property` | `newproperty.csv` | `TRAINED/new-property.pt` |
| `new_bulk-modulus` | `newbulkmodulus.csv` | `TRAINED/new_bulk-modulus.pt` |
| `new_Youngs-modulus` | `newyoungsmodulus.csv` | `TRAINED/new_Youngs-modulus.pt` |

그 밖에 `absolute-energy`, `band-gap`, `fermi-energy`, `formation-energy`도 지원한다.

참조 CSV에는 헤더 없이 두 열만 둔다.

```csv
6794,7.81
6904,8.03
6905,7.95
```

첫 번째 열은 CIF ID, 두 번째 열은 숫자 물성값이다. 빈 값과 문자열 `None`은 제거된다.

## 4. 데이터 소스 규칙

### CMD

현재 사용자 데이터의 주된 형식이다.

- CIF 위치: `DATA/train&evaluate/CIF-DATA_CMD/`
- CIF 이름: `<숫자 ID>.cif`
- 참조 CSV의 ID: 접두사 없는 `<ID>`
- 실행 옵션: `--data_src CMD`

예를 들어 CSV의 ID가 `6794`이면 CIF 파일은 `6794.cif`여야 한다. 선택한 폴더에서 CSV ID와 일치하는 CIF만 학습에 자동 사용된다.

### NEW

- CIF 위치: `DATA/train&evaluate/CIF-DATA_NEW/`
- 실행 옵션: `--data_src NEW`
- 사용자 정의 물성에는 일반적으로 `new-property`를 사용한다.

### CGCNN / MEGNET

- CIF 위치: `DATA/train&evaluate/CIF-DATA/`
- 실행 옵션: 각각 `--data_src CGCNN`, `--data_src MEGNET`
- 원본 참조 데이터와 필터 파일이 필요하다.

각 학습 data source에는 `atom_init.json`이 필요하다. 파일이 없으면 코드가 `DATA/train&evaluate`의 기존 기본 데이터 소스에서 자동 복사한다.

## 5. 학습 데이터 준비

### 대화형 실행

`train.py`, `evaluate.py`, `predict.py`를 옵션 없이 실행하면 단계별 설정 메뉴가 열린다.

```powershell
python train.py
python evaluate.py
python predict.py
```

1. `DATA/properties-reference/*.csv`에서 발견한 물성을 번호로 선택한다.
2. 학습·평가는 `DATA/train&evaluate`, 예측은 `DATA/prediction` 아래의 data source 폴더를 선택한다.
3. 나머지 옵션은 값을 직접 입력하거나 Enter를 눌러 표시된 기본값을 사용한다.

새 물성을 추가하려면 헤더 없는 `ID,값` 형식의 `<새물성>.csv`를 `DATA/properties-reference/`에 넣고 스크립트를 다시 실행한다. 새 CSV의 파일명이 메뉴에 자동으로 나타나며 기본 회귀 물성으로 처리된다. 기존 명령행 방식도 계속 사용할 수 있다.

```powershell
python train.py --property 새물성 --data_src CIF-DATA_CMD
```

### 데이터 준비 예시

예: CMD 밀도 모델을 학습하는 경우

1. `DATA/properties-reference/density.csv`를 준비한다.
2. 대응하는 `<숫자 ID>.cif`를 선택할 `DATA/train&evaluate/<data_src>/`에 넣는다.
3. CSV의 모든 ID에 대응하는 CIF가 존재하는지 확인한다.
4. 다른 프로그램에서 `id_prop.csv`를 열어 두었다면 닫는다.

`train.py`, `evaluate.py`, `predict.py`가 실행될 때 `file_setter.py`가 다음 파일을 자동 생성하거나 덮어쓴다.

```text
DATA/train&evaluate/CIF-DATA_CMD/id_prop.csv
```

따라서 이 파일을 Excel에서 열어 둔 상태로 실행하면 `PermissionError`가 발생할 수 있다.

## 6. 모델 학습

기본 CMD 학습 예시:

```powershell
python train.py --property density --data_src CIF-DATA_CMD
```

현재 사용하는 물성별 예시:

```powershell
python train.py --property thermal-conductivity --data_src CIF-DATA_CMD
python train.py --property poisson-ratio --data_src CIF-DATA_CMD
python train.py --property new_bulk-modulus --data_src CIF-DATA_CMD
python train.py --property new_Youngs-modulus --data_src CIF-DATA_CMD
```

NEW 사용자 정의 데이터 예시:

```powershell
python train.py --property new-property --data_src NEW --train_size 0.8
```

기본 학습 설정은 코드에 고정되어 있다.

- 최대 epoch: 200
- batch size: 256
- learning rate: `5e-3`
- optimizer: AdamW
- early stopping patience: 150
- random seed: 456
- 기본 train 비율: 0.8 (`training_num`이 정해진 기존 데이터는 코드 설정 우선)

학습 중 최적 체크포인트는 `TRAINED/crystal-checkpoint.pt`에 저장되고, 종료 후 다음 이름으로 복사된다.

```text
TRAINED/<property>.pt
```

같은 `--property`로 다시 학습하면 기존 모델을 덮어쓸 수 있으므로 필요한 모델은 먼저 백업한다.

## 7. 모델 평가

학습 때 사용한 옵션과 동일한 모델 구조 옵션을 사용해야 한다.

```powershell
python evaluate.py --property density --data_src CIF-DATA_CMD
```

결과는 다음 위치에 저장된다.

```text
RESULTS/density_results.csv
```

CSV에는 material ID, 실제값, 예측값, 원자 수 및 데이터 인덱스가 기록된다.

레이어 수나 attention 설정을 바꾸어 학습했다면 평가에도 그대로 전달한다.

```powershell
python evaluate.py --property density --data_src CIF-DATA_CMD --num_layers 5 --global_attention cluster --cluster_option fixed
```

모델 구조 옵션이 학습 때와 다르면 체크포인트 로딩 시 크기 불일치 오류가 발생한다.

## 8. 새로운 CIF 물성 예측

`predict.py --to_predict`는 세 가지 입력 방식을 지원한다.

### 폴더 전체 예측

```powershell
python predict.py --property density --data_src prediction-directory
```

폴더 바로 아래의 모든 `.cif`, `.cif.gz`를 정렬하여 예측한다.

### CIF 파일 하나 예측

```powershell
python predict.py --property density --data_src prediction-directory --to_predict DATA\prediction\prediction-directory\6794.cif
```

### 기본 폴더에 있는 ID 하나 예측

```powershell
python predict.py --property density --data_src prediction-directory --to_predict 6794
```

마지막 방식의 기본 경로는 `DATA/prediction/prediction-directory/6794.cif` 또는 `.cif.gz`이다.

출력 예:

```text
PREDICTIONS/pred_density_CMD_prediction-directory.csv
```

출력 CSV 형식:

```csv
material_id,prediction
6794,7.812345
```

예측할 때도 학습 당시의 `--property`, `--data_src` 및 모델 구조 옵션을 동일하게 사용한다.

## 9. CIF 부피 계산

`volume_predict.py`는 GATGNN 모델을 사용하지 않고 pymatgen으로 CIF unit-cell 부피를 계산한다.

```powershell
python volume_predict.py --to_predict DATA\prediction\prediction-directory
```

파일 하나 또는 ID 하나도 `predict.py`와 같은 방식으로 지정할 수 있다.

```powershell
python volume_predict.py --to_predict DATA\prediction\prediction-directory\6794.cif
python volume_predict.py --to_predict 6794
```

기본 출력 위치:

```text
PREDICTIONS/volume_prediction-directory.csv
```

다른 출력 폴더를 사용하려면 `--out_dir`을 지정한다.

## 10. 주요 명령 옵션

| 옵션 | 기본값 | 설명 |
| --- | --- | --- |
| `--property` | `bulk-modulus` | 학습·평가·예측할 물성 |
| `--data_src` | `CGCNN` | `CGCNN`, `MEGNET`, `NEW`, `CMD` |
| `--to_predict` | 선택한 `DATA/prediction/<data_src>` | 예측 ID, CIF 파일 또는 폴더 |
| `--num_layers` | 3 | AGAT 레이어 수 |
| `--num_neurons` | 64 | 레이어당 뉴런 수 |
| `--num_heads` | 4 | attention head 수 |
| `--global_attention` | `composition` | `composition` 또는 `cluster` |
| `--cluster_option` | `fixed` | `fixed`, `random`, `learnable` |
| `--train_size` | 0.8 | 학습 데이터 비율 |

`--use_hidden_layers`와 `--concat_comp`는 현재 `argparse`에서 `type=bool`을 사용한다. 문자열 `False`도 기대와 다르게 참으로 해석될 수 있으므로 기본값을 바꿔야 한다면 명령행보다 코드 설정을 확인한다.

## 11. 권장 전체 작업 순서

CMD 밀도 모델의 한 사이클:

```powershell
cd GATGNN

# 1. 학습
python train.py --property density --data_src CIF-DATA_CMD

# 2. 평가
python evaluate.py --property density --data_src CIF-DATA_CMD

# 3. 새 CIF 일괄 예측
python predict.py --property density --data_src prediction-directory

# 4. 필요 시 부피 계산
python volume_predict.py --to_predict DATA\prediction\prediction-directory
```

## 12. 문제 해결

### `Missing atom_init.json`

선택한 학습 data source 또는 `DATA/train&evaluate`의 기본 데이터 소스에 `atom_init.json`이 필요하다.

### CIF를 찾지 못함

- 모든 학습 CIF가 `<숫자 ID>.cif` 형식인지 확인한다.
- 예측 ID 방식은 선택한 `DATA/prediction/<data_src>/<ID>.cif`를 찾는다.
- 명령을 저장소 루트의 `GATGNN/`에서 실행했는지 확인한다.

### `Permission denied ... id_prop.csv`

Excel이나 편집기에서 해당 `id_prop.csv`를 닫고 다시 실행한다.

### 모델 체크포인트가 없음

`TRAINED/<property>.pt`가 존재하는지 확인한다. `--property`의 대소문자와 밑줄·하이픈까지 파일명과 일치해야 한다.

### 체크포인트 크기 불일치

학습, 평가, 예측에 사용한 `num_layers`, `num_neurons`, `num_heads`, attention 옵션이 동일한지 확인한다.

### CUDA 메모리 부족

`train.py`의 `batch_size`를 줄인다. 현재 batch size는 명령행 옵션이 아니라 코드에 고정되어 있다.

### CSV 값 오류

참조 CSV는 헤더 없는 2열 형식이어야 한다. ID와 물성값 사이에 불필요한 쉼표가 없는지, 물성값이 숫자인지 확인한다.

## 13. 실행 체크리스트

- [ ] 저장소 루트의 `GATGNN/`에서 명령 실행
- [ ] 물성명과 참조 CSV 파일명 확인
- [ ] 참조 CSV ID와 CIF 파일명 대응 확인
- [ ] `atom_init.json` 확인
- [ ] 기존 `TRAINED/<property>.pt` 백업 여부 확인
- [ ] 학습과 평가·예측의 모델 옵션 일치
- [ ] `RESULTS/` 평가값 검토
- [ ] `PREDICTIONS/` 예측 결과 검토

## 사용 매뉴얼

### GATGNN에서 사용하는 용어

- **property**: 모델이 학습하거나 예측할 물성이다. 예: 밀도, 열전도도, 포아송비.
- **data source**: CIF를 용도별로 모은 폴더다. 학습용과 예측용 위치가 다르다.
- **CIF ID**: 학습 CIF 파일명에 쓰는 숫자 식별자다. `1328.cif`의 ID는 `1328`이다.
- **reference CSV**: 각 CIF ID의 정답 물성값을 적은 파일이다.
- **checkpoint**: 학습된 모델 파라미터 파일이며 `TRAINED/<property>.pt`에 저장된다.
- **epoch**: 전체 학습 데이터를 한 번 학습하는 단위다.

### 1단계: 최초 환경 설치

저장소 루트에서 한 번만 실행한다.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

새 PowerShell을 열었다면 저장소 루트에서 `.\.venv\Scripts\Activate.ps1`을 다시 실행한다. 이후 GATGNN 폴더로 이동한다.

```powershell
cd GATGNN
```

### 2단계: 기존 모델로 예측만 하기

처음 사용하는 경우 가장 간단한 과정이다.

1. 예측하려는 물성의 모델이 `TRAINED/`에 있는지 확인한다. 예를 들어 밀도는 `TRAINED/density.pt`가 필요하다.
2. `DATA/prediction/` 아래에 작업 이름의 폴더를 만든다.
3. 예측할 `.cif` 또는 `.cif.gz` 파일을 그 폴더에 넣는다.

```text
DATA/prediction/my-samples/
├─ sample-a.cif
└─ sample-b.cif
```

4. 대화형 예측을 실행한다.

```powershell
python predict.py
```

5. 첫 질문에서 물성을 선택한다.
6. 두 번째 질문에서 `my-samples`를 선택한다.
7. 나머지 모델 옵션은 체크포인트가 기본 설정으로 학습되었다면 Enter를 누른다.
8. `PREDICTIONS/pred_<property>_<data_src>_<target>.csv`를 열어 결과를 확인한다.

예측 CSV의 기본 열:

| 열 | 의미 |
| --- | --- |
| `material_id` | 확장자를 제외한 CIF 파일명 |
| `prediction` | 모델이 예측한 물성값 |

### 3단계: 학습 데이터 만들기

새 모델을 학습할 때만 필요하다.

1. `DATA/train&evaluate/` 아래에 data source 폴더를 만든다.
2. 모든 학습 CIF 이름을 `<숫자 ID>.cif`로 지정한다. `cmd-1328.cif`, `sample.cif` 같은 이름은 사용할 수 없다.
3. `DATA/properties-reference/`에 `<물성명>.csv`를 만든다.
4. CSV에는 헤더 없이 `숫자 ID,물성값`을 적는다.

```text
DATA/train&evaluate/my-training-data/
├─ 1328.cif
├─ 1329.cif
└─ atom_init.json
```

```csv
1328,107.68
1329,137.49
```

파일명 `my-property.csv`는 대화형 메뉴에서 `my-property`로 나타난다. 코드가 선택한 폴더의 실제 CIF ID와 CSV ID를 비교하고, 양쪽에 모두 존재하는 행만 `id_prop.csv`로 자동 생성한다.

`atom_init.json`이 선택 폴더에 없으면 기존 기본 학습 폴더에서 자동 복사한다. 기본 파일도 없다면 먼저 유효한 `atom_init.json`을 준비해야 한다.

### 4단계: 모델 학습

```powershell
python train.py
```

대화형 질문에 답한다.

1. **property**: 학습할 CSV 물성을 선택한다.
2. **data_src**: 방금 준비한 학습 data source를 선택한다.
3. **num layers / neurons / heads**: 모델 크기다. 처음에는 Enter로 기본값을 사용한다.
4. **attention 옵션**: 처음에는 기본값을 사용한다.
5. **train size**: 학습에 사용할 비율이다. 기본값 `0.8`은 80%를 의미한다.

학습 전 출력되는 `Selected ... matching samples`에서 예상한 데이터 수가 선택됐는지 확인한다. 학습 중에는 epoch별 학습·검증 손실이 출력된다. 완료된 모델은 `TRAINED/<property>.pt`에 저장된다.

주의: 같은 property로 다시 학습하면 기존 체크포인트를 덮어쓸 수 있다. 중요한 모델은 실행 전에 복사해 둔다.

### 5단계: 학습 모델 평가

```powershell
python evaluate.py
```

학습 때와 동일한 property, data source, layer, neuron, head 및 attention 설정을 선택한다. 설정이 다르면 체크포인트 크기 불일치 오류가 발생한다.

완료 후 `RESULTS/<property>_results.csv`에서 실제값과 예측값을 비교한다. 회귀 모델에서는 터미널의 MAE가 작을수록 평균 예측 오차가 작다는 뜻이다. 단, 물성 단위와 데이터 범위를 함께 고려해 판단한다.

### 6단계: 명령행 옵션으로 다시 실행하기

대화형 설정을 기록해 두었다면 다음부터 옵션을 직접 전달할 수 있다.

```powershell
python train.py --property density --data_src CIF-DATA_NEW
python evaluate.py --property density --data_src CIF-DATA_NEW
python predict.py --property density --data_src my-samples
```

### 초보자가 자주 확인할 사항

- 명령은 저장소 루트가 아니라 `GATGNN/`에서 실행한다.
- 학습 CIF 이름은 숫자만 사용하고 CSV의 첫 열과 정확히 일치시킨다.
- 예측할 property의 `.pt` 파일이 `TRAINED/`에 있는지 확인한다.
- 학습과 평가의 모델 옵션을 동일하게 사용한다.
- `id_prop.csv`를 Excel에서 열어 둔 채 실행하지 않는다.
- CUDA 메모리 오류가 나면 batch size를 줄이거나 CPU 환경을 사용한다.

CHGNet 최적화부터 이어지는 전체 과정은 [루트 초보자 사용 매뉴얼](../README.md#사용-매뉴얼)을 참고한다.
