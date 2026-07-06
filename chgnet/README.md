# CHGNet 구조 최적화 작업 매뉴얼

[한국어](README.md) | [English](README_EN.md)

## 목차

- [폴더 구성](#1-폴더-구성)
- [실행 환경 준비](#2-실행-환경-준비)
- [입력 CIF 준비](#3-입력-cif-준비)
- [구조 최적화 실행](#4-구조-최적화-실행)
- [현재 최적화 조건](#5-현재-최적화-조건)
- [결과 확인](#6-결과-확인)
- [자주 발생하는 문제](#7-자주-발생하는-문제)
- [작업 체크리스트](#8-작업-체크리스트)
- [사용 매뉴얼](#사용-매뉴얼)

이 폴더에서는 사전 학습된 CHGNet 모델로 여러 CIF 구조를 일괄 완화(relaxation)한다.

현재 작업 흐름은 다음과 같다.

```text
Element/*.cif
  → 초기 구조에 10% strain 적용
  → 원자 위치 0.005 만큼 perturb
  → CHGNet + BFGS 구조 최적화
  → opt_cif/*.cif
```

## 1. 폴더 구성

| 경로 | 용도 | Git 추적 |
| --- | --- | --- |
| `Element/` | 최적화할 원본 CIF 입력 | 제외 |
| `opt_cif/` | 최적화된 CIF 출력 | 제외 |
| `optimizer.py` | 일괄 구조 최적화 스크립트 | 포함 |

입력 파일과 계산 결과는 용량이 커질 수 있어 Git에 올리지 않는다.

## 2. 실행 환경 준비

Python 3.10 이상이 필요하다. 저장소 루트에서 가상환경을 만들고 CHGNet을 editable 모드로 설치하는 예시는 다음과 같다.

```powershell
# 저장소 루트에서 실행
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

설치 확인:

```powershell
python -c "from chgnet.model import CHGNet; CHGNet.load(); print('CHGNet ready')"
```

CUDA를 사용하는 경우에는 설치된 PyTorch가 현재 CUDA 환경을 지원하는지 별도로 확인한다. CUDA를 사용할 수 없으면 PyTorch가 CPU로 실행하며 계산 시간이 길어질 수 있다.

## 3. 입력 CIF 준비

최적화할 `.cif` 파일을 저장소 루트의 `chgnet/Element/`에 넣는다.

```text
chgnet/
├─ Element/
│  ├─ sample_01.cif
│  └─ sample_02.cif
├─ opt_cif/
└─ optimizer.py
```

스크립트는 `Element` 바로 아래에 있는 확장자 `.cif` 파일만 처리한다. 하위 폴더는 탐색하지 않는다.

## 4. 구조 최적화 실행

상대 경로를 사용하므로 반드시 `chgnet` 폴더에서 실행한다.

```powershell
cd chgnet
python optimizer.py
```

`opt_cif` 폴더가 없으면 자동으로 생성된다. 각 입력 파일과 같은 이름의 최적화 결과가 저장된다.

```text
Element/sample_01.cif → opt_cif/sample_01.cif
```

같은 이름의 출력 파일이 이미 있으면 덮어쓸 수 있으므로 필요한 결과는 실행 전에 별도로 백업한다.

## 5. 현재 최적화 조건

`optimizer.py`에는 다음 조건이 고정되어 있다.

- 모델: `CHGNet.load()`로 불러오는 기본 사전 학습 모델
- 최적화기: `BFGS`
- 셀 변형: 각 축에 `0.1` strain 적용
- 원자 위치 교란: `0.005`
- 입력 폴더: `Element/`
- 출력 폴더: `opt_cif/`

즉, 입력 구조를 그대로 완화하는 것이 아니라 먼저 의도적인 변형과 원자 위치 교란을 적용한 뒤 최적화한다. 원본 상태에서 바로 완화하려면 아래 두 줄을 제거하거나 주석 처리해야 한다.

```python
unrelaxed_structure.apply_strain([0.1, 0.1, 0.1])
unrelaxed_structure.perturb(0.005)
```

최적화 알고리즘을 바꾸려면 다음 값을 변경한다.

```python
relaxer = StructOptimizer(optimizer_class="BFGS")
```

사용 가능한 예에는 `FIRE`, `BFGS`, `LBFGS`, `LBFGSLineSearch`, `MDMin`, `SciPyFminCG`, `SciPyFminBFGS`, `BFGSLineSearch`가 있다.

## 6. 결과 확인

실행 후 다음을 확인한다.

1. 입력 CIF 수와 출력 CIF 수가 같은지 확인한다.
2. 출력 CIF를 pymatgen, VESTA 등의 도구로 열어 구조가 정상인지 확인한다.
3. 터미널에 오류가 발생한 파일이 없는지 확인한다.
4. 중요한 계산에는 에너지, 힘, 응력 및 수렴 조건을 별도로 검증한다.

간단한 파일 수 확인:

```powershell
(Get-ChildItem .\Element -Filter *.cif).Count
(Get-ChildItem .\opt_cif -Filter *.cif).Count
```

## 7. 자주 발생하는 문제

### `FileNotFoundError: Element`

`chgnet/`이 아닌 다른 위치에서 실행했을 가능성이 크다. 저장소 루트에서 `cd chgnet` 후 다시 실행한다.

### 패키지 import 오류

가상환경이 활성화되었는지 확인하고 `python -m pip install -r requirements.txt`를 저장소 루트에서 다시 실행한다.

### 일부 CIF만 실패

CIF 문법, 원소 기호, 점유율 또는 비정상적인 격자 정보를 확인한다. 현재 스크립트는 한 파일에서 예외가 발생하면 전체 실행이 중단될 수 있으므로 실패 파일을 분리한 뒤 다시 실행한다.

### GPU 메모리 부족

현재 스크립트는 파일을 한 개씩 처리한다. 그래도 메모리가 부족하면 다른 GPU 작업을 종료하거나 CPU 환경에서 실행한다.

## 8. 작업 체크리스트

- [ ] 가상환경 활성화
- [ ] `Element/`에 입력 CIF 배치
- [ ] 기존 `opt_cif/` 결과 백업 여부 확인
- [ ] 저장소 루트의 `chgnet/`에서 `python optimizer.py` 실행
- [ ] 입력/출력 파일 수 비교
- [ ] 최적화 구조 및 계산 로그 검토

## 사용 매뉴얼

### CHGNet은 무엇을 하는가?

CHGNet은 CIF에 기록된 원자와 격자 구조를 입력받아 에너지와 힘을 계산하고, 힘이 작아지는 방향으로 원자 위치와 셀을 완화한다. 이 저장소에서는 여러 CIF를 `Element/`에서 읽어 `opt_cif/`로 일괄 저장한다.

### 처음 실행하기

1. 저장소 루트에서 가상환경을 만들고 패키지를 설치한다.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

2. 최적화할 CIF를 `chgnet/Element/`에 복사한다. 원본 보존을 위해 중요한 CIF는 별도 백업한다.
3. `Element/`와 `opt_cif/`가 다음처럼 보이는지 확인한다.

```text
chgnet/
├─ Element/
│  ├─ 1001.cif
│  └─ 1002.cif
├─ opt_cif/
└─ optimizer.py
```

4. 저장소 루트에서 CHGNet 폴더로 이동해 실행한다.

```powershell
cd chgnet
python optimizer.py
```

5. 터미널에 각 파일의 처리 시작과 완료 메시지가 표시되는지 확인한다.
6. `opt_cif/1001.cif`, `opt_cif/1002.cif`가 생성됐는지 확인한다.

### 실행 전에 반드시 알아둘 점

- 현재 코드는 원본 구조를 바로 최적화하지 않는다. 먼저 각 축에 10% strain을 주고 원자 위치를 `0.005`만큼 교란한다.
- 출력 파일명은 입력과 동일하다. 같은 이름의 결과가 있으면 덮어쓸 수 있다.
- CIF 하나가 잘못되면 전체 배치가 중단될 수 있다. 실패한 파일은 다른 폴더로 옮긴 후 다시 실행한다.
- CHGNet 결과는 계산 예측이다. 중요한 연구 결과에는 DFT 등 별도 방법으로 검증한다.

### 결과가 정상인지 확인하는 방법

1. 입력과 출력 CIF 수를 비교한다.
2. VESTA 또는 pymatgen으로 출력 CIF를 열어 원자 겹침이나 비정상 셀이 없는지 확인한다.
3. 계산 로그에 traceback, isolated atom, 수렴 실패 경고가 없는지 확인한다.
4. 동일한 이름의 파일이 예상한 구조에서 만들어졌는지 확인한다.

```powershell
(Get-ChildItem .\Element -Filter *.cif).Count
(Get-ChildItem .\opt_cif -Filter *.cif).Count
```

### 다음 단계

최적화된 CIF의 물성을 GATGNN으로 예측하려면 저장소 루트로 돌아가 결과를 예측 폴더에 복사한다.

```powershell
cd ..
New-Item -ItemType Directory -Force .\GATGNN\DATA\prediction\my-samples
Copy-Item .\chgnet\opt_cif\*.cif .\GATGNN\DATA\prediction\my-samples\
```

전체 연계 과정은 [루트 초보자 사용 매뉴얼](../README.md#사용-매뉴얼)을 참고한다.
