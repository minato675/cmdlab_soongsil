# 재료 구조 최적화 및 물성 예측 워크스페이스

[한국어](README.md) | [English](README_EN.md)

이 저장소는 CIF 결정 구조를 CHGNet으로 최적화하고, GATGNN으로 재료 물성을 학습·평가·예측하기 위한 통합 작업 환경이다.

```text
원본 CIF
  → CHGNet 구조 최적화
  → 최적화 CIF
  → GATGNN 물성 예측
  → 예측 CSV
```

## 프로젝트 구성

| 경로 | 역할 | 상세 문서 |
| --- | --- | --- |
| `chgnet/` | CHGNet 기반 CIF 구조 최적화 | [한국어](chgnet/README.md) · [English](chgnet/README_EN.md) |
| `GATGNN/` | GATGNN 모델 학습, 평가 및 물성 예측 | [한국어](GATGNN/README.md) · [English](GATGNN/README_EN.md) |
| `requirements.txt` | CHGNet·GATGNN 통합 최소 실행 환경 | — |

## 전체 작업 흐름

### 1. CHGNet 구조 최적화

최적화할 CIF 파일을 `chgnet/Element/`에 넣고 실행한다.

```powershell
cd C:\work\chgnet
python optimizer.py
```

최적화된 CIF는 같은 파일명으로 `chgnet/opt_cif/`에 저장된다.

현재 스크립트는 원본 구조에 10% strain과 `0.005` 원자 위치 perturbation을 적용한 뒤, 사전 학습된 CHGNet 모델과 BFGS 최적화기로 구조를 완화한다.

### 2. GATGNN 예측 입력 준비

최적화된 CIF 중 물성을 예측할 파일을 `GATGNN/DATA/prediction/<data_src>/`에 복사한다.

```powershell
Copy-Item C:\work\chgnet\opt_cif\*.cif C:\work\GATGNN\DATA\prediction\prediction-directory\
```

원본 CIF를 그대로 예측하려는 경우에는 원하는 CIF를 해당 폴더에 직접 넣어도 된다.

### 3. GATGNN 물성 예측

학습된 모델을 선택해 예측을 실행한다.

```powershell
cd C:\work\GATGNN
python predict.py --property density --data_src prediction-directory
```

결과는 `GATGNN/PREDICTIONS/`에 CSV로 저장된다.

다른 물성 예시:

```powershell
python predict.py --property thermal-conductivity --data_src CMD --to_predict DATA\prediction-directory
python predict.py --property poisson-ratio --data_src CMD --to_predict DATA\prediction-directory
python predict.py --property new_bulk-modulus --data_src CMD --to_predict DATA\prediction-directory
python predict.py --property new_Youngs-modulus --data_src CMD --to_predict DATA\prediction-directory
```

예측에는 `TRAINED/<property>.pt` 모델이 필요하며, 학습 당시의 데이터 소스와 모델 구조 옵션을 동일하게 사용해야 한다.

### 4. 필요 시 CIF 부피 계산

```powershell
cd C:\work\GATGNN
python volume_predict.py --to_predict DATA\prediction-directory
```

결과는 기본적으로 `GATGNN/PREDICTIONS/volume_prediction-directory.csv`에 저장된다.

## GATGNN 모델 학습과 평가

새 모델을 학습하려면 대응하는 CIF와 물성 CSV를 먼저 준비한다.

```powershell
cd C:\work\GATGNN
python train.py --property density --data_src CMD
python evaluate.py --property density --data_src CMD
```

- 물성 참조값: `GATGNN/DATA/properties-reference/<property>.csv`
- 학습 CIF: `GATGNN/DATA/train&evaluate/<data_src>/<ID>.cif`
- 학습 모델: `GATGNN/TRAINED/<property>.pt`
- 평가 결과: `GATGNN/RESULTS/<property>_results.csv`

데이터 형식과 지원 물성명은 [GATGNN 상세 매뉴얼](GATGNN/README.md)을 참고한다.

## 실행 환경

Python 3.10~3.12 환경에서 루트 `requirements.txt`로 두 프로젝트의 검증된 최소 의존성을 설치한다.

```powershell
cd C:\work
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

기본 설치는 CPU용 PyTorch를 선택할 수 있다. NVIDIA GPU를 사용하려면 PyTorch 공식 설치 방식으로 CUDA 호환 빌드를 설치한 뒤 나머지 요구사항을 설치한다.

각 스크립트는 상대 경로를 사용한다. 실행 전에 반드시 해당 프로젝트 폴더로 이동한다.

## Git 관리 정책

다음 대용량 입력·출력은 Git에서 제외된다.

- `chgnet/Element/`
- `chgnet/opt_cif/`
- `GATGNN/DATA/train&evaluate/`와 `GATGNN/DATA/prediction/`의 실제 데이터
- `GATGNN/PREDICTIONS/`의 생성 결과
- `GATGNN/RESULTS/`의 생성 결과

학습·평가 및 예측의 각 data source 폴더와 `GATGNN/PREDICTIONS`, `GATGNN/RESULTS`는 `.gitkeep`만 추적하여 빈 폴더 구조를 유지한다.

## 빠른 체크리스트

- [ ] 사용할 Python 가상환경 활성화
- [ ] `chgnet/Element/`에 원본 CIF 배치
- [ ] `C:\work\chgnet`에서 구조 최적화 실행
- [ ] 최적화 결과를 GATGNN 예측 폴더로 복사
- [ ] 필요한 `TRAINED/<property>.pt` 모델 확인
- [ ] `C:\work\GATGNN`에서 물성 예측 실행
- [ ] `PREDICTIONS/`의 결과 CSV 검토

## 상세 문서

- [CHGNet 구조 최적화 매뉴얼](chgnet/README.md)
- [GATGNN 학습·평가·예측 매뉴얼](GATGNN/README.md)
