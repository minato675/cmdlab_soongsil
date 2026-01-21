import os
import numpy as np
from pymatgen.core import Structure
from chgnet.model import CHGNet
from chgnet.model import StructOptimizer

# 폴더 경로 설정 ; 파일이 있는 경로로 설정
folder_path = "Element/"

# 기본적인 코드 세팅

# CHGNet 초기화
chgnet = CHGNet.load()

# 최적화기 생성
relaxer = StructOptimizer(optimizer_class="BFGS")

# 구조최적화 모델 선택; 사용가능한 모델 [ 'FIRE', 'BFGS', 'LBFGS', 'LBFGSLineSearch', 'MDMin', 'SciPyFminCG', 'SciPyFminBFGS', 'BFGSLineSearch' ]

# 구조를 최적화하고 저장하는 함수
def optimize_structure(file_path, output_folder):
    # CIF 파일에서 구조 불러오기
    structure = Structure.from_file(file_path)
    
    # 최적화를 위한 구조 복사
    unrelaxed_structure = structure.copy()
    
    # 필요에 따라 변형이나 섭동 적용
    unrelaxed_structure.apply_strain([0.1, 0.1, 0.1])
    unrelaxed_structure.perturb(0.005)
    
    # 최적화 수행
    result = relaxer.relax(unrelaxed_structure, verbose=True)
    
    # 최종 최적화된 구조 추출
    optimized_structure = result["final_structure"]
    
    # 출력 CIF 파일 경로 생성
    output_file = os.path.join(output_folder, os.path.basename(file_path))
    
    # 최적화된 구조를 CIF 파일로 저장
    optimized_structure.to(fmt="cif", filename=output_file)
    
    print(f"최적화된 CIF 파일이 저장되었습니다: {output_file}")
    
output_folder = "opt_cif/"  # 출력 폴더 경로 수정하세요; 출력을 원하는 경로 설정

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for file_name in os.listdir(folder_path):
    if file_name.endswith(".cif"):
        file_path = os.path.join(folder_path, file_name)
        print(f"파일 처리 중: {file_name}")
        optimize_structure(file_path, output_folder)
        print(f"파일 처리 완료: {file_name}\n")
        
