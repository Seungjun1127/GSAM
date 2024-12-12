# GSAM
Gaussian Semantic Alignment Metric

my-llm-benchmark/
├─ README.md              # 프로젝트 개요, 설치 방법, 사용법, 기여 방법 등
├─ LICENSE                # 오픈소스 라이선스 (예: Apache-2.0, MIT 등)
├─ setup.py               # 패키지 설치 스크립트 (또는 pyproject.toml을 사용할 수도 있음)
├─ pyproject.toml         # Poetry나 Hatch 사용 시 (선택사항)
├─ requirements.txt       # 의존성 목록
├─ .gitignore             # Git에서 제외할 파일 패턴
├─ src/
│  └─ my_llm_benchmark/
│     ├─ __init__.py
│     ├─ data/
│     │  ├─ __init__.py
│     │  ├─ loaders.py        # CSV, JSON 등 데이터 로딩 함수
│     │  └─ preprocess.py     # 전처리 관련 코드
│     ├─ models/
│     │  ├─ __init__.py
│     │  ├─ hf_interface.py   # Hugging Face 모델 로딩, inference 코드
│     │  └─ utils.py          # 모델 관련 유틸 함수
│     ├─ metrics/
│     │  ├─ __init__.py
│     │  ├─ gsam.py           # GSAM 계산 로직, 차원축소, 가우시안 피팅 등
│     │  └─ normality_tests.py # 기타 정규성 검정 코드(Shapiro-Wilk, KS-test 등)
│     ├─ evaluation.py        # 전체 파이프라인 실행: 데이터 → 모델 → 임베딩 → GSAM 산출
│     └─ utils/
│        ├─ __init__.py
│        ├─ logging.py        # 로깅 및 기록 관리
│        └─ plotting.py       # 결과 시각화 함수(옵션)
│
├─ examples/
│  ├─ run_example.ipynb       # Jupyter 노트북 예제 (데이터 로딩부터 GSAM 계산까지)
│  └─ minimal_script.py       # 단순히 CLI로 GSAM 돌려보는 예시 스크립트
│
└─ tests/
   ├─ __init__.py
   ├─ test_data.py            # 데이터 로더 테스트
   ├─ test_gsam.py            # GSAM 로직 테스트
   ├─ test_models.py          # 모델 로딩/추론 테스트
   └─ test_evaluation.py      # 전체 파이프라인 통합 테스트
