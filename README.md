# 🚀 MLOps CI/CD Project

[![CI Pipeline](https://github.com/AI-sunwoo/my_project/actions/workflows/ci.yml/badge.svg)](https://github.com/AI-sunwoo/my_project/actions/workflows/ci.yml)
[![Model Training](https://github.com/AI-sunwoo/my_project/actions/workflows/train.yml/badge.svg)](https://github.com/AI-sunwoo/my_project/actions/workflows/train.yml)

MNIST 손글씨 숫자 분류를 통한 **End-to-End MLOps 파이프라인** 프로젝트입니다.

---

## 📁 프로젝트 구조

```
my_project/
├── .github/
│   └── workflows/
│       ├── ci.yml          # CI 파이프라인 (테스트, 린팅)
│       ├── train.yml       # 모델 학습 파이프라인
│       └── deploy.yml      # 배포 파이프라인
├── src/
│   ├── model.py           # CNN 모델 정의
│   ├── train.py           # 학습 스크립트 + MLflow
│   └── evaluate.py        # 모델 평가 스크립트
├── api/
│   └── main.py            # FastAPI 서빙 서버
├── tests/
│   └── test_mlops.py      # 유닛 테스트
├── data/                   # 데이터셋 (자동 다운로드)
├── models/                 # 학습된 모델 저장소
├── Dockerfile             # 컨테이너화
├── requirements.txt       # Python 의존성
└── README.md
```

---

## 🔄 MLOps 파이프라인

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        GitHub Actions CI/CD                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   📝 Code Push     🧪 CI Pipeline      🏋️ Train Pipeline    🚀 Deploy   │
│   ───────────────────────────────────────────────────────────────────   │
│                                                                         │
│   main/develop  →  Lint (Black)    →   Model Training  →  Docker Build │
│   Pull Request  →  Test (Pytest)   →   MLflow Tracking →  Docker Push  │
│                 →  Docker Build    →   Model Evaluation→  Deploy       │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 파이프라인 설명

| 파이프라인 | 트리거 | 설명 |
|-----------|--------|------|
| **CI** | Push, PR | 코드 품질 검사, 테스트, Docker 빌드 |
| **Train** | 수동, 매주 월요일 | 모델 학습, MLflow 추적, 평가 |
| **Deploy** | 태그 (v*) | Docker 이미지 빌드 & 푸시, 배포 |

---

## 🚀 빠른 시작

### 1. 로컬 환경 설정

```bash
# 저장소 클론
git clone https://github.com/AI-sunwoo/my_project.git
cd my_project

# 가상환경 생성 및 활성화
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt
```

### 2. 모델 학습

```bash
cd src
python train.py --epochs 5 --batch-size 64
```

### 3. MLflow UI 확인

```bash
cd src
mlflow ui --port 5000
# http://localhost:5000 접속
```

### 4. API 서버 실행

```bash
cd api
uvicorn main:app --reload --port 8000
# http://localhost:8000/docs 에서 Swagger UI 확인
```

### 5. 테스트 실행

```bash
pytest tests/ -v --cov=src --cov=api
```

---

## 🐳 Docker 사용

### 로컬 빌드 & 실행

```bash
# 이미지 빌드
docker build -t mlops-mnist .

# 컨테이너 실행
docker run -p 8000:8000 mlops-mnist

# 헬스체크
curl http://localhost:8000/health
```

---

## ⚙️ GitHub Actions 설정

### 필요한 Secrets

GitHub 저장소 Settings > Secrets and variables > Actions에서 설정:

| Secret | 설명 |
|--------|------|
| `DOCKER_USERNAME` | Docker Hub 사용자명 |
| `DOCKER_PASSWORD` | Docker Hub 액세스 토큰 |

### 수동 학습 실행

1. GitHub 저장소 > Actions 탭 이동
2. "Model Training Pipeline" 선택
3. "Run workflow" 클릭
4. 하이퍼파라미터 입력 후 실행

---

## 📊 API 엔드포인트

| 엔드포인트 | 메서드 | 설명 |
|-----------|--------|------|
| `/` | GET | API 정보 |
| `/health` | GET | 서버 상태 확인 |
| `/model/info` | GET | 모델 메타데이터 |
| `/predict` | POST | 단일 이미지 예측 |
| `/predict/batch` | POST | 배치 이미지 예측 |
| `/docs` | GET | Swagger UI |

### 예측 API 사용 예시

```bash
# curl
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@digit.png"

# 응답
{
  "predicted_digit": 7,
  "confidence": 0.9823,
  "all_probabilities": {"0": 0.001, "1": 0.002, ..., "7": 0.982}
}
```

---

## 🛠 기술 스택

| 영역 | 기술 |
|------|------|
| ML Framework | PyTorch |
| 실험 추적 | MLflow |
| API 서빙 | FastAPI + Uvicorn |
| CI/CD | GitHub Actions |
| 컨테이너 | Docker |
| 테스트 | Pytest |
| 코드 품질 | Black, isort, flake8 |

---

## 📈 확장 가이드

이 프로젝트를 더 발전시키려면:

1. **클라우드 배포**: AWS ECS, GCP Cloud Run, Azure Container Apps
2. **모니터링**: Prometheus + Grafana
3. **데이터 버전 관리**: DVC (Data Version Control)
4. **Feature Store**: Feast
5. **모델 레지스트리**: MLflow Model Registry
6. **A/B 테스트**: 여러 모델 버전 동시 서빙

---

## 📝 라이선스

MIT License
