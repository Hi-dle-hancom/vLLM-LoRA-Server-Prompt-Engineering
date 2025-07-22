# vLLM LoRA Server - Prompt Engineering

고성능 vLLM 기반 LoRA 어댑터 서빙 시스템으로, 프롬프트 엔지니어링을 위한 완전한 솔루션을 제공합니다.

## 🚀 주요 특징

- **vLLM 엔진**: 고성능 대규모 언어 모델 추론
- **LoRA 어댑터 지원**: 동적 어댑터 로딩 및 관리
- **프롬프트 빌더**: 다양한 프롬프트 템플릿 지원
- **Docker 컨테이너화**: 완전한 마이크로서비스 아키텍처
- **모니터링**: Prometheus + Grafana 통합
- **데이터베이스**: PostgreSQL 지원
- **RESTful API**: FastAPI 기반 고성능 API

## 📁 프로젝트 구조

```
vllm-lora-server/
├── app/                    # 메인 애플리케이션
│   ├── main.py            # FastAPI 애플리케이션 엔트리포인트
│   ├── engine.py          # vLLM 엔진 관리
│   ├── prompt_builder.py  # 프롬프트 템플릿 빌더
│   ├── schemas.py         # Pydantic 스키마 정의
│   └── config.py          # 설정 관리
├── postgres_init/         # PostgreSQL 초기화 스크립트
├── grafana/              # Grafana 설정
├── docker-compose.yml    # Docker Compose 설정
├── Dockerfile           # vLLM 서버 Docker 이미지
├── requirements.txt     # Python 의존성
└── prometheus.yml       # Prometheus 설정
```

## 🛠️ 기술 스택

- **Backend**: FastAPI, vLLM, PyTorch
- **Database**: PostgreSQL
- **Monitoring**: Prometheus, Grafana
- **Containerization**: Docker, Docker Compose
- **AI/ML**: Transformers, LoRA Adapters

## 🔧 설치 및 실행

### 1. 환경 요구사항
- Docker & Docker Compose
- NVIDIA GPU (CUDA 지원)
- 8GB+ GPU 메모리 권장

### 2. 실행
```bash
# 저장소 클론
git clone git@github.com:Hi-dle-hancom/vLLM-LoRA-Server-Prompt-Engineering.git
cd vLLM-LoRA-Server-Prompt-Engineering

# Docker Compose로 전체 스택 실행
docker-compose up -d
```

### 3. 서비스 접속
- **vLLM API**: http://localhost:8002
- **Grafana**: http://localhost:3000
- **Prometheus**: http://localhost:9090

## 📡 API 엔드포인트

### 기본 추론
```bash
POST /v1/completions
Content-Type: application/json

{
  "prompt": "def fibonacci(n):",
  "max_tokens": 100,
  "temperature": 0.7
}
```

### LoRA 어댑터 사용
```bash
POST /v1/completions
Content-Type: application/json

{
  "prompt": "코드를 설명해주세요:",
  "lora_adapter": "code-explanation",
  "max_tokens": 200
}
```

## 🔍 모니터링

- **Prometheus**: 메트릭 수집 및 저장
- **Grafana**: 시각화 대시보드
- **로그**: Docker 컨테이너 로그 통합

## 🏗️ 아키텍처

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Client App    │───▶│  vLLM Server    │───▶│  LoRA Adapters  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                              │
                              ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Grafana       │◀───│  Prometheus     │◀───│  PostgreSQL     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🚀 개발 및 배포

### 개발 모드
```bash
# 개발용 실행 (볼륨 마운트)
docker-compose -f docker-compose.yml up --build
```

### 프로덕션 배포
```bash
# 프로덕션 모드
docker-compose -f docker-compose.yml up -d --build
```

## 📝 라이선스

이 프로젝트는 HancomAI 아카데미의 내부 프로젝트입니다.

**Hancom AI Team** - vLLM LoRA Server Prompt Engineering Solution