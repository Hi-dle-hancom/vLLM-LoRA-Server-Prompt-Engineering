# 베이스 이미지
FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# 시스템 패키지 및 Python 설치
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget curl git ca-certificates build-essential \
    libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev \
    libffi-dev xz-utils liblzma-dev \
    && wget https://www.python.org/ftp/python/3.12.3/Python-3.12.3.tgz \
    && tar xzf Python-3.12.3.tgz \
    && cd Python-3.12.3 \
    && ./configure --enable-optimizations \
    && make -j$(nproc) \
    && make altinstall \
    && ln -s /usr/local/bin/python3.12 /usr/bin/python \
    && ln -s /usr/local/bin/pip3.12 /usr/bin/pip \
    && cd .. \
    && rm -rf Python-3.12.3 Python-3.12.3.tgz \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# 작업 디렉토리 설정
WORKDIR /app

# 의존성 파일 복사 및 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu122

# 소스 코드 복사
COPY ./app ./app

# 헬스체크 및 포트 노출
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8002/health || exit 1
EXPOSE 8002

# 애플리케이션 실행
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8002"]