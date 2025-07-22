# app/main.py
"""
FastAPI 애플리케이션의 메인 파일.
앱 생성, 미들웨어 설정, 이벤트 핸들러, API 라우트를 정의합니다.
"""
import logging
import time

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator

# 로컬 모듈 임포트
from .config import BASE_MODEL_PATH, MODEL_CONFIGS
from .engine import VLLMMultiLoRAEngine
from .schemas import GenerateRequest, StreamGenerateRequest

# 로거 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- FastAPI 앱 및 vLLM 엔진 초기화 ---
app = FastAPI(title="vLLM Multi-LoRA Server", version="1.0.0")
engine = VLLMMultiLoRAEngine(BASE_MODEL_PATH, MODEL_CONFIGS)

# --- 미들웨어 설정 ---
# 🔐 보안 강화: 명시적 도메인 목록 (다른 서버와 일치)
origins = [
    # EC2 서버 (공인 IP)
    "http://3.13.240.111:3000",  # React Landing Page
    "http://3.13.240.111:8000",  # Backend API
    "http://3.13.240.111:8001",  # DB Module
    "http://3.13.240.111:8002",  # vLLM Server (자기 자신)
    "http://3.13.240.111:8003",  # Translator Service
    
    # 로컬 개발 환경
    "http://localhost:3000",     # React 개발용
    "http://localhost:8000",     # Backend API
    "http://localhost:8001",     # DB Module
    "http://localhost:8002",     # vLLM Server
    "http://localhost:8003",     # Translator Service
    
    # 127.0.0.1 로컬호스트
    "http://127.0.0.1:3000",
    "http://127.0.0.1:8000",
    "http://127.0.0.1:8001",
    "http://127.0.0.1:8002",
    "http://127.0.0.1:8003",
    
    # VSCode Extension 지원
    "vscode://",                 # VSCode Extension
    "vscode-webview://*"         # VSCode WebView
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Prometheus 메트릭 설정
instrumentator = Instrumentator().instrument(app)

# --- 이벤트 핸들러 ---
@app.on_event("startup")
async def startup_event():
    """애플리케이션 시작 시 vLLM 엔진을 초기화하고 Prometheus 엔드포인트를 노출합니다."""
    instrumentator.expose(app)
    try:
        engine.initialize_engine()
        logger.info("✅ FastAPI 서버 시작, /metrics 엔드포인트 활성화")
    except Exception as e:
        logger.error(f"❌ 서버 시작 실패: {e}", exc_info=True)
        # 프로덕션에서는 여기서 애플리케이션을 종료하는 것을 고려할 수 있습니다.
        raise RuntimeError(f"Engine initialization failed: {e}") from e

# --- API 엔드포인트 (라우트) ---
@app.get("/health", tags=["Health"])
async def health_check():
    """서버의 상태를 확인하는 헬스 체크 엔드포인트"""
    if engine.is_initialized:
        return {"status": "healthy", "timestamp": time.time()}
    else:
        raise HTTPException(status_code=503, detail="Service unavailable - model not initialized")

@app.get("/", tags=["Info"])
async def root():
    """서버 정보와 사용 가능한 모델 목록을 반환합니다."""
    return {
        "service": "vLLM Multi-LoRA Server",
        "version": app.version,
        "status": "running" if engine.is_initialized else "initializing",
        "available_models": list(MODEL_CONFIGS.keys()) if engine.is_initialized else []
    }

@app.post("/generate", tags=["Generation"])
async def generate(request_data: GenerateRequest):
    """단일 응답을 생성합니다."""
    try:
        if request_data.model_type not in MODEL_CONFIGS:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid model_type. Available types: {list(MODEL_CONFIGS.keys())}"
            )
        return await engine.generate_single(request_data)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ 생성 중 오류 발생: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/generate/stream", tags=["Generation"])
async def generate_stream(request_data: StreamGenerateRequest):
    """스트리밍 방식으로 응답을 생성합니다."""
    try:
        if request_data.model_type not in MODEL_CONFIGS:
             # 스트리밍 응답에서도 오류를 JSON 형태로 반환할 수 있도록 처리
            async def error_stream():
                error_msg = f"Invalid model_type. Available types: {list(MODEL_CONFIGS.keys())}"
                yield f"data: {json.dumps({'error': error_msg})}\n\n"
            return StreamingResponse(error_stream(), media_type="text/event-stream", status_code=400)

        return StreamingResponse(
            engine.generate_stream(request_data),
            media_type="text/event-stream"
        )
    except Exception as e:
        logger.error(f"❌ 스트리밍 중 오류 발생: {e}", exc_info=True)
        # 스트리밍 응답에서도 오류를 JSON 형태로 반환할 수 있도록 처리
        async def exception_stream():
            yield f"data: {json.dumps({'error': f'Internal server error: {str(e)}'})}\n\n"
        return StreamingResponse(exception_stream(), media_type="text/event-stream", status_code=500)