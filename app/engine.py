# app/engine.py
"""
vLLM 엔진을 초기화하고 텍스트 생성 로직을 처리하는 코어 모듈.
VRAM 최적화, 번역 서비스 연동, 생성 중단 로직이 모두 포함된 최종 버전입니다.
"""
import logging
import time
import json
from typing import Dict, List, Any, AsyncGenerator
import httpx  # API 호출을 위한 httpx

from fastapi import HTTPException
from transformers import AutoTokenizer
from vllm import AsyncLLMEngine, SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.lora.request import LoRARequest

# 로컬 모듈 임포트
from .config import ModelConfig
from .schemas import GenerateRequest, StreamGenerateRequest
# ✨ prompt_builder.py를 사용한다면 아래 주석을 해제하세요.
# from .prompt_builder import generate_enhanced_prompt

logger = logging.getLogger(__name__)

# 번역 서비스의 주소 (docker-compose 서비스 이름 기준)
TRANSLATOR_API_URL = "http://translator:8003/translate"

async def call_translation_service(korean_prompt: str) -> str:
    """번역 서비스를 호출하여 한국어 프롬프트를 영어로 변환합니다."""
    logger.info(f"번역 서비스 호출 시작: '{korean_prompt}'")
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(TRANSLATOR_API_URL, json={"text": korean_prompt})
            response.raise_for_status()
            data = response.json()
            translated_text = data.get('translated_text', korean_prompt)
            logger.info(f"번역 완료: -> '{translated_text}'")
            return translated_text
    except httpx.RequestError as e:
        logger.error(f"번역 서비스({e.request.url}) 호출 실패: {e}. 원본 프롬프트를 사용합니다.")
        return korean_prompt

class VLLMMultiLoRAEngine:
    """vLLM을 사용하여 여러 LoRA 모델을 비동기적으로 처리하는 엔진 클래스"""

    def __init__(self, base_model_path: str, model_configs: Dict[str, ModelConfig]):
        self.base_model_path = base_model_path
        self.model_configs = model_configs
        self.engine = None
        self.tokenizer = None
        self.is_initialized = False

    def initialize_engine(self):
        """vLLM 비동기 엔진과 토크나이저를 초기화합니다."""
        logger.info("🚀 vLLM 비동기 엔진 및 토크나이저 초기화 중 (VRAM 최적화 모드)...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_path)
            
            # ✨ VRAM 최적화 및 성능 개선을 위한 최종 EngineArgs 설정
            engine_args = AsyncEngineArgs(
                model=self.base_model_path,
                quantization="marlin",
                enable_lora=True,
                trust_remote_code=True,
                gpu_memory_utilization=0.4,
                max_model_len=2048,
                max_num_seqs=4,
                tokenizer_mode="slow",
                max_loras=5,
                max_lora_rank=32,
                dtype="half",
                tensor_parallel_size=1,
            )
            
            self.engine = AsyncLLMEngine.from_engine_args(engine_args)
            self.is_initialized = True
            logger.info("✅ 엔진 초기화 완료 (VRAM 최적화 모드)")
        except Exception as e:
            logger.error(f"❌ 엔진 초기화 실패: {e}", exc_info=True)
            self.is_initialized = False
            raise

    async def generate_stream(self, request: StreamGenerateRequest) -> AsyncGenerator[str, None]:
        """스트리밍 방식으로 텍스트를 생성합니다."""
        if not self.is_initialized or self.engine is None or self.tokenizer is None:
            raise HTTPException(status_code=503, detail="Model engine not initialized")

        if request.model_type not in self.model_configs:
            error_msg = f"Model type '{request.model_type}' not configured."
            logger.error(error_msg)
            yield f"data: {json.dumps({'error': error_msg})}\n\n"
            return
            
        english_prompt = await call_translation_service(request.prompt)

        config = self.model_configs[request.model_type]
        
        # ✨ 상세 프롬프트 빌더를 사용하거나, 단순 템플릿을 사용할 수 있습니다.
        # 여기서는 단순 템플릿을 사용합니다.
        messages = [{"role": "system", "content": config.system_prompt}, {"role": "user", "content": english_prompt}]
        final_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        sampling_params = SamplingParams(
            temperature=request.temperature if request.temperature is not None else config.temperature,
            top_p=request.top_p if request.top_p is not None else config.top_p,
            max_tokens=request.max_tokens,
            stop=config.stop or [],
            repetition_penalty=1.15,
            include_stop_str_in_output=False  # stop 토큰을 출력에서 제외하여 FIM 태그 인식 개선
        )
        
        # FIM 태그 인식을 위한 로깅
        logger.info(f"Stop tokens for {request.model_type}: {config.stop}")

        lora_request = LoRARequest(lora_name=config.name, lora_int_id=config.lora_id, lora_path=config.adapter_path)
        request_id = f"stream-{int(time.time() * 1000)}"
        results_generator = self.engine.generate(final_prompt, sampling_params, request_id, lora_request)

        # ✨ 생성 중단(Stop Token) 오류를 해결한 최종 스트리밍 로직
        last_text = ""
        async for request_output in results_generator:
            text_so_far = request_output.outputs[0].text
            
            stop_found = False
            stop_str_found = ""
            for stop_str in (config.stop or []):
                if stop_str in text_so_far:
                    # Stop Token 이전까지만 텍스트를 자름
                    text_so_far = text_so_far.split(stop_str)[0]
                    stop_found = True
                    stop_str_found = stop_str
                    break
            
            delta = text_so_far[len(last_text):]
            if delta:
                yield f"data: {json.dumps({'text': delta})}\n\n"
            
            last_text = text_so_far

            if stop_found:
                logger.info(f"Stop token '{stop_str_found}' 감지됨. 스트림을 조기 종료합니다.")
                break

        yield "data: [DONE]\n\n"

    async def generate_single(self, request: GenerateRequest) -> Dict[str, Any]:
        """단일 응답으로 텍스트를 생성합니다."""
        config = self.model_configs[request.model_type]
        stream_request = StreamGenerateRequest(
            user_id=0,
            prompt=request.prompt,
            model_type=request.model_type,
            user_select_options=request.user_select_options,
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_tokens if request.max_tokens is not None else config.max_tokens,
        )

        full_text = ""
        async for chunk in self.generate_stream(stream_request):
            if chunk.strip() == "data: [DONE]":
                break
            try:
                data_str = chunk.strip()[5:].strip()
                if data_str:
                    data = json.loads(data_str)
                    if "text" in data:
                        full_text += data["text"]
                    elif "error" in data:
                        logger.error(f"스트림에서 오류 수신: {data['error']}")
                        return {'error': data['error']}
            except json.JSONDecodeError:
                logger.warning(f"스트림에서 JSON 디코딩 오류 발생: {chunk}")
                continue
        return {'generated_text': full_text}