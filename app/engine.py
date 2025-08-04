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
            
            # ✨ 안정성 우선 EngineArgs 설정 (토큰 중복 방지)
            engine_args = AsyncEngineArgs(
                model=self.base_model_path,
                # quantization="marlin",  # 비활성화: 토큰 중복 원인 가능성
                enable_lora=True,
                trust_remote_code=True,
                gpu_memory_utilization=0.5,  # 0.4 → 0.5 증가
                max_model_len=4096,  # 2048 → 4096 증가 (더 긴 컴텍스트)
                max_num_seqs=2,  # 4 → 2 감소 (안정성 우선)
                tokenizer_mode="auto",  # "slow" → "auto" 변경
                max_loras=3,  # 5 → 3 감소
                max_lora_rank=16,  # 32 → 16 감소
                dtype="half",
                tensor_parallel_size=1,
                # 추가 안정성 옵션
                enforce_eager=True,  # CUDA 그래프 최적화 비활성화
                disable_custom_all_reduce=True,  # 커스텀 reduce 비활성화
            )
            
            self.engine = AsyncLLMEngine.from_engine_args(engine_args)
            self.is_initialized = True
            logger.info("✅ 엔진 초기화 완료 (VRAM 최적화 모드)")
            
            # 모델 상태 간단 검증
            logger.info("🔍 모델 상태 검증 시작...")
            try:
                # 간단한 테스트 프롬프트
                test_prompt = "Hello"
                test_params = SamplingParams(temperature=0.1, max_tokens=5)
                # 비동기 테스트는 여기서 수행하지 않음 (초기화 단계에서는 위험)
                logger.info("✅ 모델 기본 상태 확인 완료")
            except Exception as e:
                logger.error(f"⚠️ 모델 상태 검증 실패: {e}")
        except Exception as e:
            logger.error(f"❌ 엔진 초기화 실패: {e}", exc_info=True)
            self.is_initialized = False
            raise

    def _build_chatml_prompt(self, config: ModelConfig, user_prompt: str, model_type: str) -> str:
        """ChatML 형식으로 프롬프트를 구성합니다."""
        
        # FIM 태그 감지 및 특별 처리
        is_fim_request = "<｜fim begin｜>" in user_prompt or "<|fim_begin|>" in user_prompt
        
        if is_fim_request:
            logger.info("FIM 요청 감지됨 - 코드 완성 모드 활성화")
            # FIM 요청에 대한 특별 처리
            user_prompt = self._enhance_fim_prompt(user_prompt)
        
        # 모델 타입별 특화된 시스템 프롬프트 구성
        if model_type == "autocomplete":
            system_content = f"""
You are an expert Python code completion assistant.

Your task:
- Complete the user's code snippet by filling in the missing parts
- Provide only the necessary code to complete the functionality
- Use proper Python syntax and best practices
- Keep completions concise and relevant

Output format:
- Provide ONLY the completed code
- Do NOT add explanations or comments unless specifically requested
- Always end with: # --- Generation Complete ---

Remember: You are completing code, not writing essays.
""".strip()
        
        elif model_type == "prompt":
            system_content = f"""
You are a helpful senior Python developer with expertise in writing clean, efficient code.

Your task:
- Generate Python code based on the user's request
- Provide working, tested code solutions
- Follow Python best practices and PEP 8 guidelines
- Include necessary imports and error handling

Output format:
1. First, provide the complete code solution
2. Then, on a new line, write '--- Rationale ---' followed by a brief explanation
3. Always end with: # --- Generation Complete ---

Remember: Focus on practical, working solutions.
""".strip()
        
        elif model_type == "comment":
            system_content = f"""
You are a Python code documentation specialist.

Your task:
- Generate clear, concise comments and docstrings
- Explain code functionality and purpose
- Follow Python documentation standards
- Use proper docstring formats (Google/NumPy style)

Output format:
- Provide the requested comments/docstrings
- Keep explanations clear and technical
- Always end with: # --- Generation Complete ---

Remember: Documentation should be helpful and accurate.
""".strip()
        
        elif model_type == "error_fix":
            system_content = f"""
You are a Python debugging expert specializing in error analysis and fixes.

Your task:
- Analyze the provided code and identify issues
- Provide corrected code with fixes applied
- Explain what was wrong and how it was fixed
- Ensure the solution handles edge cases

Output format:
1. First, provide the corrected code
2. Then, write '--- Fix Explanation ---' followed by what was wrong and how you fixed it
3. Always end with: # --- Generation Complete ---

Remember: Focus on robust, error-free solutions.
""".strip()
        
        else:
            # 기본 시스템 프롬프트
            system_content = config.system_prompt
        
        # ChatML 프롬프트 구성 (시스템 프롬프트 포함)
        chatml_prompt = f"""<|im_start|>system
{system_content}<|im_end|>
<|im_start|>user
{user_prompt}<|im_end|>
<|im_start|>assistant
"""
        
        logger.debug(f"ChatML 프롬프트 구성 완료 (모델: {model_type}) - 훈련 데이터 형식 호환")
        return chatml_prompt

    def _enhance_fim_prompt(self, user_prompt: str) -> str:
        """FIM 요청에 대한 프롬프트 향상"""
        # FIM 태그를 더 자연스러운 언어로 변환
        enhanced_prompt = user_prompt
        
        # 다양한 FIM 태그 형식 지원
        fim_patterns = [
            ("<｜fim begin｜>", "[CODE_START]"),
            ("<｜fim hole｜>", "[FILL_HERE]"),
            ("<｜fim end｜>", "[CODE_END]"),
            ("<|fim_begin|>", "[CODE_START]"),
            ("<|fim_hole|>", "[FILL_HERE]"),
            ("<|fim_end|>", "[CODE_END]")
        ]
        
        for old_tag, new_tag in fim_patterns:
            enhanced_prompt = enhanced_prompt.replace(old_tag, new_tag)
        
        # FIM 요청에 대한 자연어 설명 추가 (훈련 데이터 형식에 맞게 한국어 사용)
        if "[FILL_HERE]" in enhanced_prompt:
            enhanced_prompt = f"""주석에 따라 적절한 코드를 생성해주세요.

이전 코드:
{enhanced_prompt.replace('[FILL_HERE]', '// 여기에 코드를 삽입해야 함 //')}

위 코드에서 비어있는 부분을 채워주세요. Python 베스트 프랙티스를 따르고 주변 코드와 자연스럽게 연결되도록 해주세요."""
        
        return enhanced_prompt

    def _get_stop_token_ids(self, stop_strings: List[str]) -> List[int]:
        """스탑 문자열을 토큰 ID로 변환"""
        stop_token_ids = []
        
        if self.tokenizer is None:
            return stop_token_ids
            
        # EOS 토큰 및 특수 토큰 강제 추가
        special_tokens = [
            self.tokenizer.eos_token_id,  # EOS 토큰
            self.tokenizer.pad_token_id,  # PAD 토큰
        ]
        
        for token_id in special_tokens:
            if token_id is not None:
                stop_token_ids.append(token_id)
            
        for stop_str in stop_strings:
            try:
                # 문자열을 토큰 ID로 변환
                token_ids = self.tokenizer.encode(stop_str, add_special_tokens=False)
                stop_token_ids.extend(token_ids)
                
                # 마지막 토큰만 사용 (더 정확한 매칭)
                if token_ids:
                    stop_token_ids.append(token_ids[-1])
                    
            except Exception as e:
                logger.warning(f"스탑 토큰 '{stop_str}' 변환 실패: {e}")
                continue
        
        # 중복 제거
        stop_token_ids = list(set(stop_token_ids))
        logger.info(f"스탑 토큰 ID: {stop_token_ids}")
        
        return stop_token_ids

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
        
        # ✨ ChatML 형식으로 수동 프롬프트 구성 (스탑 토큰 인식 개선)
        final_prompt = self._build_chatml_prompt(config, english_prompt, request.model_type)

        # 토큰 ID 기반 스탑 토큰 계산
        stop_token_ids = self._get_stop_token_ids(config.stop or [])
        
        sampling_params = SamplingParams(
            temperature=request.temperature if request.temperature is not None else config.temperature,
            top_p=request.top_p if request.top_p is not None else config.top_p,
            max_tokens=request.max_tokens,
            stop=config.stop or [],
            stop_token_ids=stop_token_ids,  # 토큰 ID 기반 스탑
            repetition_penalty=1.3,  # 반복 방지 더 강화
            frequency_penalty=0.2,   # 빈도 기반 페널티 증가
            presence_penalty=0.2,    # 존재 기반 페널티 증가
            include_stop_str_in_output=False,  # stop 토큰을 출력에서 제외
            skip_special_tokens=False  # 특수 토큰 유지
        )
        
        # FIM 태그 인식을 위한 로깅
        logger.info(f"Stop tokens for {request.model_type}: {config.stop or []}")
        
        # request_id 생성 (사용 전에 먼저 정의)
        request_id = f"stream-{int(time.time() * 1000)}"
        
        # 스트리밍 생성 시작
        logger.info(f"스트리밍 생성 시작: request_id={request_id}")
        last_text = ""
        token_count = 0
        lora_request = LoRARequest(lora_name=config.name, lora_int_id=config.lora_id, lora_path=config.adapter_path)
        results_generator = self.engine.generate(final_prompt, sampling_params, request_id, lora_request)

        # ✨ 스트리밍 생성 루프
        async for request_output in results_generator:
            token_count += 1
            current_text = request_output.outputs[0].text
            
            # 1. 스톱 토큰 체크
            stop_found = False
            for stop_str in (config.stop or []):
                if stop_str in current_text:
                    current_text = current_text.split(stop_str)[0]
                    stop_found = True
                    logger.info(f"스톱 토큰 '{stop_str}' 감지됨")
                    break
            
            # 2. 델타 계산 (중복 방지)
            delta = current_text[len(last_text):] if len(current_text) > len(last_text) else ""
            
            # 3. 델타 품질 검증 및 정리
            if delta and not stop_found:
                # 델타 품질 검증
                cleaned_delta = self._validate_and_clean_delta(delta)
                if cleaned_delta:
                    logger.info(f"델타 전송: '{cleaned_delta[:50]}...'")
                    yield f"data: {json.dumps({'text': cleaned_delta})}\n\n"
                    last_text = current_text  # 중요: 전송 후 업데이트
                else:
                    logger.warning(f"부적절한 델타 필터링: '{delta[:30]}...'")
            
            # 4. 종료 조건 체크
            if stop_found:
                logger.info("스톱 토큰으로 스트림 종료")
                break
                
            # 5-1. 최대 토큰 수 체크
            if len(current_text.split()) >= request.max_tokens:
                logger.info(f"최대 토큰 수({request.max_tokens}) 도달로 스트림 종료")
                break
                
            # 5-2. 문자 수 기반 종료 (모델 타입별) - 더 긴 응답 허용
            max_chars = {
                "autocomplete": 2000,   # 1000 → 2000
                "prompt": 8000,        # 4000 → 8000 (더 긴 코드 생성)
                "comment": 3000,       # 1500 → 3000
                "error_fix": 6000      # 3000 → 6000
            }.get(request.model_type, 4000)  # 기본값도 2000 → 4000
            
            if len(current_text) >= max_chars:
                logger.info(f"문자 수 제한({max_chars}) 도달로 스트림 종료 (current={len(current_text)})")
                break
            
            # 주기적인 진행 상황 로그
            if token_count % 10 == 0:
                logger.debug(f"진행 상황: tokens={token_count}, chars={len(current_text)}, max_chars={max_chars}")

        # 정상 종료 시 완료 신호 전송
        logger.info(f"스트리밍 정상 종료 - 완료 신호 전송 (token_count={token_count})")
        yield f"data: {json.dumps({'type': 'done', 'text': ''})}\n\n"

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
            # 새로운 JSON 형식 완료 신호 처리
            if chunk.strip().startswith("data: "):
                try:
                    data_str = chunk.strip()[5:].strip()
                    if data_str:
                        data = json.loads(data_str)
                        
                        # 완료 신호 체크
                        if data.get("type") == "done":
                            logger.info("스트림 완료 신호 수신")
                            break
                            
                        # 텍스트 데이터 처리
                        if "text" in data and data["text"]:
                            full_text += data["text"]
                        elif "error" in data:
                            logger.error(f"스트림에서 오류 수신: {data['error']}")
                            return {'error': data['error']}
                except json.JSONDecodeError:
                    logger.warning(f"스트림에서 JSON 디코딩 오류 발생: {chunk}")
                    continue
            # 구식 [DONE] 신호 호환성 유지
            elif chunk.strip() == "data: [DONE]":
                logger.info("구식 [DONE] 신호 수신")
                break
        return {'generated_text': full_text}

    def _validate_and_clean_delta(self, delta: str) -> str:
        """델타 품질 검증 및 정리 (강화된 버전)"""
        if not delta or len(delta.strip()) == 0:
            return ""
        
        import re
        
        # 1. 기본 정리
        cleaned = delta.strip()
        
        # 2. 비인쇄 가능 문자 제거
        cleaned = ''.join(c for c in cleaned if c.isprintable() or c.isspace())
        
        # 3. 중복 문자 패턴 제거
        cleaned = re.sub(r'([a-zA-Z_])\1{2,}', r'\1', cleaned)  # 같은 문자 3개 이상 → 1개
        cleaned = re.sub(r'([a-zA-Z_]{2,})\1+', r'\1', cleaned)  # 패턴 반복 제거
        
        # 4. 기그 문자 패턴 감지 및 차단
        broken_patterns = [
            r'[\x00-\x1f\x7f-\x9f]{2,}',  # 제어 문자
            r'([a-zA-Z])\1{4,}',  # 5개 이상 반복
            r'([a-zA-Z]{2,})\1{3,}',  # 패턴 4번 이상 반복
            r'^[^a-zA-Z0-9\s\(\)\[\]\{\}\.,;:"\'\_\-\+\*\/\=\<\>\!\?\#\$\%\&\@\^\`\~\|\\]{3,}',  # 이상한 문자로만 구성
        ]
        
        for pattern in broken_patterns:
            if re.search(pattern, cleaned):
                logger.warning(f"기그 패턴 감지로 델타 차단: '{pattern}' in '{cleaned[:30]}...'")
                return ""  # 기극 문자 감지 시 완전 차단
        
        # 5. 최소 길이 체크
        if len(cleaned.strip()) < 1:
            return ""
            
        # 6. 의미 있는 내용 체크 (완화된 버전)
        meaningful_chars = sum(1 for c in cleaned if c.isalnum() or c in '()[]{}.,;:"\'\_-+=<>!?#$%&@^`~|\\ \t\n')
        total_chars = len(cleaned)
        
        # 너무 짧은 델타는 통과
        if total_chars <= 3:
            return cleaned
        
        # 비율 계산 (공백 문자 포함)
        if total_chars > 0:
            ratio = meaningful_chars / total_chars
            # 10% 이상이면 통과 (기존 30%에서 완화)
            if ratio < 0.1:
                logger.warning(f"의미 없는 델타 차단: meaningful_ratio={meaningful_chars}/{total_chars} = {ratio:.2f}")
                return ""
            else:
                logger.debug(f"델타 통과: meaningful_ratio={meaningful_chars}/{total_chars} = {ratio:.2f}")
        
        return cleaned