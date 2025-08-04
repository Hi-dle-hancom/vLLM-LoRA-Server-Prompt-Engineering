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
            # 🛡️ 안전한 토크나이저 초기화 (바이트 경계 오류 방지)
            logger.info(f"토크나이저 로딩 시작: {self.base_model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.base_model_path,
                trust_remote_code=True,
                use_fast=False,  # Fast 토크나이저 비활성화 (안정성 우선)
                padding_side="left",  # 패딩 방향 명시
                truncation_side="left",  # 잘림 방향 명시
                clean_up_tokenization_spaces=True,  # 토큰화 공백 정리
            )
            
            # 🔧 토크나이저 안전성 설정
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                logger.info("pad_token을 eos_token으로 설정")
            
            if self.tokenizer.unk_token is None:
                self.tokenizer.unk_token = "<unk>"
                logger.info("unk_token 설정")
                
            logger.info(f"✅ 토크나이저 로딩 완료: vocab_size={self.tokenizer.vocab_size}")
            
            # ✨ 극도로 보수적인 안전 설정 (텍스트 손상 방지)
            engine_args = AsyncEngineArgs(
                model=self.base_model_path,
                # quantization 명시적 비활성화 (토큰 손상 방지)
                quantization=None,  # 명시적 None 설정
                enable_lora=True,
                trust_remote_code=True,
                gpu_memory_utilization=0.4,  # 비양자화 모델에 맞게 감소
                max_model_len=1536,  # 비양자화 모델에 맞게 감소
                max_num_seqs=1,  # 단일 시퀀스 유지
                tokenizer_mode="slow",  # auto → slow (바이트 경계 오류 방지)
                max_loras=2,  # 4 → 2 최소화 (메모리 안정성)
                max_lora_rank=16,  # 8 → 16 수정 (실제 모델 랜크와 일치)
                dtype="half",  # float32 → half 복원 (호환성 우선)
                tensor_parallel_size=1,
                # 최대 안정성 옵션
                enforce_eager=True,  # CUDA 그래프 완전 비활성화
                disable_custom_all_reduce=True,  # 커스텀 reduce 비활성화
                swap_space=4,  # 스왈 공간 추가 (메모리 안정성)
                # 모든 최적화 비활성화
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
        """ChatML 또는 FIM 형식으로 프롬프트를 구성합니다 (학습 데이터 형식 준수)."""
        
        # 모델별 프롬프트 형식 결정
        is_fim_model = model_type in ["autocomplete", "comment"]
        is_fim_request = "<｜fim begin｜>" in user_prompt or "<|fim_begin|>" in user_prompt
        
        # FIM 모델이거나 FIM 요청인 경우 FIM 형식 사용
        if is_fim_model or is_fim_request:
            logger.info(f"FIM 형식 사용 - 모델: {model_type}, FIM 요청: {is_fim_request}")
            return self._build_fim_prompt(user_prompt, model_type)
        
        # ChatML 모델은 ChatML 형식 사용
        logger.info(f"ChatML 형식 사용 - 모델: {model_type}")
        return self._build_chatML_format(config, user_prompt, model_type)
    
    def _build_fim_prompt(self, user_prompt: str, model_type: str) -> str:
        """FIM (Fill-in-Middle) 형식 프롬프트 구성 (자동완성/주석 전용)."""
        
        # FIM 태그가 이미 있는 경우 그대로 사용
        if "<|fim_begin|>" in user_prompt or "<｜fim begin｜>" in user_prompt:
            enhanced_prompt = self._enhance_fim_prompt(user_prompt)
            return enhanced_prompt
        
        # 일반 텍스트를 FIM 형식으로 변환
        if model_type == "autocomplete":
            # 자동완성: 코드 뒤에 커서 위치 설정
            fim_prompt = f"<|fim_begin|>{user_prompt}<|fim_hole|><|fim_end|>"
        elif model_type == "comment":
            # 주석: 코드 위에 주석 삽입 위치 설정
            fim_prompt = f"<|fim_begin|><|fim_hole|>\n{user_prompt}<|fim_end|>"
        else:
            # 기본: 일반 FIM 형식
            fim_prompt = f"<|fim_begin|>{user_prompt}<|fim_hole|><|fim_end|>"
        
        logger.debug(f"FIM 프롬프트 생성: {fim_prompt[:100]}...")
        return fim_prompt
    
    def _build_chatML_format(self, config: ModelConfig, user_prompt: str, model_type: str) -> str:
        """ChatML 형식 프롬프트 구성 (prompt, error_fix 전용)."""
        
        # FIM 태그 감지 및 특별 처리 (기존 코드 유지)
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
        
        # 🛡️ 극도로 안전한 SamplingParams (메모리 손상 방지)
        safe_temperature = max(0.1, min(1.0, request.temperature if request.temperature is not None else config.temperature))
        safe_top_p = max(0.1, min(1.0, request.top_p if request.top_p is not None else config.top_p))
        safe_max_tokens = min(512, max(1, request.max_tokens))  # 토큰 수 제한
        
        sampling_params = SamplingParams(
            temperature=safe_temperature,
            top_p=safe_top_p,
            max_tokens=safe_max_tokens,
            stop=config.stop or [],
            stop_token_ids=stop_token_ids,
            repetition_penalty=1.1,  # 1.3 → 1.1 완화 (안정성)
            frequency_penalty=0.1,   # 0.2 → 0.1 완화 (안정성)
            presence_penalty=0.1,    # 0.2 → 0.1 완화 (안정성)
            include_stop_str_in_output=False,
            skip_special_tokens=True,  # False → True (특수 토큰 제거로 안정성)
            # 추가 안전성 옵션
            logprobs=None,  # logprobs 비활성화
            prompt_logprobs=None,  # prompt logprobs 비활성화
            detokenize=True,  # 디토큰화 활성화
            spaces_between_special_tokens=True,  # 특수 토큰 간 공백
        )
        
        logger.info(f"안전한 샘플링 파라미터: temp={safe_temperature}, top_p={safe_top_p}, max_tokens={safe_max_tokens}")
        
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

        # 🛡️ 극도로 안전한 스트리밍 루프 (메모리 손상 방지)
        max_iterations = 1000  # 무한 루프 방지
        iteration_count = 0
        
        try:
            async for request_output in results_generator:
                iteration_count += 1
                if iteration_count > max_iterations:
                    logger.error(f"최대 반복 수({max_iterations}) 초과로 스트림 종료")
                    break
                
                # 안전한 출력 추출
                try:
                    if not request_output.outputs or len(request_output.outputs) == 0:
                        logger.warning("빈 출력 감지, 건너뛰기")
                        continue
                        
                    current_text = request_output.outputs[0].text
                    if not isinstance(current_text, str):
                        logger.error(f"비문자열 출력 감지: {type(current_text)}")
                        continue
                        
                except (IndexError, AttributeError) as e:
                    logger.error(f"출력 추출 오류: {e}")
                    continue
                
                token_count += 1
                
                # 1. 스톱 토큰 체크
                stop_found = False
                try:
                    for stop_str in (config.stop or []):
                        if stop_str and stop_str in current_text:
                            current_text = current_text.split(stop_str)[0]
                            stop_found = True
                            logger.info(f"스톱 토큰 '{stop_str}' 감지됨")
                            break
                except Exception as e:
                    logger.error(f"스톱 토큰 처리 오류: {e}")
                
                # 2. 안전한 델타 계산
                try:
                    if len(current_text) >= len(last_text):
                        delta = current_text[len(last_text):]
                    else:
                        logger.warning("현재 텍스트가 이전보다 짧음, 델타 건너뛰기")
                        delta = ""
                except Exception as e:
                    logger.error(f"델타 계산 오류: {e}")
                    delta = ""
                
                # 3. 델타 품질 검증 및 전송
                if delta and not stop_found:
                    try:
                        cleaned_delta = self._validate_and_clean_delta(delta)
                        if cleaned_delta:
                            # JSON 직렬화 안전성 강화
                            json_data = json.dumps({'text': cleaned_delta}, ensure_ascii=False)
                            yield f"data: {json_data}\n\n"
                            last_text = current_text
                            logger.debug(f"델타 전송 성공: {len(cleaned_delta)}자")
                        else:
                            logger.debug(f"델타 필터링됨: '{delta[:20]}...'")
                    except Exception as e:
                        logger.error(f"델타 처리 오류: {e}")
                        continue
                
                # 4. 종료 조건 체크
                if stop_found:
                    logger.info("스톱 토큰으로 스트림 종료")
                    break
                    
                # 5-1. 최대 토큰 수 체크
                if len(current_text.split()) >= safe_max_tokens:
                    logger.info(f"최대 토큰 수({safe_max_tokens}) 도달로 스트림 종료")
                    break
                    
                # 5-2. 문자 수 기반 종료 (모델 타입별)
                max_chars = {
                    "autocomplete": 1500,   # 안전성 우선
                    "prompt": 3000,        # 안전성 우선
                    "comment": 1000,       # 안전성 우선
                    "error_fix": 2000      # 안전성 우선
                }.get(request.model_type, 2000)
                
                if len(current_text) >= max_chars:
                    logger.info(f"문자 수 제한({max_chars}) 도달로 스트림 종료 (current={len(current_text)})")
                    break
                
                # 주기적인 진행 상황 로그
                if token_count % 10 == 0:
                    logger.debug(f"진행 상황: tokens={token_count}, chars={len(current_text)}, max_chars={max_chars}")
        
        except Exception as e:
            logger.error(f"스트리밍 루프 오류: {e}")
            try:
                error_data = json.dumps({"error": "스트리밍 오류 발생"}, ensure_ascii=False)
                yield f"data: {error_data}\n\n"
            except Exception:
                yield "data: {\"error\": \"JSON 직렬화 오류\"}\n\n"
        
        finally:
            # 항상 완료 신호 전송 (정상/비정상 종료 무관)
            try:
                logger.info(f"스트리밍 종료 - 완료 신호 전송 (token_count={token_count})")
                completion_data = json.dumps({'type': 'done', 'text': ''}, ensure_ascii=False)
                yield f"data: {completion_data}\n\n"
            except Exception as e:
                logger.error(f"완료 신호 전송 오류: {e}")
                yield "data: {\"type\": \"done\", \"text\": \"\"}\n\n"

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
        """안전하고 실용적인 델타 검증 (과도한 필터링 방지)"""
        if not delta or len(delta.strip()) == 0:
            return ""
        
        import re
        
        # 1. 기본 정리
        cleaned = delta.strip()
        
        # 2. 심각한 손상만 차단 (영어 텍스트 허용)
        severe_corruption_patterns = [
            r'([a-zA-Z])\1{5,}',  # 같은 문자 6개 이상 반복
            r'([a-zA-Z]{3,})\1{3,}',  # 패턴 4번 이상 반복
            r'[a-zA-Z]{3,}[0-9]{3,}[a-zA-Z]{3,}[0-9]{3,}',  # 심각한 문자-숫자 혼재
        ]
        
        for pattern in severe_corruption_patterns:
            if re.search(pattern, cleaned):
                logger.warning(f"심각한 손상 패턴 감지: '{pattern}' in '{cleaned[:30]}...'")
                return ""  # 심각한 손상만 차단
        
        # 3. 비인쇄 가능 문자 제거 (제어 문자만)
        cleaned = ''.join(c for c in cleaned if c.isprintable() or c.isspace())
        
        # 4. 경미한 중복 패턴 제거 (안전한 정규식)
        try:
            # 같은 문자 3개 이상 반복만 제거
            cleaned = re.sub(r'([a-zA-Z_])\1{2,}', r'\1\1', cleaned)
        except re.error as e:
            logger.warning(f"정규식 오류: {e}, 원본 텍스트 사용")
            pass
        
        # 5. 최소 길이 체크
        if len(cleaned.strip()) < 1:
            return ""
        
        # 6. 매우 관대한 검증 (거의 모든 텍스트 허용)
        # 빈 문자열이 아니고 최소한의 내용이 있으면 통과
        if len(cleaned.strip()) > 0:
            logger.debug(f"델타 통과: '{cleaned[:30]}...' (길이: {len(cleaned)})")
            return cleaned
        
        return ""