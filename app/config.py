# app/config.py
"""
애플리케이션의 모든 설정 값을 중앙에서 관리.
- 기본 모델 경로
- LoRA 모델별 설정
"""
from dataclasses import dataclass
from typing import List, Dict

@dataclass
class ModelConfig:
    """개별 LoRA 모델의 설정을 위한 데이터 클래스"""
    name: str
    adapter_path: str
    lora_id: int
    max_tokens: int
    temperature: float
    top_p: float
    system_prompt: str
    stop: List[str] | None = None

# 기본 모델 경로
BASE_MODEL_PATH = "TheBloke/deepseek-coder-6.7B-instruct-AWQ"

# 각 LoRA 어댑터별 설정
MODEL_CONFIGS: Dict[str, ModelConfig] = {
    "autocomplete": ModelConfig(
        name="autocomplete",
        adapter_path="/lora_adapters/autocomplete-finetuned/final_model",
        lora_id=1,
        max_tokens=128,
        temperature=0.3,
        top_p=0.9,
        system_prompt="You are an expert python code completion engine.Always end your response with the exact line: # --- Generation Complete ---",
        stop=["<|EOT|>", "<|fim_end|>", "\n# --- Generation Complete ---"]
    ),
    "prompt": ModelConfig(
        name="prompt",
        adapter_path="/lora_adapters/prompt-finetuned/final_model",
        lora_id=2,
        max_tokens=1024,
        temperature=0.6,
        top_p=0.95,
        system_prompt="You are a helpful senior Python developer.Always end your response with the exact line: # --- Generation Complete ---",
        stop=["<|EOT|>", "<|fim_end|>", "\n# --- Generation Complete ---"]
    ),
    "comment": ModelConfig(
        name="comment",
        adapter_path="/lora_adapters/comment-finetuned/final_model",
        lora_id=3,
        max_tokens=256,
        temperature=0.4,
        top_p=0.9,
        system_prompt="You are a python code documentation assistant.Always end your response with the exact line: # --- Generation Complete ---",
        stop=["<|EOT|>", "<|fim_end|>", "\n# --- Generation Complete ---"]
    ),
    "error_fix": ModelConfig(
        name="error_fix",
        adapter_path="/lora_adapters/error-fix-finetuned/final_model",
        lora_id=4,
        max_tokens=512,
        temperature=0.5,
        top_p=0.9,
        system_prompt="You are a python debugging expert.Always end your response with the exact line: # --- Generation Complete ---",
        stop=["<|EOT|>", "<|fim_end|>", "\n# --- Generation Complete ---"]
    ),
}