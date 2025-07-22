# app/schemas.py
"""
API 요청 및 응답 데이터 구조를 정의하는 Pydantic 모델.
이 스키마는 모델 서버와 웹 서버 백엔드 간의 "API 계약" 역할을 합니다.
"""
from pydantic import BaseModel, Field
from typing import Dict, Any

class Prompt(BaseModel):
    """프롬프트 빌더 함수에 전달될 내부 데이터 구조"""
    prompt: str
    user_select_options: Dict[str, Any]

class GenerateRequest(BaseModel):
    """
    단일 응답 생성을 위한 요청 스키마.
    웹 서버 백엔드는 이 구조에 맞춰 모델 서버에 요청해야 합니다.
    """
    # --- 웹 서버 백엔드가 채워야 할 필드 ---
    prompt: str = Field(..., description="사용자의 원본 프롬프트. 클라이언트로부터 전달받습니다.")
    model_type: str = Field(..., description="사용할 LoRA 모델 타입. 웹 서버가 비즈니스 로직에 따라 결정합니다.")
    user_select_options: Dict[str, Any] = Field(
        default_factory=dict,
        description="사용자 맞춤 설정. 웹 서버가 DB에서 조회한 후 이 필드를 채워서 전달해야 합니다."
    )

    # --- 웹 서버가 선택적으로 오버라이드할 수 있는 필드 ---
    temperature: float | None = Field(default=None, ge=0.0, description="Sampling temperature, between 0 and 2.")
    max_tokens: int | None = Field(default=None, gt=0, description="Maximum number of tokens to generate.")
    top_p: float | None = Field(default=None, gt=0.0, le=1.0, description="Top-p sampling parameter.")
    stop: list[str] | None = Field(default=None, description="List of strings that stop the generation.")


class StreamGenerateRequest(BaseModel):
    """
    스트리밍 응답 생성을 위한 요청 스키마.
    웹 서버 백엔드는 이 구조에 맞춰 모델 서버에 요청해야 합니다.
    """
    # --- 웹 서버 백엔드가 채워야 할 필드 ---
    user_id: int = Field(..., description="사용자 식별자. 로깅 및 추적 용도로 사용됩니다.")
    prompt: str = Field(..., description="사용자의 원본 프롬프트. 클라이언트로부터 전달받습니다.")
    model_type: str = Field(..., description="사용할 LoRA 모델 타입. 웹 서버가 비즈니스 로직에 따라 결정합니다.")
    user_select_options: Dict[str, Any] = Field(
        ...,
        description="사용자 맞춤 설정. 웹 서버가 DB에서 조회한 후 이 필드를 채워서 전달해야 합니다."
    )

    # --- 웹 서버가 선택적으로 오버라이드할 수 있는 필드 ---
    temperature: float | None = Field(default=None, ge=0.0, description="Sampling temperature.")
    top_p: float | None = Field(default=None, gt=0.0, le=1.0, description="Top-p sampling parameter.")
    max_tokens: int = Field(default=2048, gt=0, description="스트림을 통해 생성할 전체 최대 토큰 수")