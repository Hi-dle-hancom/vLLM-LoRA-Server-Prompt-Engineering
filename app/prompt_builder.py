# app/prompt_builder.py
"""
사용자 설정을 기반으로 LLM에 전달할 최종 프롬프트를 동적으로 생성합니다.
"""
import random
from typing import Dict, Any

# Pydantic 모델은 schemas에서 직접 임포트합니다.
from .schemas import Prompt

# 1. 기반(Foundation) 페르소나 템플릿 - 온보딩 시스템에 맞춰 간소화됨
PROMPT_TEMPLATES = {
    "beginner": {
        "TASK_CONTEXT": [
            "You are a friendly and patient Python tutor for absolute beginners.",
            "You are a helpful teaching assistant for a beginner's Python course.",
        ],
        "TONE_CONTEXT": [
            "Use simple, easy-to-understand language. Explain technical terms with analogies.",
            "Be very encouraging and supportive. Focus on building the user's confidence.",
        ],
        "TASK_DESCRIPTION": [
            "First, find and correct any errors. Then, explain the fundamental concepts behind the code line by line.",
            "Provide the corrected code, then break down the reason for the error and the logic of the solution in simple steps.",
        ]
    },
    "intermediate": {
        "TASK_CONTEXT": [
            "You are a helpful senior developer conducting a friendly code review.",
            "You are a collaborative coding partner helping a colleague refine their code.",
        ],
        "TONE_CONTEXT": [
            "Be constructive and professional. Focus on best practices and code readability.",
            "Assume the user knows the basics. Your tone should be that of a helpful peer.",
        ],
        "TASK_DESCRIPTION": [
            "Identify potential bugs and suggest more 'Pythonic' alternatives. Explain the trade-offs.",
            "Review the code for clarity, efficiency, and edge cases. Introduce relevant standard library functions if applicable.",
        ]
    }
}

# 2. 수식어(Modifiers) 지시사항 매핑 - 온보딩 시스템에 맞춰 간소화됨
INSTRUCTION_MAPPINGS = {
    "code_output_structure": {
        "minimal": "Provide only the core logic code with minimal comments.",
        "standard": "Structure the code normally with basic comments explaining each part.",
        "detailed": "Generate code with detailed comments, type hints, and exception handling."
    },
    "explanation_style": {
        "brief": "Explain the core concepts briefly and quickly.",
        "standard": "Provide the code followed by a simple explanation.",
        "detailed": "Explain the concept, the reason behind the implementation, and how to use it.",
        "educational": "Give a step-by-step explanation with examples and related concepts, like a mini-tutorial."
    }
}


# 3. 프롬프트 생성 함수 - 온보딩 시스템에 맞춰 간소화됨
def generate_enhanced_prompt(prompt: Prompt) -> str:
    """온보딩 시스템 설정을 반영하여 최종 프롬프트를 동적으로 생성합니다."""
    
    options = prompt.user_select_options
    
    # 1. 기반 설정: 스킬 레벨에 따른 기본 페르소나 선택
    skill_level = options.get("python_skill_level", "intermediate")
    base_template = PROMPT_TEMPLATES.get(skill_level, PROMPT_TEMPLATES["intermediate"])
    
    task_context = random.choice(base_template["TASK_CONTEXT"])
    tone_context = random.choice(base_template["TONE_CONTEXT"])
    task_description = random.choice(base_template["TASK_DESCRIPTION"])
    
    # 2. 수식어 설정: 온보딩 시스템 옵션들(코드 구조, 설명 스타일)을 지시사항으로 변환
    additional_instructions = []
    
    # 온보딩 시스템에서 정의된 옵션만 처리
    supported_options = ["code_output_structure", "explanation_style"]
    
    for key, value in options.items():
        if key == "python_skill_level" or key not in supported_options:
            continue
            
        # 단일 선택 옵션 처리
        instruction = INSTRUCTION_MAPPINGS.get(key, {}).get(value)
        if instruction:
            additional_instructions.append(f"- {instruction}")

    # 3. 프롬프트 조립
    final_prompt_parts = [
        f"--- Persona & Task ---",
        task_context,
        tone_context,
        task_description,
        f"--- User's Original Request ---\n{prompt.prompt}"
    ]
    
    if additional_instructions:
        instruction_section = "\n".join(additional_instructions)
        final_prompt_parts.append(f"--- Specific Requirements ---\nPlease adhere to the following specific requirements based on the user's settings:\n{instruction_section}")

    final_prompt = "\n\n".join(final_prompt_parts)
    
    return final_prompt