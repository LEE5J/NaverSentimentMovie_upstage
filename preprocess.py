import pandas as pd
from vllm import LLM, SamplingParams
from tqdm import tqdm
import os
import sys

# --- 1. 설정 ---
CSV_INPUT_FILE = "train.csv"
CSV_OUTPUT_FILE = "train_processed_vllm.csv"
REVIEW_COLUMN_NAME = "review"
NEW_COLUMN_NAME = "processed_review"

# vLLM 모델 설정
# Qwen 모델은 trust_remote_code=True가 필요할 수 있습니다.
LLM_MODEL_NAME = "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct" 
BATCH_SIZE = 32
# vLLM이 사용할 GPU 메모리 비율 (예: 90%)
# tensor_parallel_size는 보유한 GPU 개수
llm = LLM(
    model=LLM_MODEL_NAME,
    trust_remote_code=True, 
    gpu_memory_utilization=0.9,
    # tensor_parallel_size=1  # 1개의 GPU를 사용하는 경우
)

# vLLM 샘플링 파라미터 (API의 temperature, max_tokens 등)
sampling_params = SamplingParams(
    temperature=0.0,
    max_tokens=2048 # 파싱 오류 방지를 위해 넉넉하게
)

# --- 2. LLM 프롬프트 설정 (이전과 동일) ---
SYSTEM_PROMPT = """당신은 BERT 학습용 리뷰 데이터를 전처리하는 NLP 전문가입니다.
목표: 감정은 보존하되, 노이즈와 스팸은 제거하여 BERT에 최적화된 텍스트 생성.
출력 규정: 반드시 [생각] 태그 (적용 규칙과 판단 근거)를 먼저 서술한 뒤, [결과] 태그 (최종 정제 텍스트 또는 '[삭제]' 토큰)를 출력합니다."""

RULES = """<규칙>
1.  [삭제] 대상 (스팸/노이즈):
    * 의미 없는 키보드 연타 (예: `asdasd`, `ㅇ렇ㄹㄹㄹ`).
    * 의미 없는 자음/모음 나열 (예: `ㅈㅈㅈㅈ`). (단, `ㅋㅋ`, `ㅠㅠ`와 같이 감정이 있는 경우는 제외)
    * 리뷰와 무관한 내용 (예: "이벤트 참여").
2.  표준화 (BERT 최적화):
    * 명백한 오탈자 수정 (예: `재밋네요` -> `재밌네요`).
    * 감정 신조어/줄임말 (예: `꿀잼`, `노잼`)은 재미있다 재미없다 와 같이 순화
    * 토크나이징 되지 않을 것 같은 말들은 적절히 대체할 것
3.  반복 문자 처리 (감정 보존):
    * 의미 있는 반복 ( `ㅋㅋ`, `ㅠㅠ`, `ㅏㅏ` 등) -> 2글자로 축약 (예: `ㅋㅋㅋㅋ` -> `ㅋㅋ`).
    * 문장 부호 (`!`, `?`) -> 1글자로 축약 (예: `!!!!` -> `!`).
    * 마침표 (`.`) -> 3글자(`...`)로 표준화 (예: `.`/`..` -> `...`).
    * 이모지 (예: 👍)는 토크나이징 되지 않으니 글자로 변환
    * 관련 없는 특수기호는 제거."""

FEW_SHOT_EXAMPLES = """< Few-shot 예시 >
* `<입력>`: `ㅇ렇ㄹㄹㄹㅇㄹㄹㄹㄹ`
* `<출력>`:
    `[생각]`
    1. 입력 텍스트는 `ㅇ렇ㄹㄹㄹㅇㄹㄹㄹㄹ`입니다.
    2. 규칙 1 (스팸 처리)에 따라, 이는 의미를 알 수 없는 자음/모음의 단순 나열에 해당합니다.
    3. 학습에 방해가 되므로 `[삭제]` 토큰을 반환합니다.
    `[결과]`
    `[삭제]`
* `<입력>`: `이 영화 진짜 재밋네요!!!! 짱짱 👍`
* `<출력>`:
    `[생각]`
    1. 입력 텍스트는 `이 영화 진짜 재밋네요!!!! 짱짱 👍`입니다.
    2. 규칙 2 (오탈자)에 따라 `재밋네요`를 `재밌네요`로 수정합니다.
    3. 규칙 3 (문장 부호 표준화)에 따라 `!!!!`를 `!`로 표준화합니다.
    4. 규칙 3 (이모지)에 따라 `👍`는 `따봉`으로 대체합니다.
    `[결과]`
    `이 영화 진짜 재밌네요! 짱짱 따봉`
* `<입력>`: `ㅠㅠㅠㅠㅠㅠㅠㅠ`
* `<출력>`:
    `[생각]`
    1. 입력 텍스트는 `ㅠㅠㅠㅠㅠㅠㅠㅠ`입니다.
    2. 규칙 3 (의미 있는 반복 표준화)에 따라 `ㅠㅠ`가 3회 이상 반복되었으므로 `ㅠㅠ`로 표준화합니다.
    `[결과]`
    `ㅠㅠ`
"""

USER_PROMPT_TEMPLATE = f"""
[작업 지시]
아래 `<규칙>`에 따라 입력된 `<원본 리뷰>`를 정제하십시오.
반드시 `[생각]` 태그로 먼저 추론 과정을 설명한 뒤, `[결과]` 태그에 최종 정제된 텍스트를 출력해야 합니다.

{RULES}

{FEW_SHOT_EXAMPLES}

---
[처리할 리뷰 원본]

{{review_text}}
"""

def generate_chat_prompt(review_text):
    """
    vLLM의 generate 함수는 단일 문자열 프롬프트가 아닌,
    채팅 포맷(메시지 리스트)을 처리할 수 없습니다.
    따라서 채팅 템플릿을 모델에 적용하여 단일 문자열로 만들어야 합니다.
    """
    # Exaone 모델의 채팅 템플릿을 직접 구성
    # (모델마다 템플릿이 다를 수 있으므로 확인 필요)
    # 일반적인 <|system|>, <|user|>, <|assistant|> 형식 사용
    
    prompt = f"<|system|>\n{SYSTEM_PROMPT}\n"
    prompt += f"<|user|>\n{USER_PROMPT_TEMPLATE.format(review_text=review_text)}\n"
    prompt += f"<|assistant|>\n" # 모델이 이어서 생성하도록
    return prompt

def parse_result(raw_response):
    """
    vLLM의 응답에서 [결과] 태그를 파싱하는 함수
    """
    if "[결과]" in raw_response:
        processed_text = raw_response.split("[결과]", 1)[-1].strip()
        return processed_text
    elif pd.isna(raw_response) or not raw_response.strip():
        return "[입력없음]"
    else:
        print(f"경고: 파싱 실패. 원본 응답: {raw_response[:200]}...", file=sys.stderr)
        return "[파싱오류]"

# --- 3. 메인 실행 로직 ---

def main():
    print(f"'{CSV_INPUT_FILE}'에서 데이터를 읽어옵니다...")
    try:
        df = pd.read_csv(CSV_INPUT_FILE)
    except FileNotFoundError:
        print(f"오류: '{CSV_INPUT_FILE}'을 찾을 수 없습니다.", file=sys.stderr)
        return
    except pd.errors.EmptyDataError:
        print(f"오류: '{CSV_INPUT_FILE}'이 비어있습니다.", file=sys.stderr)
        return

    if REVIEW_COLUMN_NAME not in df.columns:
        print(f"오류: CSV에 '{REVIEW_COLUMN_NAME}' 컬럼이 없습니다.", file=sys.stderr)
        return

    print("프롬프트를 생성 중입니다...")
    reviews_texts = df[REVIEW_COLUMN_NAME].fillna("").tolist()
    
    # 모든 프롬프트를 미리 생성합니다.
    prompts = [generate_chat_prompt(text) for text in reviews_texts]
    
    total_reviews = len(prompts)
    if not total_reviews:
        print("처리할 리뷰가 없습니다.")
        return

    print(f"총 {total_reviews}개의 리뷰를 {BATCH_SIZE}개씩 미니 배치로 vLLM으로 처리합니다...")
    print(f"사용 모델: {LLM_MODEL_NAME}")
    
    processed_results = []
    
    # *** 핵심: 전체 프롬프트를 BATCH_SIZE만큼 나눠서 루프 실행 ***
    for i in tqdm(range(0, total_reviews, BATCH_SIZE), desc="Processing batches"):
        # 현재 배치를 가져옵니다 (예: [0:1024], [1024:2048], ...)
        batch_prompts = prompts[i:i + BATCH_SIZE]
        
        # vLLM의 llm.generate() 호출 (배치 단위)
        batch_outputs = llm.generate(batch_prompts, sampling_params)
        
        # 현재 배치 결과 파싱
        for output in batch_outputs:
            raw_response = output.outputs[0].text 
            processed_results.append(parse_result(raw_response))

    print("일괄 처리 완료.")
    
    # 결과를 새 컬럼으로 추가
    df[NEW_COLUMN_NAME] = processed_results
    
    # 새 CSV 파일로 저장
    print(f"결과를 '{CSV_OUTPUT_FILE}'에 저장합니다...")
    df.to_csv(CSV_OUTPUT_FILE, index=False, encoding='utf-8-sig')
    
    print("작업 완료.")
    print(f"저장된 파일: {os.path.abspath(CSV_OUTPUT_FILE)}")
    
    # (오류 확인 로직은 동일)
    error_count = sum(1 for res in processed_results if str(res).startswith("["))
    if error_count > 0:
        print(f"\n총 {error_count}개의 처리 오류/실패가 발생했습니다.")
        delete_count = processed_results.count("[삭제]")
        print(f"정상적으로 [삭제] 처리된 항목: {delete_count}개")


if __name__ == "__main__":
    main()