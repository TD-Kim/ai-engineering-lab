import re
import textwrap
# 파이썬: collections는 표준 라이브러리. Counter는 "무엇이 몇 번 나왔나"를 세는 딕셔너리다.
#   예: c = Counter(); c["a"] += 1 → c["a"]는 1. 없는 키를 꺼내도 오류 없이 0이 나온다.
from collections import Counter
from pydoc import doc
from typing import List

import requests

OLLAMA_LLM_HOST="http://localhost:11434/api/chat"
MODEL_NAME="qwen2.5:1.5b"
BASE_OPTIONS = {"temperature": 0.7}  # seed 말고 모든 실행에서 같게 맞출 설정
SEEDS = [42, 1, 2]  # 같은 질문을 seed마다 한 번씩. 42는 지난번 실행과 비교하기 위해 맨 앞에 둠
NOT_IN_DOC = "문서에 없습니다"

def chat_raw(messageList: List, options: dict | None = None) -> dict:
    """응답 JSON 전체를 돌려준다(content + prompt_eval_count 등을 함께 쓰려고)"""
    payload = {"model": MODEL_NAME, "messages": messageList, "stream": False}
    if options:
        payload["options"] = options
    response = requests.post(url=OLLAMA_LLM_HOST, json=payload, timeout=120)
    response.raise_for_status()
    return response.json()

def chat(messageList: List) -> str:
    # payload = {"model": MODEL_NAME, "messages": messageList, "stream": False}
    # response = requests.post(url=OLLAMA_LLM_HOST, json=payload, timeout=30)
    # response.raise_for_status() # 실패시 예외처리하는 함수
    # responseDict = response.json()
    # return responseDict["message"]["content"]
    return chat_raw(messageList)["message"]["content"]

def makeDedentStr(s):
    return textwrap.dedent(s).strip()

def build_messages(layout: str, rule: str, docs: List[str], question: str) -> List[dict]:
    labeled = [f"[문서{i}]\n{doc}" for i, doc in enumerate(docs, start=1)]
    document = "\n\n".join(labeled)
    reminder = '(문서에 없으면 "문서에 없습니다"라고 답하라)'
    
    if layout == "A":
        system = "\n".join([rule, document])
        user = question
    elif layout == "B":
        system = document
        user = "\n\n".join([rule, question])
    elif layout == "C":
        system = "\n".join([rule, document])
        user = "\n".join([question, reminder])
    else:
        raise ValueError(f"알 수 없는 배치: {layout}")
    
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user}
    ]
    
def is_success(answer: str, expect: str | None) -> bool:
    """expect가 있으면(문서에 답 있음): 정답 말이 있고 거절은 없어야 성공.
    expect가 None이면(문서에 답 없음): 거절 문구가 있고 숫자가 없어야 성공(지어낸 숫자 걸러내기)."""
    refused = NOT_IN_DOC in answer
    if expect is None:
        return refused and not re.search(r"\d", answer)
    return expect in answer and not refused

# ── 응답에서 부분을 꺼내는 패턴 3개 ────────────────────────────
# 파이썬: re.compile(패턴, 옵션) → 정규식 "패턴"을 미리 만들어 두고 여러 번 재사용한다.
#   re.S(=DOTALL) 옵션: 패턴의 `.`이 줄바꿈 글자까지 포함하게 한다.
# 패턴 읽는 법:
#   \s*   공백·줄바꿈 0개 이상      \[ \]  대괄호 글자 그 자체 (그냥 [ ]는 다른 뜻이라 \로 막는다)
#   \d+   숫자 1개 이상             ["“]   따옴표 " 또는 “ 중 하나
#   (.+?) 아무 글자 1개 이상, 되도록 짧게. ( )로 감싼 부분이 나중에 group으로 꺼내진다
CITE_RE = re.compile(r'근거:\s+\[문서\s+(\d+)\]\s*["“](.+?)["”]', re.S)
NONE_RE = re.compile(r"근거:\s*없음")
ANSWER_RE = re.compile(r"답:\s*(.+)", re.S) # "답:" 뒤의 나머지 전부

def parse_response(text: str):
    # 개념: LLM 응답 문자열 하나를 (번호, 인용 문장, 답) 세 조각으로 나눈다.
    # 입력: 응답 문자열. 출력: 아래 세 경우 중 하나
    #   - 형식이 깨졌으면 None
    #   - 정상이면 (번호, 인용 문장, 답)
    #   - "근거: 없음"이면 (None, None, 답)
    # WHY: 조각으로 나눠야 인용이 진짜인지 코드로 검사할 수 있다.
    a = ANSWER_RE.search(text)  # 파이썬: 패턴.search(문자열) -> 문자열 안에서 처음 맞는 곳을 찾는다. 없으면 None
    if a is None:
        return None
    answer = a.group(1).strip() # 파이썬: .group(1) -> 첫 번째 ( ) 안에서 잡힌 글자(문자열)
                                #        .strip() -> 문자열 앞뒤의 공백, 줄바꿈을 없앤다
    c = CITE_RE.search(text)
    if c:
        return int(c.group(1)), c.group(2), answer  # 파이썬: int("2") -> 숫자 2.  group의 결과는 항상 문자열이다
    if NONE_RE.search(text):
        return None, None, answer
    return None

def normalize(s: str) -> str:
    # 개념: 인용과 원문을 비교하기 전에 공백·마침표·쉼표·따옴표를 없앤다.
    # WHY: 모델이 마침표를 빼거나 띄어쓰기를 바꿔도 같은 문장이면 "진짜 인용"으로 봐야 해서.
    # 주의: 양쪽(인용·원문)에 같은 정리를 해야 공정하다. 너무 느슨하면 이 패턴을 줄일 것.
    # 파이썬: re.sub(패턴, 바꿀 글자, 문자열) → 패턴에 맞는 곳을 모두 바꾼 새 문자열을 돌려준다(원본은 그대로).
    #   [\s.,\"'“”‘’] → 공백류, 마침표, 쉼표, 여러 종류의 따옴표 중 한 글자
    return re.sub(r"[\s.,\"'“”‘’]", "", s)

def check_citation(num, quote, docs: List[str]) -> str:
    # 개념: 인용이 문서에 "존재하는지"만 검사한다(질문에 맞는지는 못 본다).
    # 출력: "인용없음" / "번호가짜" / "문장가짜" / "정상인용" 중 하나. 위에서부터 순서대로 걸러 낸다.
    if num is None:                    # 근거: 없음 → 검사할 인용이 없다
        return "인용없음"
    if not 1 <= num <= len(docs):      # 파이썬: 1 <= num <= 3 처럼 "범위 안인가"를 한 줄로 쓰는 연쇄 비교
                                       #        len(docs) → 리스트에 든 문서 개수
        return "번호가짜"
    q = normalize(quote)
    # 파이썬: A in B → 문자열 B 안에 A가 통째로 들어 있으면 True
    #        not q → 빈 문자열("")이면 True. 빈 문자열은 어느 글에도 "들어 있다"로 나오므로 따로 막는다
    #        docs[num - 1] → 리스트는 0부터 세므로 문서 N은 N-1번 칸
    if not q or q not in normalize(docs[num - 1]):
        return "문장가짜"
    return "정상인용"

# 개념: 근거를 먼저 쓰게 하는 규칙 문구. 모델이 줄 형식(`근거:` → `답:`)을 정해 준다.
# 파이썬: makeDedentStr은 우리 파일에 있는 함수. 안의 textwrap.dedent(s)가 줄마다 같은 들여쓰기를 지우고, .strip()이 앞뒤 빈 줄을 지운다.
RULE_CITE = makeDedentStr("""
    문서에 있는 내용으로만 답하라.
    답하기 전에 먼저 아래 형식으로 쓴다.
    근거: [문서N] "문서의 문장을 그대로"
    답: (답)
    문서에 없으면 아래처럼 쓴다.
    근거: 없음
    답: 문서에 없습니다
""")

def run_one(layout, rule, docs, question, expect, seed, use_cite):
    # 개념: 질문 하나를 모델에 한 번 보내고, 판정과 토큰 수까지 묶어서 돌려준다.
    # 입력: 배치(A/B/C), 규칙 문구, 문서 목록, 질문, 기대값(expect), seed, 인용 검사를 할지
    # 출력: (정답 여부, 인용 결과, 입력 토큰, 출력 토큰, 응답 원문)
    # 서비스 관점: 지난번에는 입력 토큰만 봤지만, 근거를 요구하면 출력이 길어지므로 출력 토큰(eval_count)도 비용·지연의 판단 재료다.
    res = chat_raw(build_messages(layout, rule, docs, question),
                   {**BASE_OPTIONS, "seed": seed})   # 파이썬: {**a, "k": v} → 딕셔너리 a를 펼쳐 새 딕셔너리에 합치고 "k"를 더한다(원본 a는 안 바뀜)
    content = res["message"]["content"]              # 파이썬: 딕셔너리["키"]로 값 꺼내기. 응답 JSON이 파이썬 딕셔너리다
    cite = "-"                                       # "-"는 "검사 안 함"
    answer_part = content
    if use_cite:
        parsed = parse_response(content)
        if parsed is None:
            cite = "형식깨짐"                          # 이 경우 정답 판정은 응답 전체로 한다
        else:
            num, quote, answer_part = parsed         # 파이썬: 세 값 튜플을 세 변수에 한 번에 나눠 담는다(언패킹)
            cite = check_citation(num, quote, docs)
    # 답 부분만 넣어야 "[문서3]"의 숫자 3에 "숫자가 있으면 지어낸 답" 검사가 걸리지 않는다
    ok = is_success(answer_part, expect)
    return ok, cite, res["prompt_eval_count"], res["eval_count"], content

# 비교할 배치 하나. 이번 실험에서 바꾸는 것은 '규칙'뿐이다
LAYOUT = "A"

if __name__ == "__main__":
    rule_old = makeDedentStr("""
        문서에 있는 내용으로만 답하라.
        문서에 없으면 "문서에 없습니다"라고 답하라.
    """)
    doc1 = makeDedentStr("달토끼 마을의 시장 이름은 별이다. 마을 인구는 12명이다.")
    doc2 = makeDedentStr("철수와 영희는 소꿉친구이다. 철수는 학교를 1년 일찍 입학했다.")
    doc3 = makeDedentStr("영준이의 키는 174cm 이다. 몸무게는 82kg 이다.")

    # (질문, 답이 들어 있는 문서, 정답에 들어갈 말). 문서에 답이 없는 질문은 (질문, None, None)
    questions = [
        ("달토끼 마을의 시장 이름은 뭐야?", doc1, "별"),
        ("달토끼 마을의 인구는 몇 명이야?", doc1, "12"),
        ("철수와 영희는 어떤 사이야?", doc2, "소꿉친구"),
        ("철수는 학교에 언제 입학했어?", doc2, "1년 일찍"),
        ("영준이의 키는 얼마야?", doc3, "174"),
        ("영준이의 몸무게는 얼마야?", doc3, "82"),
        ("달토끼 마을의 부시장 이름은 뭐야?", None, None),
        ("영희의 키는 얼마야?", None, None),        # 영희(문서2)와 키 숫자(문서3)가 섞이는지
        ("철수의 몸무게는 얼마야?", None, None),    # 같은 함정
        ("영준이의 나이는 몇 살이야?", None, None),
        ("달토끼 마을의 면적은 얼마야?", None, None),
        ("대한민국의 수도는 어디야?", None, None),  # 문서 밖 상식도 거절이 정답이라고 본 경우
    ]

    doc_sets = {"문서 3개": [doc1, doc2, doc3], "문서 1개": [doc1]}
    rules = {"기존": rule_old, "근거": RULE_CITE}   # 비교할 두 규칙. 이름이 표의 "규칙" 열에 나온다

    # 파이썬: print(...)는 화면에 출력한다. \t는 탭(표처럼 칸을 띄움)
    print("세트\t규칙\t있음 성공\t없음 성공\t형식깨짐\t번호가짜\t문장가짜\t입력토큰\t출력토큰\tseed별 합계")

    # 파이썬: 딕셔너리.items() → (키, 값) 쌍을 하나씩 꺼내며 반복한다
    for set_name, docs in doc_sets.items():
        for rule_name, rule_text in rules.items():
            use_cite = rule_name == "근거"            # True/False. "근거" 규칙일 때만 인용 검사를 한다
            hit = {"있음": [0, 0], "없음": [0, 0]}    # 종류별 [성공 횟수, 전체 횟수]
            # 파이썬: {s: 값 for s in 목록} → 딕셔너리 컴프리헨션. seed마다 [성공, 전체] 칸을 만든다
            per_seed = {s: [0, 0] for s in SEEDS}
            cites = Counter()                          # 인용 결과("정상인용" 등)를 종류별로 센다
            tin, tout = [], []                         # 입력·출력 토큰 수를 모은다(나중에 평균)

            # 파이썬: enumerate(목록, start=1) → (1, 첫 항목), (2, 둘째 항목)… 번호와 항목을 같이 준다
            for n, (question, answer_doc, expect) in enumerate(questions, start=1):
                # 이 세트에 답이 든 문서가 없으면 "문서에 답 없음" 질문으로 본다
                # 파이썬: answer_doc in docs → 리스트 안에 있는지. None은 어느 리스트에도 없으므로 False
                if answer_doc in docs:
                    kind, expect_now = "있음", expect
                else:
                    kind, expect_now = "없음", None

                for seed in SEEDS:
                    ok, cite, t_in, t_out, content = run_one(
                        LAYOUT, rule_text, docs, question, expect_now, seed, use_cite)
                    tin.append(t_in)                  # 파이썬: 리스트.append(x) → 리스트 맨 끝에 x를 더한다
                    tout.append(t_out)
                    cites[cite] += 1                  # Counter: 이 결과의 횟수를 1 올린다
                    hit[kind][1] += 1                 # 전체 +1
                    per_seed[seed][1] += 1
                    if ok:
                        hit[kind][0] += 1             # 성공 +1
                        per_seed[seed][0] += 1
                    # 파이썬: f"...{식}..." → { } 안의 값을 글자 속에 넣는다.
                    #        {content!r} → 글자를 따옴표·\n 포함한 원래 모양으로 보여 준다(공백·줄바꿈 확인용)
                    print(f"  [{set_name}/{rule_name}/seed{seed}/Q{n}/{kind}] "
                          f"{'O' if ok else 'X'} {cite} {content!r}")

            # 파이썬: lambda k: 식 → 이름 없는 한 줄짜리 함수. "기존" 규칙은 인용 검사를 안 했으니 "-"로 보여 준다
            fmt = (lambda k: str(cites[k])) if use_cite else (lambda k: "-")
            # 파이썬: "구분자".join(목록) → 목록의 글자들을 구분자로 이어 붙인다
            seed_text = "  ".join(f"{s}:{v[0]}/{v[1]}" for s, v in per_seed.items())
            # 파이썬: sum(리스트) / len(리스트) → 평균. {값:.1f}는 소수 첫째 자리까지 표시
            print(f"{set_name}\t{rule_name}\t{hit['있음'][0]}/{hit['있음'][1]}\t"
                  f"{hit['없음'][0]}/{hit['없음'][1]}\t{fmt('형식깨짐')}\t{fmt('번호가짜')}\t"
                  f"{fmt('문장가짜')}\t{sum(tin)/len(tin):.1f}\t{sum(tout)/len(tout):.1f}\t{seed_text}")