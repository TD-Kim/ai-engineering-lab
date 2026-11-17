import textwrap
from typing import List

import requests

OLLAMA_LLM_HOST="http://localhost:11434/api/chat"
MODEL_NAME="qwen2.5:1.5b"

def chat(messageList: List) -> str:
    payload = {"model": MODEL_NAME, "messages": messageList, "stream": False}
    response = requests.post(url=OLLAMA_LLM_HOST, json=payload, timeout=30)
    response.raise_for_status() # 실패시 예외처리하는 함수
    responseDict = response.json()
    return responseDict["message"]["content"]

def makeDedentStr(s):
    return textwrap.dedent(s).strip()

if __name__ == "__main__":
    rule = makeDedentStr("""
        문서에 있는 내용으로만 답하라.
        문서에 없으면 "문서에 없습니다"라고 답하라.
    """)
    doc1 = makeDedentStr("""
                         달토끼 마을의 시장 이름은 별이다. 마을 인구는 12명이다.
                         """)
    doc2 = makeDedentStr("""
                         철수와 영희는 소꿉친구이다. 철수는 학교를 1년 일찍 입학했다.
                         """)
    doc3 = makeDedentStr("""
                         영준이의 키는 174cm 이다. 몸무게는 82kg 이다.
                         """)
    docs = [doc1, doc2, doc3]
    labeled = [f"[문서{i}]\n{doc}" for i, doc in enumerate(docs, start=1)]
    document = "\n\n".join(labeled)
    system = {"role": "system", "content": "\n".join([rule, document])}
    print(repr(system["content"]))
    message = {"role": "user", "content": "달토끼 마을의 부시장 이름은 뭐야?"}
    messageList = [system, message]
    resText = chat(messageList)
    # print(json.dumps(responseDict, indent=2, ensure_ascii=False))
    print(resText)
