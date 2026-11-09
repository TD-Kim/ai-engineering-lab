# tests

> 상태: **자리표시자**

여러 Part에 걸친 **공통 테스트**를 둔다(예: 평가 지표 함수, 데이터 로딩 유틸).
각 실습 전용 테스트는 해당 실습 디렉터리의 `test_*.py`에 둔다.

## 실행

```powershell
.\.venv\Scripts\Activate.ps1
pytest                      # 기본: solution 검증
$env:IMPL = "starter"; pytest part1_ai_fundamentals/transformer   # starter 검증
```

규약: [docs/ARCHITECTURE.md §3](../docs/ARCHITECTURE.md#3-실습-파일-규약-startersolutiontest)
