## Báo cáo chi tiết CI & MLflow Experiment Tracking

---

## I. MLflow Experiment Tracking

### Bối cảnh & Vấn đề

Dự án SLABAI có pipeline AI agent (`src/agent.py`) dùng OpenAI GPT-4o với tool calling. Trước khi thêm MLflow, toàn bộ cấu hình bị hardcode:

```python
# Trước — không thể thay đổi hay so sánh
response = client.chat.completions.create(
    model="gpt-4o",        # cứng
    temperature=0,         # không có
    messages=messages,
)
# Không biết tốn bao nhiêu tiền
# Không biết latency bao lâu
# Không so sánh được prompt A vs prompt B
```

Arize Phoenix đã có sẵn để trace OpenTelemetry, nhưng không đủ cho experiment comparison — Phoenix log traces từng request, MLflow log và **so sánh nhiều runs** với nhau.

---

### Giải pháp — `src/mlflow_tracker.py`

Tạo module wrapper độc lập với 2 thành phần:

#### A. `calculate_cost()` — Tính chi phí USD

```python
def calculate_cost(model, prompt_tokens, completion_tokens) -> float
```

Bảng giá được tích hợp sẵn:

| Model | Prompt ($/1K tokens) | Completion ($/1K tokens) |
|-------|---------------------|------------------------|
| gpt-4o | $0.005 | $0.015 |
| gpt-4o-mini | $0.00015 | $0.0006 |
| gpt-4-turbo | $0.01 | $0.03 |

Ví dụ thực tế: 1 lần gọi agent với 150 prompt + 60 completion tokens tốn khoảng $0.0016.

#### B. `AgentTracker` class — Wrapper MLflow

Thiết kế với 3 nguyên tắc:

**1. Graceful degradation** — nếu MLflow không cài hoặc server down, toàn bộ method trở thành no-op, không crash workflow chính:
```python
tracker = AgentTracker()
# MLflow không cài → tracker.enabled = False
# Gọi tracker.log_metrics(...) → không làm gì, không crash
```

**2. Flexible tracking URI** — tự động chọn storage:
```
MLFLOW_TRACKING_URI không set → lưu local ./mlruns
MLFLOW_TRACKING_URI=http://server → gửi lên MLflow server
MLFLOW_TRACKING_URI=https://dagshub.com/... → DagsHub cloud
```

**3. Context manager support** — tự động end run dù có exception:
```python
with AgentTracker() as tracker:
    tracker.start_run("gpt-4o_v1")
    result = run_agent_workflow(query, tracker=tracker)
# Tự động end_run("FINISHED") hoặc end_run("FAILED")
```

**Các method chính:**

| Method | Log gì |
|--------|--------|
| `log_params()` | model, prompt_version, temperature, tool_choice |
| `log_metrics()` | prompt_tokens, completion_tokens, total_tokens, latency_ms, cost_usd, used_tool |
| `log_eval_scores()` | qa_correctness, hallucination, toxicity (prefix "eval_") |
| `log_text()` | nội dung query.txt, response.txt |
| `log_dict()` | run_summary.json đầy đủ |

---

### Cập nhật `src/agent.py`

Thêm 3 tham số mới, hoàn toàn backward-compatible:

```python
# Signature mới
def run_agent_workflow(
    user_query: str,
    model: str = "gpt-4o",           # có thể thay bằng gpt-4o-mini
    prompt_version: str = "v1",       # track version prompt
    temperature: float = 0.0,         # có thể tune
    tracker: AgentTracker | None = None,  # None = không track
) -> dict:
    # Return thêm 2 fields mới
    return {
        "query": ...,
        "response": ...,
        "token_usage": {...},
        "latency_ms": 1240.5,    # ← mới
        "cost_usd": 0.0016,      # ← mới
    }
```

---

### Giá trị thực tế của MLflow

Khi SLABAI phát triển AI Coach thật, MLflow cho phép so sánh experiments có hệ thống:

```
MLflow UI hiển thị:
┌────────────┬──────────┬───────┬───────┬──────────┬────────────┐
│ Run        │ Prompt   │ Model │ Temp  │ Cost $   │ QA Score   │
├────────────┼──────────┼───────┼───────┼──────────┼────────────┤
│ run_001    │ v1       │ 4o    │ 0.0   │ 0.045    │ 0.72       │
│ run_002    │ v2       │ 4o    │ 0.0   │ 0.062    │ 0.89       │
│ run_003    │ v2       │ mini  │ 0.3   │ 0.004    │ 0.85 ✓best │
└────────────┴──────────┴───────┴───────┴──────────┴────────────┘
→ Kết luận: prompt_v2 + gpt-4o-mini tiết kiệm 90% chi phí, chất lượng gần bằng
```

---

## II. CI Pipeline — GitHub Actions

### Bối cảnh & Vấn đề

Trước khi có CI, dự án không có cơ chế tự động kiểm tra. Rủi ro:
- Ai sửa `agent.py` sai → chỉ phát hiện khi chạy thật
- Ai xóa `mlflow` khỏi requirements → tracker crash production
- Guardrails bị bypass → lỗ hổng security
- Không ai biết coverage code là bao nhiêu

---

### Giải pháp — `.github/workflows/ci-agent.yml`

#### Trigger conditions

```yaml
on:
  push:
    paths: ["src/**", "tests/**", "requirements.txt"]
  pull_request:
    branches: [main]
```

CI **chỉ chạy khi thay đổi code liên quan** — không chạy khi sửa README hay frontend.

#### 4 Jobs theo thứ tự phụ thuộc

```
┌─────────────────────────────────────────────────────────┐
│                                                         │
│   Job 1: lint (~30s)                                    │
│   ruff check + ruff format                              │
│         │                                               │
│         ▼ pass                                          │
│   ┌─────────────────┐    ┌──────────────────────┐      │
│   │ Job 2: unit-test│    │ Job 3: guardrails    │      │
│   │ pytest 34 tests │    │ pytest 18 tests      │      │
│   │ + coverage.xml  │    │                      │      │
│   │ ~1 phút         │    │ ~30s                 │      │
│   └────────┬────────┘    └──────────────────────┘      │
│            │ pass                                       │
│            ▼                                            │
│   Job 4: mlflow-smoke (~1 phút)                        │
│   Verify AgentTracker chạy end-to-end                  │
│            │                                            │
│            ▼                                            │
│   Job 5: ci-summary                                     │
│   ✅ All pass → cho phép merge                          │
│   ❌ Any fail → block merge + thông báo job nào fail    │
└─────────────────────────────────────────────────────────┘
```

**Tổng thời gian:** ~2-3 phút (một số job chạy song song)

---

### 34 Unit Tests

#### `tests/test_agent.py` — 14 tests

**Nhóm `TestGetStockStatus` (6 tests)** — test hàm tra cứu sản phẩm:
- PROD-101 phải có stock=5, price=$1200
- PROD-102 phải có stock=0 (out-of-stock case)
- PROD-103 phải có stock=15
- PROD-999 không tồn tại → phải trả `error` key
- Return value phải là valid JSON string
- Mỗi sản phẩm phải có description không rỗng

**Nhóm `TestRunAgentWorkflow` (8 tests)** — dùng mock OpenAI, không tốn tiền API:
- Không có API key → phải raise `ValueError` rõ ràng
- LLM trả lời thẳng không qua tool → hoạt động đúng
- 2-turn tool call → token usage cộng đúng cả 2 lần gọi
- Result phải có đủ: `query`, `response`, `token_usage`, `latency_ms`, `cost_usd`
- Tham số `model` phải được forward xuống OpenAI client
- Tham số `temperature` phải được forward xuống OpenAI client
- Khi có tracker → phải gọi `log_params` và `log_metrics`
- `tracker=None` → không crash gì

#### `tests/test_mlflow_tracker.py` — 20 tests

**Nhóm `TestCalculateCost` (6 tests)**:
- 1000+1000 tokens gpt-4o = $0.020 chính xác
- gpt-4o-mini phải rẻ hơn gpt-4o ít nhất 10 lần
- 0 tokens = $0
- Model lạ → fallback gpt-4o pricing, không crash
- Kết quả phải là float, không âm

**Nhóm `TestAgentTrackerNoMLflow` (2 tests)**:
- MLflow không cài → `tracker.enabled = False`
- Khi disabled → tất cả 7 methods đều là no-op hoàn toàn

**Nhóm `TestAgentTrackerWithMLflow` (11 tests)**:
- MLflow có sẵn → tracker enabled
- `start_run()` trả về run_id
- `log_params()` gọi đúng MLflow API
- `log_metrics()` gọi đúng MLflow API
- `log_eval_scores()` tự prefix `eval_` vào tên metric
- `end_run()` gọi MLflow với status FINISHED
- `end_run("FAILED")` gọi MLflow với status FAILED
- `get_run_id()` trả None khi chưa start
- Context manager thành công → tự gọi `end_run("FINISHED")`
- Context manager có exception → tự gọi `end_run("FAILED")`
- MLflow server down → không crash workflow chính

**Nhóm `TestTrackerIntegrationWithAgent` (1 test)**:
- Chạy toàn bộ flow: agent workflow → tracker log params/metrics/artifacts

---

### Kết quả kiểm tra local

```
============================= test session starts =============================
collected 34 items

tests/test_agent.py::TestGetStockStatus::test_known_product_in_stock PASSED
tests/test_agent.py::TestGetStockStatus::test_known_product_out_of_stock PASSED
... (34 tests)

============================= 34 passed in 4.26s ==============================
```

---

## III. Tóm tắt những gì đã thay đổi

| | Trước | Sau |
|--|-------|-----|
| MLflow tracking | Không có | Đủ: params, metrics, cost, artifacts |
| Cost tracking | Không có | Tự tính USD theo model |
| Latency tracking | Không có | Đo ms mỗi lần chạy |
| CI workflow | 0 file | 1 workflow, 4 jobs tự động |
| Unit tests Python | 0 tests | 34 tests, 100% pass |
| Khi MLflow down | Crash | Graceful no-op |
| Khi thiếu API key | Lỗi mơ hồ | Raise ValueError rõ ràng |
| Model có thể thay đổi | Không | Có, qua parameter |

**Files tạo mới:** `src/mlflow_tracker.py`, `tests/test_agent.py`, `tests/test_mlflow_tracker.py`, `.github/workflows/ci-agent.yml`

**Files cập nhật:** `src/agent.py`, `requirements.txt`

**PR:** `https://github.com/AI20K-Build-Cohort-2/C2-App-038/pull/3`