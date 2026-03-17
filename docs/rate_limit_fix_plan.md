# arXiv / 文献检索限流问题 — 调研报告与修复方案

> Created: 2026-03-15
> Status: **DONE** — All 7 tasks completed, 1117/1117 tests passing
> Severity: High — 直接影响用户体验和 Pipeline 稳定性

---

## 1. 问题描述

Pipeline 中多个阶段需要通过 API 调用外部论文数据库（arXiv、Semantic Scholar、OpenAlex），在高频请求时遭遇 **HTTP 429 (Too Many Requests)** 限流错误，导致文献检索失败或降级。

**受影响阶段：**

| 阶段 | 功能 | API 调用量 | 严重程度 |
|------|------|-----------|---------|
| Stage 4 | 文献收集 | ~12 次 (6 query × 2 source) | **高** — 直接影响论文质量 |
| Stage 8 | 假设新颖性检查 | ~8-12 次 | 中 — 非阻塞 |
| Stage 23 | 引用验证 | ~40-50 次 | **高** — 最密集的 API 调用 |

---

## 2. 根因分析

### 2.1 代码层面定位

| 文件 | 问题 | 影响 |
|------|------|------|
| `researchclaw/literature/arxiv_client.py` | 无显式 HTTP 429 检测 — URLError/OSError 统一捕获，无法区分限流和真正的网络错误 | 限流时无法做针对性处理 |
| `researchclaw/literature/arxiv_client.py` | 无熔断器 (Circuit Breaker) — S2 有但 arXiv 没有 | 连续 429 时仍不停重试 |
| `researchclaw/literature/arxiv_client.py` | 未解析 `Retry-After` 响应头 | 服务器建议的等待时间被忽略 |
| `researchclaw/literature/semantic_scholar.py` | 虽有熔断器，但 Stage 23 的密集调用仍可能触发 | 一旦熔断，所有后续 S2 请求被跳过 |
| `researchclaw/literature/verify.py` | Stage 23 逐条顺序验证 40+ 引用，每条间隔 1.5s | 总耗时 60-80s，集中 burst 可触发限流 |
| `researchclaw/literature/search.py` | OpenAlex 仅用于 L3 title search fallback，未作为主搜索源 | 浪费了最宽松的 API 额度 |

### 2.2 各 API 官方限流策略

| API | 限制 | 我们当前的间隔 | 是否合规 |
|-----|------|---------------|---------|
| **arXiv** | 1 request / 3 seconds | 3.1s (`_RATE_LIMIT_SEC`) | 合规，但无 429 重试 |
| **Semantic Scholar** (无 Key) | 共享池 5000/5min | 1.5s | 偏保守但可行 |
| **Semantic Scholar** (有 Key) | 1 req/s (可申请更高) | 0.3s | 合规 |
| **OpenAlex** (有 Key) | 10,000 list/day; 1,000 search/day | 仅 L3 fallback | 远未用满 |
| **CrossRef** | 50 req/s (polite pool) | 1.5s | 远低于上限 |

### 2.3 arXiv 特殊性

- arXiv 元数据每天午夜更新一次 → 同一查询 24h 内重复请求无意义
- arXiv 返回 HTTP 200 但内容为空 (phantom empty page) 是已知 bug
- arXiv ToS 明确要求：所有你控制的机器合计不超过 1 req/3s

---

## 3. 竞品方案调研

### 3.1 PaperClaw (guhaohao0991/PaperClaw)

| 方面 | 实现 |
|------|------|
| arXiv 搜索 | `urllib.request` + 3s 固定延迟，**无重试，无 429 处理** |
| S2 搜索 | `requests` + `_request_with_retry()`: 指数退避 (2^attempt)，3 次重试 |
| S2 缓存 | 文件 JSON 缓存，按类型 TTL（论文 7 天，作者 30 天，引用 1 天） |
| arXiv 缓存 | **无** — 每次直接调 API |

**可借鉴**：S2 缓存按类型差异化 TTL 的思路。

### 3.2 Sibyl Research System (Sibyl-Research-Team/sibyl-research-system)

| 方面 | 实现 |
|------|------|
| arXiv 搜索 | 通过 `arxiv-mcp-server` MCP 工具，依赖 `arxiv` Python 库的 3s delay |
| 429 处理 | **代码层面无** — 依赖 CLAUDE.md 指令让 LLM agent 行为级重试 |
| 体量控制 | 刻意限制在 15-30 篇论文，"速度优先" |
| S2 | **未使用** |

**可借鉴**：控制搜索体量、不过度爬取的理念。

### 3.3 Idea2Paper (AgentAlphaAGI/Idea2Paper)

| 方面 | 实现 |
|------|------|
| 文献来源 | **完全离线** — 预构建 ICLR Knowledge Graph，运行时不调任何论文 API |
| 重试策略 | `urllib3.util.retry.Retry(total=3, backoff_factor=2, status_forcelist=[429,500,502,503,504])` |
| 降级策略 | Embedding 失败后降级为 Jaccard 相似度 |

**可借鉴**：`urllib3.Retry` 的干净实现；embedding 失败的优雅降级。

### 3.4 对比总结

| 项目 | arXiv 429 处理 | S2 429 处理 | 缓存 | 备选源 |
|------|---------------|-------------|------|--------|
| **AutoResearchClaw (当前)** | 无显式处理 | 熔断器 ✓ | 7 天 TTL ✓ | OpenAlex (仅 L3) |
| PaperClaw | 无 | 指数退避 | S2 有/arXiv 无 | 无 |
| Sibyl | 无 (靠 LLM) | 未使用 | 论文下载缓存 | WebSearch |
| Idea2Paper | 不涉及 | 不涉及 | 离线 KG | 不涉及 |

---

## 4. 综合解决方案

### 4.1 方案总览

```
┌─────────────────────────────────────────────────────┐
│                   防御层次                            │
├─────────────────────────────────────────────────────┤
│ L1: 智能限速器 (Adaptive Rate Limiter)              │
│     - 根据 API 类型自动调节请求间隔                   │
│     - 解析 Retry-After 响应头                        │
│                                                     │
│ L2: 熔断器 (Circuit Breaker)                        │
│     - arXiv 也加熔断器 (参考 S2 实现)                 │
│     - 三态切换: CLOSED → OPEN → HALF_OPEN            │
│                                                     │
│ L3: 多源降级 (Source Fallback)                       │
│     - arXiv 限流 → 切换 OpenAlex/S2                  │
│     - S2 限流 → 切换 OpenAlex/arXiv                  │
│     - 全部限流 → 返回缓存结果                         │
│                                                     │
│ L4: 结果缓存 (Cache Layer)                          │
│     - 24h TTL for arXiv (每天只更新一次)              │
│     - 差异化 TTL (论文元数据 vs 搜索结果)              │
│     - 引用验证结果永久缓存                            │
│                                                     │
│ L5: 请求优化 (Request Optimization)                 │
│     - S2 batch API 批量查询                          │
│     - 合并重复查询                                   │
│     - OpenAlex 提升为主搜索源                         │
└─────────────────────────────────────────────────────┘
```

---

## 5. 实施任务列表

### Task 1: arXiv 客户端增强 — 显式 429 处理 + 熔断器

**文件**: `researchclaw/literature/arxiv_client.py`

**改动**:
- [x]1.1 改用 `urllib.request.urlopen` 的 `HTTPError` 子类捕获，区分 429 和其他错误
- [x]1.2 解析 `Retry-After` 响应头，优先使用服务器建议的等待时间
- [x]1.3 添加 arXiv 熔断器（复用 S2 的三态模式）: 3 次连续 429 → OPEN (180s cooldown)
- [x]1.4 增加 `_RATE_LIMIT_SEC` 动态调整: 收到 429 后临时提升到 5s，成功后恢复 3.1s

**预期效果**: arXiv 429 错误从"静默失败/重试 3 次放弃"变为"智能等待 + 熔断保护"

### Task 2: OpenAlex 提升为主搜索源

**文件**: `researchclaw/literature/search.py`, 新建 `researchclaw/literature/openalex_client.py`

**改动**:
- [x]2.1 新建 `openalex_client.py`: 封装 OpenAlex Works API (`https://api.openalex.org/works`)
  - 支持 `title.search` / `default.search` 两种查询模式
  - 字段映射到 `Paper` 数据类 (title, abstract, year, venue, citation_count, authors, doi, arxiv_id)
  - 配置 polite pool email (`researchclaw@users.noreply.github.com`)
  - 指数退避 + 3 次重试
- [x]2.2 在 `search.py` 的 `search_papers()` 中注册 OpenAlex 为第三个源
- [x]2.3 调整 `search_papers_multi_query()` 的源顺序策略:
  - 默认: OpenAlex → Semantic Scholar → arXiv
  - 任一源 429 → 跳过该源，增加其他源的 limit
- [x]2.4 在 `config.researchclaw.example.yaml` 中添加 `openalex_email` 配置项

**预期效果**: 文献检索默认走 OpenAlex (10K/day)，arXiv 和 S2 作为补充和验证，大幅降低 429 风险

### Task 3: 搜索结果缓存增强

**文件**: `researchclaw/literature/cache.py`

**改动**:
- [x]3.1 arXiv 搜索结果 TTL 从 7 天改为 24 小时（arXiv 每天午夜更新一次）
- [x]3.2 添加"源级别"缓存策略: 如果 arXiv 缓存存在且 <24h，直接返回而不请求 API
- [x]3.3 缓存命中时记录日志 `[cache] HIT query=... source=... age=...`

**预期效果**: 同一 Pipeline 运行中不会重复请求同一查询，跨运行也可复用 24h 内的结果

### Task 4: S2 batch API + 去重优化

**文件**: `researchclaw/literature/semantic_scholar.py`

**改动**:
- [x]4.1 新增 `batch_fetch_papers(paper_ids: list[str]) -> list[Paper]`
  - 使用 `POST /graph/v1/paper/batch` 端点
  - 一次最多 500 个 ID（S2 限制）
  - 单次请求替代 N 次请求
- [x]4.2 在 Stage 23 引用验证中使用 batch API: 先收集所有有 S2 ID 的引用，一次性批量获取

**预期效果**: Stage 23 的 S2 API 调用从 ~20 次降至 1-2 次

### Task 5: Stage 23 引用验证并行化 + 智能调度

**文件**: `researchclaw/literature/verify.py`

**改动**:
- [x]5.1 按源分组，相同源的验证串行（遵守限速），不同源的验证并行
  - L1 (arXiv) 和 L2 (CrossRef/DataCite) 和 L3 (OpenAlex) 可并行
- [x]5.2 引用验证缓存标记为"永久有效"（已验证的论文不会变）
- [x]5.3 优先使用 DOI → OpenAlex 验证（比 arXiv API 限制宽松得多），L1 arXiv 降为备选

**预期效果**: Stage 23 耗时从 60-80s 降至 20-30s，arXiv API 调用减少 50%+

### Task 6: 用户反馈 + 日志改善

**文件**: `researchclaw/pipeline/executor.py`, `researchclaw/literature/search.py`

**改动**:
- [x]6.1 文献检索阶段添加进度日志: `[literature] Searching OpenAlex... (1/3 sources)`
- [x]6.2 429 错误时输出友好提示: `[rate-limit] arXiv rate limit hit, switching to OpenAlex...`
- [x]6.3 最终搜索统计: `[literature] Found 47 papers (OpenAlex: 28, S2: 12, arXiv: 7, cache: 23 hits)`

**预期效果**: 用户能清楚看到搜索进度和限流处理过程，不再困惑

### Task 7: 测试覆盖

**文件**: `tests/test_rc_literature.py`

**改动**:
- [x]7.1 arXiv 429 + Retry-After header 解析测试
- [x]7.2 arXiv 熔断器三态切换测试
- [x]7.3 OpenAlex 客户端正常搜索 + 429 退避测试
- [x]7.4 多源降级测试: 模拟 arXiv 429 → 自动切换到 OpenAlex
- [x]7.5 S2 batch API 测试
- [x]7.6 缓存 24h TTL 测试

---

## 6. 优先级排序

| 优先级 | 任务 | 理由 | 预计工时 |
|--------|------|------|---------|
| **P0** | Task 1: arXiv 429 显式处理 + 熔断器 | 直接修复当前 crash 问题 | 30min |
| **P0** | Task 3: 缓存 TTL 调整 | 零成本减少请求量 | 15min |
| **P1** | Task 2: OpenAlex 主搜索源 | 根本性降低 arXiv 依赖 | 1.5h |
| **P1** | Task 6: 用户反馈日志 | 提升用户体验 | 20min |
| **P2** | Task 4: S2 batch API | 优化 Stage 23 | 45min |
| **P2** | Task 5: Stage 23 并行化 | 性能优化 | 1h |
| **P3** | Task 7: 测试覆盖 | 质量保障 | 1h |

**总计**: ~5 小时

---

## 7. 实施进度

| 任务 | 状态 | 完成时间 | 备注 |
|------|------|---------|------|
| Task 1: arXiv 429 + 熔断器 | [x] 完成 | 2026-03-15 | 三态熔断器 + Retry-After 解析 + 动态限速 |
| Task 2: OpenAlex 主搜索源 | [x] 完成 | 2026-03-15 | 新建 openalex_client.py，搜索源顺序 OA→S2→arXiv |
| Task 3: 缓存增强 | [x] 完成 | 2026-03-15 | 按源差异化 TTL (arXiv 24h, S2/OA 3d, verify 永久) |
| Task 4: S2 batch API | [x] 完成 | 2026-03-15 | batch_fetch_papers() POST 批量端点，500 ID/batch |
| Task 5: Stage 23 优化 | [x] 完成 | 2026-03-15 | 验证顺序 DOI→CrossRef→OpenAlex→arXiv→S2，差异化延迟 |
| Task 6: 用户反馈日志 | [x] 完成 | 2026-03-15 | Stage 4/23 进度日志 + 源统计 |
| Task 7: 测试覆盖 | [x] 完成 | 2026-03-15 | +14 新测试 (熔断器×5, OpenAlex×4, 降级×1, TTL×2, 现有修复×2) |

---

## 附录 A: API 速率限制速查表

| API | 端点 | 免费限制 | 认证限制 | 我们的使用量 |
|-----|------|---------|---------|------------|
| arXiv | `export.arxiv.org/api/query` | 1 req / 3s | 无认证选项 | ~20 req/run |
| Semantic Scholar | `api.semanticscholar.org/graph/v1` | 共享池 5K/5min | 1 req/s (API key) | ~30 req/run |
| OpenAlex | `api.openalex.org/works` | 10K list/day, 1K search/day | 同左 (polite pool) | 待启用 |
| CrossRef | `api.crossref.org/works` | 50 req/s (polite) | 同左 | ~15 req/run |
| DataCite | `api.datacite.org/dois` | 无明确限制 | — | ~5 req/run |

## 附录 B: 参考实现

### arXiv 显式 429 处理（目标代码）

```python
try:
    resp = urllib.request.urlopen(req, timeout=_TIMEOUT_SEC)
except urllib.error.HTTPError as exc:
    if exc.code == 429:
        retry_after = exc.headers.get("Retry-After")
        wait = int(retry_after) if retry_after else _RATE_LIMIT_SEC * (2 ** attempt)
        _cb_on_429()  # 通知熔断器
        time.sleep(wait + random.uniform(0, wait * 0.2))
        continue
    raise
```

### OpenAlex 搜索客户端（目标签名）

```python
def search_openalex(
    query: str,
    limit: int = 50,
    year_min: int | None = None,
    email: str = "researchclaw@users.noreply.github.com",
) -> list[Paper]:
    """Search OpenAlex Works API with polite pool access."""
    ...
```

### S2 Batch API（目标签名）

```python
def batch_fetch_papers(
    paper_ids: list[str],
    fields: str = "paperId,title,abstract,year,venue,citationCount,authors,externalIds",
) -> list[dict]:
    """Batch fetch paper details via POST /graph/v1/paper/batch."""
    ...
```
