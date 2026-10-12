# 更新日志

本文件记录**交付历史与验收证据**：每个 Part 交付了什么、怎么验证的、与上一版相比变了什么。
稳定性约定（架构、写法、工作流、验收门槛）在 `AGENTS.md`；与教程的偏离账本在 `docs/deviations.md`。

格式参考 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)。

---

## [未发布]

### P4 — 7.5 工具系统（Tool 基类 / 注册表 / 内置工具）

新增：

- `tools/base.py`：`ToolType`、`ToolParameter`、`Tool`、`FunctionTool`。
  `Tool.run` 是唯一的抽象同步实现；默认 `Tool.arun` 经 `asyncio.to_thread` 落到 `run`（同步工具不阻塞事件循环）；
  `Tool.validate` 统一执行「未声明参数 / 缺必填 / 类型折算」；`Tool.to_openai_schema` 导出 function calling 结构。
  `FunctionTool` 用 `inspect.signature` 推断参数声明，并识别 PEP 563 下的字符串注解（`"int"`、`"list[str]"`、`"int | None"`）。
- `tools/registry.py`：`ToolRegistry` —— `register_tool` / `register_function` / `unregister` / `get_tool` /
  `list_tools` / `get_tools_description` / `to_openai_schemas` / `execute_tool` / `aexecute_tool`。
  输入统一为「恰好一个必填参数 → 裸字符串；多参数 → 参数字典」；未知工具、未声明参数、缺必填、类型不符、
  工具内部异常一律抛携带定位信息的 `ToolException`。
- `tools/builtin/calculator.py`：`CalculatorTool` —— 白名单 AST 求值（四则 / 整除 / 取模 / 幂 / 一元运算、
  17 个数学函数、`pi` / `e` / `tau` / `inf`），拒绝非白名单节点、复数结果、定义域错误、超限幂指数与阶乘入参；
  整数值 float 归一为整数文本。
- `tools/builtin/search.py`：`SearchTool` —— Tavily / SerpAPI REST（httpx 同步客户端复用连接池），
  `hybrid` 降级链与 `attempts` 失败详情、`limit` 默认 3、无密钥时给出配置指引。

### P5 — 7.5.4 工具链与并发执行器

新增：

- `tools/chain.py`：`ToolStep`（冻结 dataclass）、`ToolChain`（`add_step` / 只读 `steps` / `execute` / `aexecute`）、
  `ToolChainManager`（注册 / 注销 / 查询 / 同步与异步执行）；模板变量缺失抛 `ToolException`。
- `tools/async_executor.py`：`AsyncToolExecutor` —— 并发上限（信号量按当前事件循环惰性创建，跨 `asyncio.run` 复用安全）、
  保序 `gather`、`return_exceptions`、显式 `aclose()` / `async with`。

### P3 — 7.4 四种 Agent 范式

新增：

- `agents/simple_agent.py`：`SimpleAgent`（可选工具多轮循环、`astream` / `stream_run`、`add_tool` 等便利方法）。
- `agents/react_agent.py`：`ReActAgent` + `ReActResult` / `ReActStep` / `DEFAULT_REACT_PROMPT`
  （执行历史随调用传递，结构化 `stop_reason` ∈ finished / max_steps / no_action）。
- `agents/reflection_agent.py`：`ReflectionAgent` + `DEFAULT_PROMPTS`（initial / reflect / refine）。
- `agents/plan_solve_agent.py`：`PlanAndSolveAgent` + `PlanAndSolveResult` + 默认规划器/执行器提示词。
- 导出：`myagent`、`myagent.agents`、`myagent.tools` 均导出上述公开名。

变更：

- 相对教程的逐条偏离：`D19`~`D28`（见 `docs/deviations.md`）。

演示脚本：

- `examples/ch7/02_simple_agent.py`~`07_tool_chain.py` 共六个：02~05 为面向真实服务的交互式演示
  （工具对话 / ReAct 轨迹 / 反思循环 / 计划执行），06 为工具注册表展示，07 为工具链 + 并发执行器；
  均遵循 §5.6：顶部注明启动命令、真实 stdin 交互、非 tty 读到 EOF 正常退出。

### 演示脚本交互化（P1 调整）

变更：

- `examples/ch7/01_llm_demo.py`：硬编码两次提问的演示 → **交互式对话**：stdin 多轮提问、流式输出并保留上下文；
  `/invoke <问题>` 走非流式路径作对照；空行跳过；`exit` / `quit` / Ctrl-D 退出；文件顶部注释启动命令。
- `AGENTS.md` §5.6：增补演示脚本约定 —— 顶部注释启动命令、需要人参与的场景必须真实 stdin 交互
  （禁硬编码问题冒充交互）、非 tty 下可读到 EOF 正常退出。
- `README.md`：演示脚本脚注改为「交互式对话演示」。

### 修复：等待响应期间误敲的输入被当成提问自动提交

- `examples/ch7/01_llm_demo.py`：新增 `_muted_stdin()` —— 请求模型期间关闭 tty 的 `ECHO` / `ICANON`，
  响应结束后恢复终端状态并 `tcflush` 丢弃等待期误敲的字符；非 tty 或缺少 `termios` 的平台下为 no-op。
  修复症状：等待模型响应时敲下的字符由 tty 直接回显、穿插进流式输出（看起来"删不掉"），
  并在下一轮 `input()` 时被立刻吃掉、作为新提问自动提交（幽灵提问）。
- 验收证据（真服务 **deepseek / deepseek-flash**，非 stub）：
  - pty 复现（修复前）：流式期间敲 `GUHOST\r` → 输出出现该回显，且脚本自动追加一轮 `GUHOST` 提问。
  - pty 验证（修复后）：同一操作 → 输出不含 `GUHOST`、无自动提问；`/invoke` 非流式路径同样成立；
    `exit` 退出码 0。
  - 非 tty：`printf 'exit\n' | uv run python -X utf8 examples/ch7/01_llm_demo.py` 退出码 0。
  - `ruff format --check` / `ruff check` / `mypy src tests examples` / `pytest`（97 passed）全绿。

### P2 — 7.3 框架接口实现（Message / Config / Agent 基类）

新增：

- `core/message.py`：`Message`、`MessageRole`（`dataclass`，UTC 时间戳）。
- `core/config.py`：`Config`（含字段校验）、`Config.from_env()`（默认读 `.env`）、`configure_logging()`、
  `env_int` / `env_float` / `env_bool`。
- `core/agent.py`：`Agent` 抽象基类，`arun()` 为唯一真实入口，`run()` 为其同步包装；
  `add_message` 按 `config.max_history_length` 裁剪历史（教程中该配置从未接线）。
- `core/exceptions.py`：9 个异常类，携带 `context` 定位信息与 `retryable` 标记。
- `core/_sync.py`：`run_sync` / `ensure_no_running_loop`（教程无对应文件）。
- 包导出：`myagent` 与 `myagent.core` 均导出上述公开名。

变更：

- 教程 `core/*.py` 的 pydantic v1 写法 → `dataclass`；空 `pass` 异常类 → 带上下文的分层异常。
- 相对教程的逐条偏离：`D2`、`D4`、`D5`、`D10`、`D11`、`D16`、`D17`、`D18`（见 `docs/deviations.md`）。

### P1 — 7.2 HelloAgentsLLM 扩展

新增：

- `core/llm.py`：`ProviderSpec` + 12 个 provider 注册表（openai / deepseek / qwen / modelscope / kimi /
  zhipu / ollama / vllm / local / anthropic / auto / custom）；
  `detect_provider()` 按正文 7.2.3 优先级（专属环境变量 → base_url → 密钥格式）、
  `resolve_credentials()`、`default_model_for()`。
- 两条协议路径：OpenAI 兼容（`/chat/completions`）与 Anthropic（`/v1/messages`）；
  SSE 流式解析、非 SSE 响应回退、HTTP 状态码 → 异常分类映射。
- 对外方法：`athink` / `ainvoke`（真实实现），`think` / `invoke` / `stream_invoke`（教程同签名的同步包装，
  流式包装在同步专用事件循环里逐步驱动，仍为真实增量输出）。
- `examples/ch7/01_llm_demo.py`：provider 自动检测 + 流式/非流式调用的可运行演示。

修复：

- **同步接口跨调用抛 `RuntimeError: Event loop is closed`**（首次真服务调用暴露，非环境问题）：
  httpx 连接池绑定在创建连接的事件循环上，而 `think` / `invoke` 原先每次调用都 `asyncio.run`
  新建循环，第二次调用即复用/关闭属于已死循环的连接。
  现改为同步入口独占一个长期复用的事件循环与客户端（`_sync_resources()`），
  `close()` 在同步上下文回收它，`aclose()` 只关异步池并在混用时给出告警；
  取消 `stream_invoke` 的中断路径也不会关闭循环（只关本次响应）。
  回归测试：`test_sync_calls_reuse_one_loop_and_one_pool`、
  `test_sync_pool_closed_by_close_and_unused_async_pool_alongside`、
  `test_async_pool_closed_by_aclose_without_touching_sync_pool`、
  `test_close_after_async_use_warns_instead_of_crossing_loops`。

变更：

- 相对教程的逐条偏离：`D1`、`D3`、`D14`、`D15`（见 `docs/deviations.md`）。

### 基础设施

新增：

- `tests/`：`conftest.py` 提供脚本化本地 HTTP+SSE 服务夹具；`test_llm_client` / `test_llm_provider` /
  `test_agent` / `test_config` / `test_message` 共 92 项测试，全部离线可跑。
- `docs/deviations.md`：与教程的偏离账本。
- `reference/hello-agents/`：教程只读快照（`code/chapter4`、`code/chapter7`、`docs/chapter7`，
  commit `78dcd8f4821c96ba52ed60d72d523a7173b0206a`，来源见其 `SOURCE.md`）。
- `tests/`、`examples/ch7/`、`.gitignore`（忽略 `.env` 与各缓存目录）。

变更：

- `pyproject.toml`：运行时依赖 `httpx` + `python-dotenv`；开发依赖 `ruff` / `mypy` / `pytest` /
  `pytest-asyncio`；ruff 行宽 100、目标 py312、`extend-exclude = ["reference"]`；mypy `disallow_untyped_defs`；
  pytest `asyncio_mode = "auto"`。**移除**失效的 `[project.scripts] myagent = "myagent:main"`。
- 包布局迁移：`src/{core,agents,tools}` → `src/myagent/{core,agents,tools}`（适配 `uv_build`）。
- `README.md`：由空文件补为项目入口（定位、命令、进度指向）。

---

## 验收证据

### P1 + P2（2026-10-12）

命令与结果：

```
uv run ruff format --check .   → 33 files already formatted
uv run ruff check .            → All checks passed!
uv run mypy src tests examples → Success: no issues found in 29 source files
uv run pytest -q               → 97 passed
uv build --no-sources          → tar.gz + whl 构建成功（wheel 30 条目，无 tests/examples/reference 泄漏）
```

真服务（`deepseek`，`https://api.deepseek.com`，模型 `deepseek-flash`）：

```
uv run python -X utf8 examples/ch7/01_llm_demo.py
检测到的 provider: deepseek
--- 流式调用 think() ---
ReAct 智能体是一种让大模型交替进行“推理—行动—观察”循环、从而借助外部工具完成复杂任务的智能体范式。
--- 非流式调用 invoke() ---
ReAct 智能体是一种让大模型交替进行“推理”和“行动”的智能体范式，通过思考后调用工具、再根据结果继续推理来完成任务。
exit=0
```

离线（无 key）：

```
LLM_MODEL_ID=stub-model LLM_API_KEY=test-key LLM_BASE_URL=http://127.0.0.1:8917/v1 \
  uv run python -X utf8 examples/ch7/01_llm_demo.py    → provider 判为 local，流式与非流式均正确
```

**覆盖边界**：真实 provider 只验证了 `deepseek` 一家（含流式、非流式、连接复用）。
`anthropic` / `openai` / `qwen` / `kimi` / `zhipu` / `modelscope` / `ollama` / `vllm` 的默认
`base_url` 与默认模型名**未经真实服务核验**；离线测试覆盖的是协议与错误映射，不覆盖各家默认值是否正确。

### 演示脚本交互化（2026-10-12）

真服务（`deepseek`，`https://api.deepseek.com`，模型 `deepseek-flash`，仓库根 `.env` 的真实 key）；
用管道模拟非 tty 输入：

```
printf '用一句话解释什么是 ReAct 智能体\n/invoke 只回复 OK\n\n你好，你叫什么\nexit\n' \
  | uv run python -X utf8 examples/ch7/01_llm_demo.py
```

```
检测到的 provider: deepseek
model=deepseek-flash base_url=https://api.deepseek.com timeout=60.0
输入问题开始对话；/invoke <问题> 走非流式；exit 退出。

你: 助手: ReAct 智能体是一种将“推理（Reasoning）”与“行动（Acting）”交替结合，通过边思考边调用工具来逐步完成任务的 AI 代理。
你: 助手: OK
你: 你: 助手: 你好，我是 AI 助手，没有固定名字，你可以叫我助手。
你: 
```

覆盖：流式与 `/invoke` 非流式两条调用路径、空行跳过、多轮上下文、`exit` 退出、
管道（非 tty）读到 EOF 正常退出。`uv run ruff format --check .` / `uv run ruff check .` /
`uv run mypy src tests examples` / `uv run pytest` 全部通过（97 passed）。

### P3 + P4 + P5（2026-10-12）

命令与结果：

```
uv run ruff format --check .   → 44 files already formatted
uv run ruff check .            → All checks passed!
uv run mypy src tests examples → Success: no issues found in 40 source files
uv run pytest -q               → 236 passed
```

离线测试新增 139 项（共 236）：`tests/test_tools_registry.py`（参数声明/折算/校验、单一调用路径、
异常语义、容器协议）、`tests/test_tools_builtin.py`（计算器白名单与规模保护、搜索的请求就位与降级链，
用 `httpx.MockTransport`）、`tests/test_tools_chain.py`（模板渲染、只读步骤、错误路径、同步/异步一致）、
`tests/test_tools_async_executor.py`（保序、真实并发、并发上限、跨事件循环复用）、
`tests/test_agents.py`（四种范式的循环、工具回灌、停止原因、错误路径，全部走本地 stub LLM 服务）。

真服务（`deepseek`，`https://api.deepseek.com`，模型 `deepseek-flash`，仓库根 `.env` 的真实 key），
管道模拟非 tty 输入：

```
printf '计算 (25 + 15) * 3 - 8 等于多少？\nexit\n' | uv run python -X utf8 examples/ch7/02_simple_agent.py
  → agent=simple 调用工具 calculator((25 + 15) * 3 - 8)；回答 112；exit 0

printf '请帮我计算 (25 + 15) * 3 - 8\nexit\n' | uv run python -X utf8 examples/ch7/03_react_agent.py
  → stop_reason = finished
    [第 1 步] action: calculator[(25 + 15) * 3 - 8]  observation: 112
    [第 2 步] action: Finish[112]   最终答案: 112；exit 0

printf '用三句话说明为什么苏格拉底式提问有助于思考。\nexit\n' | uv run python -X utf8 examples/ch7/04_reflection_agent.py
  → 反思循环后输出三条编号论述；exit 0

printf '小明有 12 颗糖…每人几颗？\nexit\n' | uv run python -X utf8 examples/ch7/05_plan_solve_agent.py
  → 5 步计划 + 各步结果（12 - 5 = 7 → 7 + 8 = 15 → 15 ÷ 3 = 5）；最终答案: 每人分到 5 颗糖；exit 0
```

无 LLM 的两支（不需要 `.env`）：

```
printf '2 + 3\nsqrt(16) + 2 * 3\nbogus$$\n/shout hello world\nexit\n' \
  | uv run python -X utf8 examples/ch7/06_tools.py      → 5 / 10 / 友好报错 / HELLO WORLD（共 11 个字符）；exit 0

printf '5 + 3\n2 ** 10\nexit\n' | uv run python -X utf8 examples/ch7/07_tool_chain.py
  → 链中间结果 value=8 → squared=64；1024 → 1048576；
    并行演示 4 个 0.25s 任务：串行 1.019s vs 并行 0.254s；exit 0
```

**覆盖边界**：真实 provider 仍只验证了 `deepseek` 一家（02~05 各跑 1 轮，见上）；
`SearchTool` 的 Tavily / SerpAPI 路径只用 `httpx.MockTransport` 做过离线断言，
未用真实搜索密钥跑通（`.env` 未配置 `TAVILY_API_KEY` / `SERPAPI_API_KEY`），
`06` / `07` 的搜索分支因此在实跑中走的是「未配置密钥」提示路径。
