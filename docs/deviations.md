# 与教程的偏离清单

**格式**：每个有意偏离一行一条，四要素齐全 —— 位置 / 教程写法 / 本项目写法 / 原因。缺任何一项的条目视为未完成。

**规则**：

- 教程侧参照 `reference/hello-agents/`（commit 见其 `SOURCE.md`）。
- 有意偏离**必须**在实现该 Part 的同一轮里补录，不得事后补。
- 教程缺陷改法也算偏离，即使"改法显然更好"。
- 不是偏离的：教程没实现而我们新增的内容（例如测试、`examples/`），无需记录。

---

## A. 结构性偏离

| # | 位置 | 教程写法 | 本项目写法 | 原因 |
|---|---|---|---|---|
| D1 | 全局 | 教程要求 Python >= 3.10 | `requires-python = ">=3.12"` | 使用 `X \| None`、`list[...]` 等现代写法；教程自身的过时注解将被 ruff `UP` 规则拦住 |
| D2 | `core/message.py`、`core/config.py` | `pydantic.BaseModel` + `.dict()` | `@dataclass` + 显式 `to_dict()` | 避免 pydantic v1→v2 迁移陷阱与运行时依赖；构造调用形式保持 `Message(content, role)`、`Config.from_env()` 不变 |
| D3 | `core/llm.py` | `openai.OpenAI` / `openai.AsyncOpenAI` | `httpx.AsyncClient` + provider 适配层 | 同一客户端覆盖 OpenAI 兼容 / Anthropic / 本地推理；运行时不引入 openai SDK |
| D4 | 全部 Agent | 同步 `run()`（教程示例全部同步调用） | `arun()` 为唯一真实实现，`run()` 为同步包装（运行中的事件循环内调用则明确抛错） | 同时满足"对外 API 对齐教程"与"async 真实链路"两个既定目标 |
| D5 | 全部模块 | `print()` 兼作日志与用户输出 | `logging` 记录诊断；`examples/` 里的 `print` 只做面向人的结果输出 | 教程的 emoji 进度打印会污染 stdout，无法被库使用者过滤 |
| D6 | `tools/registry.py`、`agents/*` | 调用工具存在两条路径：`registry.execute_tool()` 与 `tool.run()` 混用（见 `reference/.../my_simple_agent.py` 的 `_execute_tool_call`） | 唯一对外路径是 `ToolRegistry.execute_tool()` | 双路径导致参数解析与错误处理在同一工具上行为不一致 |
| D7 | `agents/simple_agent.py` | 按工具名硬编码参数解析分支（`if tool_name == 'calculator'` / `'search'` / `'memory'`） | 参数校验与解析由工具自身声明，registry 统一调度 | 硬编码使新增工具必须改动 Agent 代码 |
| D8 | `agents/simple_agent.py` 等 | 裸 `dict` 拼 messages（`{"role": ..., "content": ...}`） | 统一 `Message` 对象，仅在 provider 边界转协议格式 | 教程中裸 dict 绕过 `Message`，导致 timestamp/metadata 丢失 |
| D9 | `agents/react_agent.py` | 执行历史存于实例态 `self.current_history`；步数用尽返回固定文案 | 执行历史随单次调用传递；步数用尽返回结构化 `stop_reason` | 实例态使同一 Agent 并发调用互相污染 |
| D10 | `core/exceptions.py` | 5 个空 `pass` 异常类 | 同名异常携带定位上下文（provider/tool/model），并区分可重试与不可重试 | 教程异常无信息量，调用方无法处置 |
| D11 | `core/config.py` | `max_history_length` 定义后从未被任何代码读取 | 真正接线：`Agent.add_message` 超出上限即丢弃最早消息 | 未接线的配置项会误导使用者（已核对教程 `core/agent.py` 从未读取它） |
| D12 | `agents/*_agent.py` | 工具调用走文本协议 `[TOOL_CALL:name:args]` 正则解析 | **保留**文本协议（v0.1.1 形态即如此），仅修 D6/D7 的双路径与硬编码 | 教程 v0.1.1 基线用文本协议；原生 Function Calling 属 0.2.8 后的 `FunctionCallAgent`，不在 P1~P5 范围 |
| D14 | `core/llm.py` | 教程 7.2 只覆盖 OpenAI 兼容族（openai / deepseek / qwen / modelscope / kimi / zhipu / ollama / vllm / local） | 新增 Anthropic 适配器：`/v1/messages`、`x-api-key` + `anthropic-version` 头、`system` 独立于 messages、`content_block_delta` 流式事件、`max_tokens` 必填 | 你在第一轮明确选择"多 provider 抽象（anthropic + openai + 本地）"；协议差异收敛在 `ProviderSpec.protocol`、`_build_payload` 与 `_extract_delta` |
| D15 | `core/llm.py::detect_provider` | 正文 7.2.3 的优先级为"专属环境变量 → base_url → 密钥格式"，但框架参考实现把密钥格式排在 base_url 之前 | 按**正文**顺序实现（base_url 先于密钥格式） | 已定基线为"教程正文为准"；两种顺序在 base_url 与密钥指向不同 provider 时结果不同 |
| D16 | `core/config.py` | `default_model="gpt-3.5-turbo"`、`default_provider="openai"` | `default_model=None`（交由 provider 推断默认模型）、`default_provider="auto"` | `gpt-3.5-turbo` 已过时；硬编码 openai 与"自动检测 provider"目标冲突 |
| D17 | `core/_sync.py` | 教程无此文件（其 API 全部同步） | 新增 `run_sync` / `ensure_no_running_loop`，供 `run` / `think` / `invoke` 包装异步实现 | D4 的同步包装需要统一的"禁止在运行中的事件循环内调用"检查，避免两处各写一遍 |
| D18 | `core/config.py::Config.from_env` | 不读 `.env`，由各脚本自行 `load_dotenv()` | 默认先 `load_dotenv(override=False)`，可用 `load_dotenv_file=False` 关闭 | 免去每个入口重复样板；`override=False` 保证不覆盖已有环境变量 |
| D19 | `tools/base.py` | 7.5.1 `ToolParameter(BaseModel)`（pydantic） | `@dataclass`，并在 `__post_init__` 校验 `required` 与 `default` 不能同时给出 | 与 D2 同因；顺便把「必填但给了默认值」这种自相矛盾的声明挡在构造期 |
| D20 | `tools/base.py`、`tools/registry.py` | 7.5.1 只有同步 `Tool.run`，`ToolRegistry.execute_tool` 也是同步 | 新增 `Tool.arun`（默认 `asyncio.to_thread(self.run, ...)`）与 `ToolRegistry.aexecute_tool`；同步与异步各一条公开路径，都落到同一个 `run` | 教程的 Agent 是同步的，本项目 Agent 是异步的；没有异步出口时同步工具（网络搜索、计算）会阻塞事件循环 |
| D21 | `tools/base.py`、`tools/registry.py`、`tools/builtin/calculator.py` | 7.5.1/7.5.2 参数解析硬编码在 Agent 里（`if tool_name == 'calculator'`）；`my_calculator` 经 `register_function` 注册，用裸 `except:` 把错误吞成字符串 | 参数声明与校验由 `Tool`/`ToolParameter` 声明、`Tool.validate` 执行；`ToolRegistry` 接受「单必填参数 → 裸字符串、多参数 → 字典」；计算器做成 `Tool` 子类，白名单 AST 求值，越界/非法一律抛 `ToolException` | 教程写法要求每加一个工具就改 Agent 代码；计算器需要规模保护与可定位的错误（D6/D7 的落地） |
| D22 | `tools/builtin/search.py` | 7.5.3 依赖 `tavily` / `google-search-results` SDK，`print` 打日志，`except Exception as e: return f"❌ {e}"` 吞错 | 只用运行时已有的 `httpx` 直连 Tavily / SerpAPI REST；日志走 `logging`；失败抛 `ToolException`（hybrid 在 context 里带 `attempts`） | 不新增运行时依赖（AGENTS §3），且错误必须可被调用方区分处置 |
| D23 | `tools/chain.py` | 7.5.4 `ToolChain` 用 `List[Dict[str, Any]]` 存步骤（可被外部改写）、`print` 进度、模板变量缺失返回 `"❌ …"` 字符串、只有同步 `execute` | 不可变 `ToolStep` + 只读 `steps`；`logging` 进度；缺失变量抛 `ToolException`（`chain`/`step`/`missing`）；新增 `aexecute` 与 `ToolChainManager.aexecute_chain`，同步/异步共用同一套步进骨架 | 教程正文有代码块但 `code/chapter7/` 无脚本（G1）；可变步骤与吞错字符串不符合本仓库约定 |
| D24 | `tools/async_executor.py` | 7.5.4 `AsyncToolExecutor` 用 `ThreadPoolExecutor` + `__del__` 清理，不校验任务缺键 | 纯 `asyncio` + `Semaphore` 并发上限（信号量按当前事件循环惰性创建，跨 `asyncio.run` 复用安全），显式 `aclose()`/`async with`；任务缺键抛 `ToolException`；`execute_tools_parallel` 支持 `return_exceptions` | `__del__` 触发时机不可靠（资源泄漏）；`asyncio` 原语绑定首次使用它的循环，实测跨两次 `asyncio.run` 会报 `is bound to a different event loop` |
| D25 | `agents/react_agent.py` | 7.4.2 执行历史存实例态 `self.current_history`，步数用尽返回固定文案 `"抱歉，我无法在限定步数内完成这个任务。"` | 执行历史随单次调用传递；新增 `ReActResult`（`answer` + `stop_reason` ∈ finished/max_steps/no_action + 逐步轨迹），`arun()` 仍返回 `str` | 实例态让同一 Agent 并发调用互相污染（D9 的落地）；固定文案无法区分「步数用尽」与「模型没按格式输出」 |
| D26 | `agents/plan_solve_agent.py` | 7.4.4 练习性质，计划解析失败时教程示例静默返回 `None`/空结果 | 计划解析失败抛 `AgentException`（context 带 `agent` 与原始响应）；解析支持 ```python 代码块、裸列表字面量、正文夹杂列表三种形态；新增 `PlanAndSolveResult` | 静默返回空结果让调用方无法区分「没计划」与「模型乱答」；教程正文只给了提示词与片段 |
| D27 | `agents/simple_agent.py`、`agents/react_agent.py` | 7.4.1 `except Exception as e: return f"❌ 工具调用失败：{e}"` | 只捕获 `ToolException`（结构化错误）并以定位信息完整的观察文本回灌模型；其他异常照常上抛 | 吞掉一切异常会掩盖工具实现的 bug；`ToolException` 是「预期内的失败」，回灌给模型让它换策略才是 Agent 语义 |
| D28 | `agents/simple_agent.py` | 7.4.1 `stream_run()` 内部 `print` 进度，且只走无工具的简单对话 | `astream()`（异步、真实增量）与 `stream_run()`（同步包装，教程同名）；不打印；工具循环不参与流式（与教程一致） | `print` 归 `examples/`（D5）；异步是真实链路（D4） |

---

## B. 教程自身的缺口（非我们主动偏离，但要在此登记）

| # | 位置 | 情况 |
|---|---|---|
| G1 | 7.5.4 `ToolChain` / `AsyncToolExecutor` | 正文有完整代码块，但 `code/chapter7/` 目录**无对应文件**，无教程版本可对照运行 |
| G2 | 7.4.5 `FunctionCallAgent` | 正文介绍，但属 `hello-agents` 0.2.8 之后形态，与正文自述基线 `0.1.1` 冲突；`code/chapter7/` 无对应文件。本项目 P1~P5 不纳入 |
| G3 | 第 8 章起 | 教程代码为扁平 `NN_xxx.py` 演示脚本，无包结构；落到 `src/myagent/` 的模块归属需逐章在对话中切分 |
