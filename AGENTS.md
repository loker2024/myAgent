# AGENTS.md

本文件是本仓库的**唯一权威工程约定**。任何改动先读本文件；与本文件冲突的做法一律视为错误。

---

## 1. 项目定位

**跟练并优化 Hello-Agents 教程，构建自己的智能体框架 `myagent`。**

- 教程正文：<https://github.com/datawhalechina/hello-agents>（在线阅读 <https://hello-agents.datawhale.cc>）
- 框架参考实现：<https://github.com/jjyaoao/HelloAgents>
- 起点是教程**第 7 章「构建你的智能体框架」**，之后逐章推进到第 16 章。
- 本项目**不是**教程的 fork，也**不是**教程代码的搬运：以教程正文与 `code/chapterN` 脚本为实现蓝本，**修掉其中的过时语法与设计缺陷**，产出可维护版本。

### 基线锁定

- 教程侧基线：`reference/hello-agents/SOURCE.md` 记录的 commit。
- 教程正文自述基线是 `hello-agents==0.1.1`；第 7 章正文 7.4.5 介绍的 `FunctionCallAgent` 属于 0.2.8 之后引入的形态，**不纳入 P1~P5**。要用它必须先在对话中显式确认。
- 教程正文与 `code/chapterN` 目录**不等价**：例如 7.5.4 的 `ToolChain`、`AsyncToolExecutor` 只在正文代码块里存在，`code/chapter7/` 没有对应文件。遇到这种缺口，以正文代码块为准实现，并在 `docs/deviations.md` 记录「教程无配套脚本」。

### `reference/` 规则

`reference/hello-agents/` 是上游内容的**只读快照**，用途仅有三个：对照实现、比对偏离、离线查阅。

- **禁止**修改、格式化、重命名其中任何文件。
- **禁止**在其中添加文件（除重新抓取时更新 `SOURCE.md`）。
- **禁止**从 `reference/` 直接被 `src/` 导入 —— 它是参考物，不是依赖。
- 重新抓取时更新 `SOURCE.md` 的 commit 与日期。上游内容为 CC BY-NC-SA 4.0，仅作学习参考，**不得**用于商业用途。

---

## 2. 目录结构与职责

```
myAgent/
├── AGENTS.md                    # 本文件（稳定约定）
├── CHANGELOG.md                 # 交付历史与验收证据（随每次 Part 交付追加）
├── README.md                    # 面向人的入口与使用说明
├── pyproject.toml               # uv 管理；ruff/mypy/pytest 配置集中于此
├── src/myagent/                 # 唯一包（src 布局，uv_build 后端）
│   ├── core/                    # 底座：llm / message / config / agent / exceptions / _sync(同步包装)
│   ├── agents/                  # 各范式实现：simple / react / reflection / plan_solve
│   └── tools/                   # 工具系统：base / registry / chain / async_executor / builtin
├── examples/ch{N}/              # 面向真实服务的演示脚本（需 .env；无 key 时的验证方式见 §7.3）
├── tests/                       # pytest；假 provider，离线必须全绿
├── docs/deviations.md           # 与教程的偏离清单
└── reference/hello-agents/      # 教程只读快照（见 §1）
```

分层依赖方向**单向**：`tools` ← `core` ← `agents`。即 `core` 不得导入 `agents`，`agents` 不得互相导入（`reflection_agent` 复用 `react_agent` 的循环时，把公共部分下沉到 `core`）。

---

## 3. 技术栈与命令

- Python **3.12**（`requires-python = ">=3.12"`，与教程的 `>=3.10` 不同，见 deviations）。
- 运行时依赖**只有两个**：`httpx`（异步 HTTP/SSE）、`python-dotenv`（读 `.env`）。
- 开发依赖：`pytest`、`pytest-asyncio`、`ruff`、`mypy`。
- 包管理统一 `uv`。

```bash
uv sync                                  # 安装依赖
uv run ruff format .                     # 格式化
uv run ruff check .                      # lint
uv run mypy src                          # 类型检查
uv run pytest                            # 测试（离线，无需 API key）
uv run python -X utf8 examples/ch7/01_llm_demo.py   # 交互式对话演示（需 .env）
```

**引入任何新的运行时依赖必须先在对话中确认。**

`reference/` 已在 `ruff`(`extend-exclude`) 与 `mypy`(`exclude`) 中排除：快照里就是教程原始的过时写法，既不能改也不能被 lint 判定为错误。任何"修 ruff 报错"的动作都不得指向 `reference/`。

---

## 4. 语言与写法约定

### 禁止回归到教程的过时写法

| 教程写法 | 本项目写法 |
|---|---|
| `from typing import Optional, List, Dict, Iterator, Tuple` | `X \| None`、`list[...]`、`dict[...]`、`collections.abc.Iterator` |
| `pydantic.BaseModel` + `.dict()` | `@dataclass` + `to_dict()`（不引入 pydantic） |
| `openai.OpenAI` / `openai.AsyncOpenAI` | `httpx.AsyncClient`，在 `core/llm.py` 内做 provider 适配 |
| `datetime.now()` | `datetime.now(UTC)` |
| `timestamp: datetime = None` | `timestamp: datetime \| None = None` |
| `print(...)` 当日志/进度 | `logging`（演示脚本可 `print` 面向人读的结果） |
| `except Exception as e: return f"❌ {e}"` | 抛分层异常，或返回结构化错误结果；**禁止**把异常吞成字符串 |
| 硬编码工具名分支（`if tool_name == "calculator"`） | 一切经 `ToolRegistry`，工具自带参数校验 |
| 裸 `dict` 拼 messages | 统一 `core.message.Message`，仅在 provider 边界转协议格式 |
| `typing.Iterator[str]` 作流式返回 | `collections.abc.AsyncIterator[str]` |
| 同一调用两条路径（`registry.execute_tool` vs `tool.run`） | 只有 `ToolRegistry.execute_tool` 是对外调用路径 |

- 标识符英文，**docstring 与注释用中文**（与教程一致）。
- 每个公开函数/方法必须有类型注解；`mypy` 必须通过。
- 行宽 100，由 `ruff format` 决定，不手工对齐。

### 异步模型

教程全部示例是同步的（`agent.run(...)`、`llm.think(...)`），本项目**异步是唯一真实实现**：

- `async def arun(...)` 承载真实逻辑；`def run(...)` 只是同步包装（内部 `asyncio.run(arun(...))`，检测到已在运行的事件循环时**明确抛错**，不得静默）。
- 理由：既保住"对外 API 对齐教程"（教程示例可原样运行），又不放弃异步链路。
- **同步包装的连接资源必须与事件循环同生命周期**：httpx 连接池绑定在创建连接的那个事件循环上，同步接口**禁止**每次调用都新建循环（否则第二次调用会 `RuntimeError: Event loop is closed`）。做法：同步入口独占一个长期复用的事件循环与客户端（见 `core/llm.py::_sync_resources`），由 `close()` 在同步上下文回收；异步池由 `await aclose()` 回收。

### 异常

保留教程的异常类名（`HelloAgentsException` / `LLMException` / `AgentException` / `ConfigException` / `ToolException`），但：

- **禁止**空 `pass` 体 —— 每个异常必须携带定位信息（provider、tool 名、模型名等）。
- 网络与 provider 错误必须能被上层区分为**可重试**与**不可重试**。

---

## 5. 每个 Part 的工作流（强制）

用户会说"开始 P{N}"。收到后**按此顺序**执行，不得跳步：

1. **读教程正文**：读 `reference/hello-agents/docs/chapter7/` 中对应小节，列出该 Part 要实现的所有类与函数签名。
2. **读教程代码**：读 `reference/hello-agents/code/chapter{4,7}/` 中对应文件；若正文有代码块而目录无文件，以正文为准。
3. **输出契约清单**：把接口签名（类名、方法名、参数、返回类型、异常）先给用户过一遍，**确认后再写代码**。
4. **实现**到 `src/myagent/`，按 §4 修正过时写法。
5. **测试**：`tests/` 下用假 provider 覆盖行为、边界、错误路径；禁写只验证"不抛异常"或"转发调用"的测试。
6. **演示脚本**：`examples/ch7/NN_<name>.py`，面向真实服务、可独立运行、输出面向人读。
   - 文件顶部必须用注释写明启动命令（在项目根目录执行的完整命令行）；脚本自身不得依赖命令行参数。
   - 需要人参与的场景（对话、工具选择、Agent 体验）必须支持**真实 stdin 输入**并保留多轮上下文，
     给出明确退出方式（`exit` / `quit` / Ctrl-D）；**禁止**把问题硬编码在脚本里冒充"交互"。
   - 非 tty（管道 / 重定向）下也要能读到 EOF 并正常退出，便于无人值守验收。
7. **记录偏离**：任何与教程不同的地方追加到 `docs/deviations.md`（位置 / 教程写法 / 本项目写法 / 原因）。
8. **记录交付**：在 `CHANGELOG.md` 追加本 Part 条目（新增/变更/验收证据）。
9. **验收**：跑 `ruff format` `ruff check` `mypy` `pytest` + 按 §7.3 的方式实际运行该 Part 的演示脚本，观察输出。

**禁止**：跳过 §5.1/5.2 凭印象实现；只交付"能编译的骨架"；用假 fallback、占位实现或 `TODO` 糊弄；把异常路径留给后续 Part。

---

## 6. Part 路线图

### 第 7 章（已切分，逐个交付）

| Part | 教程小节 | 产出 |
|---|---|---|
| P1 | 7.2 / 7.2.1 / 7.2.2 / 7.2.3 多提供商、本地模型、自动检测 | `core/llm.py` |
| P2 | 7.3.1 Message、7.3.2 Config、7.3.3 Agent 抽象基类 | `core/message.py`、`core/config.py`、`core/agent.py`、`core/exceptions.py`、`core/_sync.py` |
| P3 | 7.4.1~7.4.4 四种范式框架化 | `agents/{simple,react,reflection,plan_solve}_agent.py` |
| P4 | 7.5.1 Tool 基类与注册机制、7.5.2 calculator、7.5.3 多源搜索 | `tools/base.py`、`tools/registry.py`、`tools/builtin/{calculator,search}.py` |
| P5 | 7.5.4 ToolChain + AsyncToolExecutor | `tools/chain.py`、`tools/async_executor.py` |

上表只定义 Part 的**边界与产出**（约定，改动需在对话中确认）。各 Part 的完成状态与验收证据见 `CHANGELOG.md` —— 历史不写进本文件。

### 第 8 章及以后（按章节推进，届时再切 Part）

第 8 章记忆与检索 → 第 9 章上下文工程 → 第 10 章通信协议（MCP/A2A/ANP）→ 第 11 章 Agentic RL → 第 12 章性能评估 → 第 13~16 章综合案例。

> 注意：第 8 章起的教程代码是扁平的 `NN_xxx.py` 演示脚本，落在框架上的模块归属需届时在对话中切分后再实现。

---

## 7. 验收门槛（Definition of Done）

一个 Part 只有全部满足才允许报告完成：

1. `uv run ruff format --check .`、`uv run ruff check .`、`uv run mypy src tests examples` 全部通过。
2. `uv run pytest` 全绿，且**离线**（不依赖真实 API key）。
3. 该 Part 的 `examples/ch7/NN_*.py` **实际运行过**并观察到预期输出（不是"应该能跑"）。允许且仅允许两种方式：
   - **真服务**：仓库根目录有 `.env` 且凭据可用，脚本直连真实 provider；报告中写明跑的哪个 provider 与模型。
   - **无 key 时**：把 `LLM_BASE_URL` 指向本地 OpenAI 兼容 stub（`127.0.0.1`）驱动**同一个脚本**跑通；报告中**必须**标注"未使用真实 API key"。
   禁止把 stub 运行表述为"真 API 跑通"；禁止在没有凭据时假设真服务可用。
4. `docs/deviations.md` 已记录该 Part 的全部有意偏离。
5. `CHANGELOG.md` 已追加本 Part 条目（含新增/变更与本次验收证据）。

---

## 8. 禁止事项（汇总）

- 修改或导入 `reference/` 内容。
- 新增运行时依赖而未先确认。
- 照抄教程的过时写法与设计缺陷（见 §4 表）。
- 为空 `pass` 的异常类、吞异常转字符串、无类型注解的公开接口。
- 交付占位实现、假 fallback、`TODO: implement`、未接线的死配置。
- 把"能编译"或"测试通过"当成"功能可用"：必须实际跑过演示脚本。
- 在 `examples/` 与 `tests/` 之间混职责：需要真 API key 的进 `examples/`，可离线断言的进 `tests/`。
- 把交付历史、Part 进度状态、单次命令输出写进本文件 —— 那属于 `CHANGELOG.md`；本文件只放稳定约定。
