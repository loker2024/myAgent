# myagent

跟练 [Hello-Agents](https://github.com/datawhalechina/hello-agents) 教程（自第 7 章「构建你的智能体框架」起），以现代 Python 写法重构的智能体框架。

- 教程基线快照与来源 commit：`reference/hello-agents/SOURCE.md`
- 工程约定（**改代码前必读**）：[`AGENTS.md`](./AGENTS.md)
- 与教程的偏离清单：[`docs/deviations.md`](./docs/deviations.md)

## 安装

```bash
uv sync
```

## 常用命令

```bash
uv run ruff format .   # 格式化
uv run ruff check .    # lint
uv run mypy src tests examples   # 类型检查
uv run pytest          # 测试（离线，无需 API key）
```

`examples/` 下的演示脚本需要 `.env`（`LLM_MODEL_ID` / `LLM_API_KEY` / `LLM_BASE_URL`），
在仓库根目录执行，脚本自身不取命令行参数：

```bash
uv run python -X utf8 examples/ch7/01_llm_demo.py        # 交互式对话：流式 + 非流式
uv run python -X utf8 examples/ch7/02_simple_agent.py    # SimpleAgent：带工具的对话
uv run python -X utf8 examples/ch7/03_react_agent.py     # ReActAgent：推理—行动轨迹
uv run python -X utf8 examples/ch7/04_reflection_agent.py # ReflectionAgent：生成—反思—改进
uv run python -X utf8 examples/ch7/05_plan_solve_agent.py # PlanAndSolveAgent：计划 + 逐步执行
uv run python -X utf8 examples/ch7/06_tools.py           # 工具注册表：计算器 + 函数工具（无需 LLM）
uv run python -X utf8 examples/ch7/07_tool_chain.py      # 工具链 + 并发执行器（无需 LLM）
```

`06` / `07` 不调用大模型，因此不需要 `.env`；`02`~`05` 的交互输入以 `exit` / `quit` / Ctrl-D 结束，
非 tty（管道/重定向）下读到 EOF 也会正常退出。

## 进度

- 各 Part 的完成情况与验收证据：[`CHANGELOG.md`](./CHANGELOG.md)
- Part 边界与产出映射（第 7 章 P1~P5）：[`AGENTS.md` §6](./AGENTS.md)

## 许可

`reference/hello-agents/` 内为上游教程内容的只读快照，遵循 CC BY-NC-SA 4.0，仅供学习参考，不得用于商业用途。
