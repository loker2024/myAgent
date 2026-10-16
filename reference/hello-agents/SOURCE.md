# 上游快照来源

本目录是 [Hello-Agents](https://github.com/datawhalechina/hello-agents) 教程内容的**只读快照**，
用途仅有三个：对照实现、比对偏离、离线查阅（见 `AGENTS.md` §1）。

**禁止**修改、格式化、重命名其中任何文件；**禁止**在其中添加文件（除重新抓取时更新本文件）；
**禁止**从 `reference/` 直接被 `src/` 导入 —— 它是参考物，不是依赖。

## 抓取信息

| 项 | 值 |
|---|---|
| 上游仓库 | <https://github.com/datawhalechina/hello-agents> |
| 在线正文 | <https://hello-agents.datawhale.cc> |
| commit | `78dcd8f4821c96ba52ed60d72d523a7173b0206a` |
| commit 日期 | 2026-10-12 |
| 抓取日期 | 2026-10-12 |
| 收录范围 | `code/chapter4/`、`code/chapter7/`、`docs/chapter7/` |

重新抓取时更新本表的 commit、commit 日期与抓取日期。

## 收录方式的已知缺口

上游正文与 `code/chapterN` 目录**不等价**：例如 7.5.4 的 `ToolChain`、`AsyncToolExecutor`
只存在于正文代码块中，`code/chapter7/` 没有对应文件。此类缺口以正文代码块为准实现，
并在 `docs/deviations.md` 记为「教程无配套脚本」。

## 许可证

上游内容采用 **CC BY-NC-SA 4.0**（Attribution-NonCommercial-ShareAlike 4.0 International，
见上游仓库根目录 `LICENSE.txt`）。本快照按原许可保留：

- **署名**：版权归 Hello-Agents 项目及其贡献者所有。
- **非商业性使用**：不得用于商业用途。
- **相同方式共享**：基于本快照的演绎作品必须以相同许可分发。

本项目自身代码（`src/`、`tests/`、`examples/`、`docs/`）是与本快照相互独立的作品，
第三方内容声明见仓库根部 `NOTICE.md`。
